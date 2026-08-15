# syntax=docker/dockerfile:1
#
# Imagem do playground em Streamlit (`src/app/streamlit_app.py`).
#
# A imagem é autossuficiente: sobe sem rede e sem volumes, com as duas
# famílias de modelo prontas para responder. Além das dependências e dos
# pesos ajustados, ela carrega o checkpoint base do Gemma 3, exigido pelo
# modelo Gaia — que é apenas um adaptador LoRA e não guarda os pesos base.
#
#     docker build -t qa-embrapa-playground:cpu .
#     docker run --rm -p 8501:8501 qa-embrapa-playground:cpu
#
# Dois argumentos de construção escolhem o que entra na imagem.
#
# `TORCH_VARIANT` — qual PyTorch instalar:
#
#   cpu   (padrão) rodas de CPU. Imagem ~12 GB, roda em qualquer máquina.
#         O PTT5 responde em segundos; o Gemma carrega em float32 e gera a
#         ~1,7 token/s, porque a quantização em 4 bits exige GPU.
#   cuda  rodas com CUDA (+4,5 GB de bibliotecas `nvidia-*`). Com
#         `--gpus all`, o Gemma é carregado em NF4 pelo `bitsandbytes` e
#         gera a ~9 tokens/s — 5,5× mais rápido. Exige o NVIDIA Container
#         Toolkit no hospedeiro.
#
#     docker build -t qa-embrapa-playground:cuda --build-arg TORCH_VARIANT=cuda .
#     docker run --rm --gpus all -p 8501:8501 qa-embrapa-playground:cuda
#
# `GEMMA_SOURCE` — de onde vêm os pesos base do Gemma 3:
#
#   hub      (padrão) baixa do Hugging Face Hub durante a construção;
#            funciona em qualquer máquina, ao custo de ~8 GB de tráfego.
#   context  copia do cache do Hugging Face do hospedeiro, informado como
#            contexto adicional — mais rápido em quem já tem o checkpoint:
#
#     docker build -t qa-embrapa-playground:cpu \
#         --build-arg GEMMA_SOURCE=context \
#         --build-context hfcache="$HOME/.cache/huggingface" .
#
# Os pesos ficam em um estágio próprio, independente do PyTorch escolhido:
# construir as duas variantes reaproveita o mesmo download de 8 GB.

#: Nome do checkpoint base no Hub. Precisa ser idêntico ao `hf_name` do
#: modelo Gaia em `src/app/models.py`: é por esse nome que o `transformers`
#: procura os pesos no cache, em tempo de execução.
ARG GEMMA_MODEL=CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it
#: O mesmo repositório no layout de diretórios do cache (`/` vira `--`).
#: O Dockerfile não sabe derivar um do outro, por isso os dois convivem.
ARG GEMMA_CACHE_DIR=models--CEIA-UFG--Gemma-3-Gaia-PT-BR-4b-it
ARG GEMMA_SOURCE=hub
ARG TORCH_VARIANT=cpu

# ---------------------------------------------------------------------------
# Estágio 1 — dependências
# ---------------------------------------------------------------------------
# As versões vêm todas do `poetry.lock`: o `requirements.txt` versionado é o
# resultado de `poetry export --without-hashes`, de modo que a imagem
# instala exatamente as mesmas versões resolvidas pelo Poetry.

FROM python:3.13-slim AS deps-base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# `build-essential` atende às poucas dependências sem roda pré-compilada;
# nada dele é copiado para a imagem final.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app
COPY requirements.txt ./

# Variante de CPU: descarta as bibliotecas CUDA (`nvidia-*`, `triton`), que
# são a maior parte do ambiente e não têm utilidade sem uma GPU visível ao
# contêiner, e troca o PyTorch pela roda `+cpu` correspondente.
FROM deps-base AS deps-cpu
RUN sed -E \
        -e '/^(nvidia-|cuda-|triton==)/d' \
        -e 's/^(torch|torchvision)==([0-9][^ ;]*)/\1==\2+cpu/' \
        requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-cpu.txt

# Variante CUDA: o `requirements.txt` intacto. As rodas `nvidia-*` já
# trazem o runtime CUDA de que o `bitsandbytes` precisa para quantizar em
# 4 bits; do hospedeiro vem só o driver, injetado pelo Container Toolkit.
FROM deps-base AS deps-cuda
RUN pip install --no-cache-dir -r requirements.txt

# Estágio de dependências efetivamente usado.
FROM deps-${TORCH_VARIANT} AS builder

# O pacote da aplicação é instalado em modo editável apontando para
# `/app/src`. Isso mantém `PROJECT_ROOT` (em `app/config.py`, resolvido a
# partir do próprio arquivo) igual a `/app`, que é onde ficam `models/`,
# `data/` e `reports/` — uma instalação comum o deslocaria para dentro do
# site-packages e quebraria todos os caminhos do projeto.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# ---------------------------------------------------------------------------
# Estágio 2 — checkpoint base do Gemma 3
# ---------------------------------------------------------------------------
# Os pesos ficam no formato de cache do Hugging Face, e não em um diretório
# solto, porque é assim que o `transformers` os encontra pelo nome do
# repositório — que é o que `src/app/models.py` informa. Guardá-los sob
# outro layout exigiria reescrever o caminho do modelo na aplicação.
#
# O estágio parte da imagem limpa, e não do ambiente já instalado, para que
# a camada dos pesos não dependa da variante de PyTorch: as duas imagens
# compartilham os mesmos 8 GB, baixados uma única vez.

FROM python:3.13-slim AS weights-base
ENV HF_HOME=/opt/hf-cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
WORKDIR /tmp/pesos
COPY requirements.txt ./
# A versão sai do próprio `requirements.txt`, para que a ferramenta de
# download seja a mesma que a aplicação usa em tempo de execução.
RUN pip install "$(sed -nE 's/^(huggingface-hub==[^ ;]+).*/\1/p' requirements.txt)"

# Variante "hub": baixa durante a construção.
FROM weights-base AS weights-hub
ARG GEMMA_MODEL
RUN python - "$GEMMA_MODEL" <<'PY'
import sys

from huggingface_hub import snapshot_download

# A documentação do repositório e formatos alternativos de peso não são
# usados na inferência e apenas engordariam a imagem.
snapshot_download(
    sys.argv[1],
    ignore_patterns=["*.md", "*.gguf", "*.pth", "*.bin", "original/*"],
    max_workers=4,
)
PY

# Variante "context": copia do cache do hospedeiro, passado com
# `--build-context hfcache=...`. Evita rebaixar 8 GB já presentes na máquina.
FROM weights-base AS weights-context
ARG GEMMA_CACHE_DIR
COPY --from=hfcache hub/${GEMMA_CACHE_DIR} /opt/hf-cache/hub/${GEMMA_CACHE_DIR}

# Estágio de pesos efetivamente usado.
FROM weights-${GEMMA_SOURCE} AS weights

# ---------------------------------------------------------------------------
# Estágio 3 — execução
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# `libgl1` e `libglib2.0-0` são exigidas pelo OpenCV, que entra como
# dependência da Docling (usada pelo comando `qa-embrapa convert`, também
# disponível nesta imagem).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.streamlit \
    && chown -R app:app /home/app

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/app \
    TOKENIZERS_PARALLELISM=false

# Fragmentação do alocador da GPU: o mesmo ajuste que `streamlit_app.py`
# faz por conta própria, aqui também para os comandos da CLI.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cache do Hugging Face embutido, com o checkpoint base do Gemma 3. O modo
# offline é o padrão para que o contêiner não dependa da rede: sem ele, o
# `transformers` consultaria o Hub a cada carregamento só para conferir a
# revisão. Para permitir o download de outros checkpoints em tempo de
# execução, suba com `-e HF_HUB_OFFLINE=0`.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1

# O rastreamento no MLflow depende do banco em `mlflow/`, que não entra na
# imagem. Sem esta variável, cada carregamento da página tentaria criar um
# banco SQLite vazio. Quem monta o diretório como volume remove a variável
# (`-e QA_EMBRAPA_NO_MLFLOW=`).
ENV QA_EMBRAPA_NO_MLFLOW=1

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /opt/venv /opt/venv
COPY --from=weights --chown=app:app /opt/hf-cache /opt/hf-cache

# O PyTorch com CUDA roteia parte das operações por kernels Triton — no
# Gemma, a codificação rotacional (RoPE) é uma delas —, e o Triton compila
# esses kernels na primeira execução, o que exige um compilador C presente
# na imagem. Sem ele a geração falha com "Failed to find C compiler", já
# depois de carregar os pesos. A variante de CPU não passa por esse caminho
# e não paga os ~100 MB.
#
# Vem depois das cópias pesadas de propósito: assim alternar de variante
# não invalida as camadas do ambiente virtual e do cache de modelos.
ARG TORCH_VARIANT
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends gcc libc6-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Código, pesos ajustados e artefatos consumidos pela interface.
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models
COPY --chown=app:app data/processed ./data/processed
COPY --chown=app:app reports ./reports

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; \
u.urlopen('http://localhost:8501/_stcore/health', timeout=4).read()"

CMD ["streamlit", "run", "src/app/streamlit_app.py"]
