"""Playground em Streamlit para testar o poder generativo do modelo.

Execute com::

    poetry run qa-embrapa playground

ou diretamente::

    poetry run streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import streamlit as st

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.config import (  # noqa: E402  (após configurar as variáveis de ambiente)
    COMPARISON_MARKDOWN_FILE,
    MLFLOW_CHAMPION_ALIAS,
    QA_DATASET_FILE,
)
from app.model_registry import champion_uri  # noqa: E402
from app.models import (  # noqa: E402
    AVAILABLE_MODELS,
    DEFAULT_MODEL_KEY,
    generation_config_for,
    resolve_spec,
)
from app.topics import TOPIC_LABELS  # noqa: E402

PAGE_TITLE = "O produtor pergunta, a Embrapa responde — Playground"

#: Perguntas de partida oferecidas na interface.
EXAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("Soja", "Qual é a melhor época de semeadura da soja?"),
    ("Milho", "Qual o espaçamento recomendado para o plantio de milho?"),
    ("Gado de Leite", "O que é mastite e como preveni-la no rebanho?"),
    ("Mandioca", "Como escolher as manivas para o plantio da mandioca?"),
    ("Hortas", "Como preparar o solo de uma horta doméstica?"),
    ("Banana", "Quais os sintomas do mal-do-panamá na bananeira?"),
    ("Citros", "Como controlar o greening nos pomares de citros?"),
    ("Café", "Quando devo irrigar a lavoura em período de estiagem?"),
]


# ---------------------------------------------------------------------------
# Recursos em cache
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Carregando o modelo ajustado ...")
def load_generator(model_dir: str, model_key: str):
    """Carrega o modelo uma única vez por sessão do servidor.

    ``model_key`` participa da chave de cache: trocar de modelo na barra
    lateral precisa recarregar os pesos, e dois modelos podem apontar para
    o mesmo tipo de referência (um URI do MLflow, por exemplo).
    """
    from app.generate import QAGenerator

    return QAGenerator(model_dir, fallback_to_base=True, spec=model_key)


@st.cache_data(show_spinner=False)
def load_reference_index(path: str) -> dict[str, dict[str, str]]:
    """Índice pergunta -> resposta original, para comparação lado a lado."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return {}
    index: dict[str, dict[str, str]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = " ".join(record["question"].lower().split())
            index[key] = {
                "answer": record["answer"],
                "topic": record["topic"],
                "source": record["source_pdf"],
            }
    return index


@st.cache_data(show_spinner=False, ttl=60)
def load_registry_versions(registered_model: str) -> list[dict[str, object]]:
    """Versões disponíveis no Model Registry do MLflow, para um modelo."""
    try:
        from app.model_registry import list_registered_versions

        return list_registered_versions(registered_model)
    except Exception:  # pragma: no cover - MLflow pode estar indisponível
        return []


@st.cache_data(show_spinner=False)
def load_evaluation_summary(path: str) -> dict[str, object]:
    """Carrega o resumo do relatório de avaliação, se existir."""
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dataset_questions(path: str) -> list[tuple[str, str]]:
    """Amostra de perguntas reais do conjunto, para o botão "sortear"."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []
    questions: list[tuple[str, str]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                questions.append((record["topic"], record["question"]))
    return questions


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def sidebar_controls():
    """Desenha a barra lateral e devolve os parâmetros escolhidos."""
    st.sidebar.header("Modelo")
    chaves = list(AVAILABLE_MODELS)
    model_key = st.sidebar.selectbox(
        "Modelo pré-treinado",
        chaves,
        index=chaves.index(DEFAULT_MODEL_KEY),
        format_func=lambda chave: (
            f"{AVAILABLE_MODELS[chave].label} "
            f"({AVAILABLE_MODELS[chave].parameters})"
        ),
        help=(
            "Os dois modelos foram ajustados sobre a mesma coleção. Troque "
            "aqui para comparar as respostas lado a lado."
        ),
    )
    spec = resolve_spec(model_key)
    st.sidebar.caption(f"`{spec.hf_name}` — {spec.summary}")

    origens = {
        "Diretório local": str(spec.output_dir),
        f"MLflow — alias '{MLFLOW_CHAMPION_ALIAS}'": champion_uri(
            spec.registered_model
        ),
    }
    versoes = load_registry_versions(spec.registered_model)
    for item in versoes:
        etiqueta = f"MLflow — versão {item['version']}"
        if item["aliases"]:
            etiqueta += f" ({', '.join(item['aliases'])})"
        origens[etiqueta] = item["uri"]

    escolha = st.sidebar.selectbox(
        "Origem dos pesos",
        list(origens),
        index=1 if versoes else 0,
        help=(
            "O playground pode carregar os pesos de um diretório local ou de "
            "uma versão registrada no Model Registry do MLflow."
        ),
    )
    model_dir = st.sidebar.text_input(
        "Referência do modelo",
        value=origens[escolha],
        key=f"ref-{model_key}",
        help="Caminho local ou URI do MLflow (models:/... ou runs:/...).",
    )

    st.sidebar.header("Tema (cultura)")
    topics = ["(nenhum)"] + sorted(TOPIC_LABELS.values())
    topic = st.sidebar.selectbox(
        "Cultura da pergunta",
        topics,
        index=topics.index("Soja") if "Soja" in topics else 0,
        help=(
            "A mesma pergunta tem respostas diferentes em cada cultura. "
            "O tema entra no prompt do modelo."
        ),
    )

    st.sidebar.header("Decodificação")
    mode = st.sidebar.radio(
        "Estratégia",
        ["Busca em feixe (fiel)", "Amostragem (criativa)"],
        help=(
            "A busca em feixe é determinística e mais fiel ao material "
            "técnico. A amostragem produz texto mais original, com maior "
            "risco de imprecisão."
        ),
    )
    creative = mode.startswith("Amostragem")

    padrao = generation_config_for(spec)
    if creative:
        temperature = st.sidebar.slider("Temperatura", 0.3, 1.5, 0.9, 0.05)
        top_p = st.sidebar.slider("top-p (núcleo)", 0.5, 1.0, 0.92, 0.01)
        top_k = st.sidebar.slider("top-k", 0, 200, 50, 5)
        num_beams = 1
    else:
        # O limite superior é menor no modelo de 4B: cada feixe replica o
        # cache de atenção, e a GPU de referência não comporta muitos.
        maximo = 4 if spec.is_causal else 8
        num_beams = st.sidebar.slider(
            "Número de feixes", 1, maximo, min(padrao.num_beams, maximo), 1
        )
        temperature, top_p, top_k = 1.0, 1.0, 0

    max_new_tokens = st.sidebar.slider("Tamanho máximo (tokens)", 32, 512, 256, 16)
    repetition_penalty = st.sidebar.slider(
        "Penalidade de repetição", 1.0, 2.0, 1.2, 0.05
    )
    variants = st.sidebar.slider(
        "Respostas alternativas",
        1,
        4,
        1,
        help="Gera múltiplas respostas para a mesma pergunta.",
    )

    config = generation_config_for(
        spec,
        max_new_tokens=max_new_tokens,
        num_beams=max(num_beams, variants) if not creative else 1,
        do_sample=creative,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_return_sequences=variants,
    )
    return config, (None if topic == "(nenhum)" else topic), model_dir, spec


def render_metrics_panel(spec) -> None:
    """Mostra as métricas do relatório de avaliação, quando disponível."""
    report = load_evaluation_summary(str(spec.evaluation_report_file))
    if not report:
        st.info(
            "Nenhum relatório de avaliação encontrado para "
            f"**{spec.label}**. Execute "
            f"`poetry run qa-embrapa evaluate --model {spec.key}`."
        )
        return

    st.caption(
        f"Modelo avaliado: `{report.get('model')}` — "
        f"{report.get('test_examples')} exemplos do conjunto de teste."
    )
    for name, data in report.get("presets", {}).items():
        with st.expander(f"Decodificação: {name}", expanded=name == "beam_search"):
            quality = data["quality"]
            originality = data["originality"]
            columns = st.columns(4)
            columns[0].metric("ROUGE-L", f"{quality['rougeL']:.1f}")
            columns[1].metric("ROUGE-1", f"{quality['rouge1']:.1f}")
            columns[2].metric("BLEU", f"{quality['bleu']:.1f}")
            columns[3].metric("chrF", f"{quality['chrf']:.1f}")
            columns = st.columns(3)
            columns[0].metric(
                "4-gramas inéditos",
                f"{originality['novel_4gram_ratio_mean'] * 100:.0f}%",
                help="Proporção de 4-gramas gerados ausentes do corpus de treino.",
            )
            columns[1].metric(
                "Maior trecho copiado",
                f"{originality['longest_copied_span_mean']:.1f} palavras",
            )
            columns[2].metric("distinct-2", f"{originality['distinct_2']:.2f}")


def render_mlflow_panel() -> None:
    """Mostra o estado do Model Registry e como abrir a interface do MLflow."""
    st.subheader("Ciclo de vida no MLflow")
    for spec in AVAILABLE_MODELS.values():
        versoes = load_registry_versions(spec.registered_model)
        st.markdown(f"**{spec.label}** — `{spec.registered_model}`")
        if not versoes:
            st.info(
                "Nenhuma versão registrada. Execute "
                f"`poetry run qa-embrapa train --model {spec.key}` para "
                "treinar e registrar o modelo."
            )
            continue
        st.dataframe(
            [
                {
                    "versão": item["version"],
                    "aliases": ", ".join(item["aliases"]) or "—",
                    "execução": item["run_id"],
                    "URI": item["uri"],
                }
                for item in versoes
            ],
            hide_index=True,
            use_container_width=True,
        )
    st.caption(
        "Selecione qualquer uma dessas versões na barra lateral para "
        "comparar o comportamento de duas gerações do modelo."
    )

    st.markdown("""
**Comandos úteis**

```bash
poetry run qa-embrapa mlflow-ui       # interface web em http://localhost:5000
poetry run qa-embrapa models          # modelos e versões registradas
poetry run qa-embrapa promote 2       # aponta o alias champion para a versão 2
```

Na interface do MLflow ficam disponíveis o histórico de perdas por época, a
comparação entre as decodificações avaliadas (cada uma é uma execução
aninhada), a linhagem dos conjuntos de dados e os artefatos de cada etapa.
        """)


def render_comparison_panel() -> None:
    """Mostra o relatório comparativo entre os modelos, quando existir."""
    st.subheader("Comparação entre os modelos")
    if not COMPARISON_MARKDOWN_FILE.exists():
        st.info(
            "Nenhuma comparação gravada. Avalie os dois modelos e execute "
            "`poetry run qa-embrapa compare`."
        )
        return
    st.markdown(COMPARISON_MARKDOWN_FILE.read_text(encoding="utf-8"))


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🌱", layout="wide")
    st.title("🌱 O produtor pergunta, a Embrapa responde")
    st.caption(
        "Playground dos modelos em português ajustados sobre a coleção "
        "digital de perguntas e respostas da Embrapa. Escolha o modelo na "
        "barra lateral."
    )

    config, topic, model_dir, spec = sidebar_controls()

    try:
        generator = load_generator(model_dir, spec.key)
    except Exception as error:  # pragma: no cover - depende do ambiente
        st.error(f"Não foi possível carregar o modelo: {error}")
        st.stop()

    st.info(
        f"**{spec.label}** — `{spec.hf_name}` · {spec.parameters} parâmetros · "
        f"{'decoder-only, LoRA em 4 bits' if spec.is_causal else 'encoder-decoder, ajuste completo'}",
        icon="🧠",
    )
    if not generator.is_fine_tuned:
        st.warning(
            "O modelo ajustado não foi encontrado em "
            f"`{model_dir}`; as respostas vêm do modelo base, ainda sem "
            "conhecimento da coleção. Execute "
            f"`poetry run qa-embrapa train --model {spec.key}`."
        )

    tab_playground, tab_metrics, tab_compare, tab_mlflow, tab_about = st.tabs(
        ["Playground", "Avaliação", "Comparação", "MLflow", "Sobre o projeto"]
    )

    with tab_playground:
        if "question" not in st.session_state:
            st.session_state.question = EXAMPLE_QUESTIONS[0][1]

        st.markdown("**Perguntas de exemplo**")
        columns = st.columns(4)
        for index, (example_topic, example_question) in enumerate(EXAMPLE_QUESTIONS):
            column = columns[index % 4]
            if column.button(
                f"{example_topic}",
                key=f"ex-{index}",
                help=example_question,
                use_container_width=True,
            ):
                st.session_state.question = example_question

        dataset_questions = load_dataset_questions(str(QA_DATASET_FILE))
        if dataset_questions and st.button("🎲 Sortear pergunta do conjunto"):
            st.session_state.question = random.choice(dataset_questions)[1]

        question = st.text_area(
            "Pergunta do produtor",
            key="question",
            height=90,
            placeholder="Ex.: Como controlar a lagarta-do-cartucho no milho?",
        )
        generate = st.button("Gerar resposta", type="primary")

        if generate:
            if not question.strip():
                st.warning("Escreva uma pergunta.")
            else:
                with st.spinner("Gerando ..."):
                    answers = generator.answer_variants(question, topic, config=config)
                st.markdown("### Resposta gerada")
                if not answers or not any(answer.strip() for answer in answers):
                    st.warning("O modelo não produziu texto para esta pergunta.")
                for index, answer in enumerate(answers, start=1):
                    if len(answers) > 1:
                        st.markdown(f"**Alternativa {index}**")
                    st.success(answer or "(vazio)")

                reference = load_reference_index(str(QA_DATASET_FILE)).get(
                    " ".join(question.lower().split())
                )
                if reference:
                    with st.expander(
                        "Resposta original da Embrapa (esta pergunta consta da coleção)"
                    ):
                        st.markdown(
                            f"*Tema: {reference['topic']} — fonte: "
                            f"`{reference['source']}`*"
                        )
                        st.write(reference["answer"])
                else:
                    st.caption(
                        "Esta pergunta não consta da coleção: a resposta acima é "
                        "conteúdo inteiramente gerado pelo modelo."
                    )

                with st.expander("Prompt enviado ao modelo"):
                    st.code(generator.prompt_for(question, topic), language="text")

    with tab_metrics:
        st.subheader("Qualidade e originalidade")
        render_metrics_panel(spec)

    with tab_compare:
        render_comparison_panel()

    with tab_mlflow:
        render_mlflow_panel()

    with tab_about:
        st.subheader("Como este playground funciona")
        lista_de_modelos = "\n".join(
            f"- **{item.label}** (`{item.hf_name}`, {item.parameters}) — "
            f"{item.summary}"
            for item in AVAILABLE_MODELS.values()
        )
        st.markdown(f"""
1. **Conversão** — os 20 PDFs da coleção em `data/raw` são convertidos para
   Markdown com a **Docling** (`data/interim`). A Docling reconhece os
   enunciados como títulos de seção e descarta cabeçalhos e rodapés de
   página, o que torna a extração determinística.
2. **Extração** — os pares pergunta/resposta são extraídos para JSON Lines
   em `data/processed`, com correção dos enunciados partidos ou invertidos
   pela diagramação em colunas do original.
3. **Ajuste fino** — o modelo escolhido na barra lateral é ajustado sobre
   os pares extraídos. A estratégia muda com a arquitetura (ver abaixo).
4. **Avaliação** — qualidade por ROUGE/BLEU/chrF e originalidade por
   n-gramas inéditos e maior trecho copiado do material de treino, sempre
   sobre as mesmas perguntas de teste, para que os modelos sejam
   comparáveis.

**Os modelos disponíveis**

{lista_de_modelos}

**Sobre os parâmetros de decodificação**

- *Busca em feixe*: determinística, maximiza a probabilidade da sequência.
  Melhores métricas de qualidade, texto mais próximo do original.
- *Amostragem*: sorteia o próximo token dentro do núcleo de probabilidade
  (`top-p`) reescalado pela `temperatura`. Aumenta a originalidade e o
  risco de erro factual.
- *Penalidade de repetição*: desencoraja a repetição de trechos, defeito
  comum em modelos seq2seq pequenos.

**Aviso** — o conteúdo é gerado automaticamente a partir de material
técnico da Embrapa e pode conter imprecisões. Consulte as publicações
originais antes de aplicar qualquer recomendação na lavoura.
            """)


main()
