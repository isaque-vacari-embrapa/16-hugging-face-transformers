"""Configurações centrais e caminhos do projeto.

Reúne em um único lugar os diretórios de dados, os artefatos de modelo e
os hiperparâmetros padrão, de modo que os demais módulos não precisem
recalcular caminhos relativos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Diretórios
# ---------------------------------------------------------------------------

#: Raiz do repositório (``src/app/config.py`` -> ``src/app`` -> ``src`` -> raiz).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_DIR: Path = PROJECT_ROOT / "mlflow"
#: O registro de modelos (Model Registry) exige um backend com banco de
#: dados; o armazenamento em arquivos puro não o suporta. SQLite atende sem
#: exigir serviço externo.
MLFLOW_DB_FILE: Path = MLFLOW_DIR / "mlflow.db"
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_FILE}"
MLFLOW_ARTIFACT_ROOT: Path = MLFLOW_DIR / "artifacts"
MLFLOW_EXPERIMENT_NAME: str = "qa-embrapa"
#: Nome do modelo no Model Registry e alias apontando para a versão em uso.
MLFLOW_REGISTERED_MODEL: str = "qa-embrapa-ptt5"
MLFLOW_CHAMPION_ALIAS: str = "champion"
#: Subdiretório do artefato pyfunc que guarda os pesos no formato Hugging Face.
MLFLOW_MODEL_ARTIFACT: str = "model_dir"

#: Modelo ajustado padrão (destino do treinamento e origem da inferência).
#: Os caminhos de cada modelo selecionável vivem em :mod:`app.models`; esta
#: constante é o atalho para o modelo usado quando nenhum é informado.
DEFAULT_MODEL_DIR: Path = MODELS_DIR / "ptt5-qa-embrapa"

#: Arquivos gerados pelo pipeline de dados.
QA_DATASET_FILE: Path = PROCESSED_DIR / "qa_embrapa.jsonl"
SPLITS_DIR: Path = PROCESSED_DIR / "splits"
EXTRACTION_REPORT_FILE: Path = REPORTS_DIR / "extraction_report.json"
DATASET_REPORT_FILE: Path = REPORTS_DIR / "dataset_report.json"

#: Relatórios do modelo padrão. Os demais modelos gravam em arquivos com o
#: sufixo da sua chave (ver ``ModelSpec.training_report_file`` e afins).
TRAINING_REPORT_FILE: Path = REPORTS_DIR / "training_report_ptt5.json"
EVALUATION_REPORT_FILE: Path = REPORTS_DIR / "evaluation_report_ptt5.json"
GENERATION_SAMPLES_FILE: Path = REPORTS_DIR / "generation_samples_ptt5.md"

#: Comparação entre os modelos treinados.
COMPARISON_REPORT_FILE: Path = REPORTS_DIR / "model_comparison.json"
COMPARISON_MARKDOWN_FILE: Path = REPORTS_DIR / "model_comparison.md"


def ensure_dirs(*paths: Path) -> None:
    """Cria os diretórios informados, incluindo os pais, se necessário."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Modelo e prompt
# ---------------------------------------------------------------------------

#: Checkpoint pré-treinado com vocabulário português (T5-base, ~223M params).
BASE_MODEL_NAME: str = "unicamp-dl/ptt5-v2-base"

#: Prefixo de tarefa no estilo T5. Mantém o modelo condicionado ao domínio.
TASK_PREFIX: str = "responda a pergunta do produtor rural: "

#: Rótulo usado quando a cultura/tema não é informado.
UNKNOWN_TOPIC: str = "geral"


def build_prompt(question: str, topic: str | None = None) -> str:
    """Monta a entrada do modelo a partir da pergunta e do tema (cultura).

    O tema é incluído no prompt porque a coleção da Embrapa cobre 20
    culturas diferentes e a mesma pergunta ("qual o espaçamento ideal?")
    tem respostas distintas em cada uma delas.
    """
    question = " ".join(question.split())
    topic = " ".join((topic or UNKNOWN_TOPIC).split()).lower()
    return f"{TASK_PREFIX}tema: {topic} | pergunta: {question}"


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros de ajuste fino (fine-tuning) do modelo seq2seq.

    Os valores padrão foram escolhidos para caber em uma GPU de 8 GB
    (NVIDIA Quadro RTX 4000, arquitetura Turing, sem suporte a ``bfloat16``).
    T5 é numericamente instável em ``float16``, por isso o treinamento é
    feito em ``float32`` com lotes pequenos e acumulação de gradiente.

    O porte do modelo base é parte dessa restrição: em ``float32``, os
    pesos e os gradientes de um T5-large (738M parâmetros) já ocupariam
    5,5 GB dos ~6,1 GB livres, e o ajuste fino não cabe nesta placa nem
    com Adafactor e ``gradient_checkpointing``.
    """

    base_model: str = BASE_MODEL_NAME
    output_dir: Path = DEFAULT_MODEL_DIR
    max_source_length: int = 128
    max_target_length: int = 256
    learning_rate: float = 3e-4
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing_factor: float = 0.0
    #: Só é necessário em GPUs com menos de ~6 GB livres; custa ~30% de tempo.
    gradient_checkpointing: bool = False
    seed: int = 42
    early_stopping_patience: int = 2
    logging_steps: int = 50
    optim: str = "adamw_torch"

    # -- modelos causais (decoder-only) ------------------------------------
    #: Família da arquitetura: ``seq2seq`` (T5) ou ``causal`` (Gemma). Ver
    #: :mod:`app.models`.
    family: str = "seq2seq"
    #: Carrega os pesos quantizados em 4 bits (NF4, com dupla quantização).
    #: Único modo de ajustar um modelo de 4B de parâmetros em 8 GB de VRAM.
    load_in_4bit: bool = False
    #: Treina apenas adaptadores LoRA, com os pesos base congelados.
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    #: Projeções lineares que recebem adaptadores. Cobrem atenção e MLP,
    #: que é o que a literatura de LoRA recomenda quando há orçamento.
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    #: Expressão regular que, quando definida, substitui a lista acima.
    #: Necessária em modelos multimodais: casar apenas pelo sufixo
    #: (``q_proj``) alcançaria também a torre de visão, treinando
    #: adaptadores inúteis para uma tarefa que só usa texto.
    lora_target_regex: str | None = None
    #: Comprimento máximo da sequência única (prompt + resposta) usada pelos
    #: modelos causais. Não se aplica ao T5, que separa entrada e rótulo.
    max_sequence_length: int = 512


@dataclass(frozen=True)
class GenerationConfig:
    """Parâmetros de decodificação usados na inferência e no playground.

    ``do_sample`` ligado com ``temperature``/``top_p`` favorece respostas
    originais (criativas); desligado, a decodificação por feixes (beam
    search) favorece fidelidade ao conteúdo técnico.
    """

    max_new_tokens: int = 256
    #: Mínimo baixo de propósito: há respostas legítimas de uma única frase
    #: curta na coleção ("O rizoma da bananeira é seu caule verdadeiro.").
    min_new_tokens: int = 8
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 4
    #: Acima de 1.0 favorece sequências mais longas na busca em feixe. O
    #: modelo ajustado tende a responder em uma única frase, bem mais curto
    #: que as respostas de referência (mediana de 66 palavras); 1.3 aproxima
    #: os comprimentos sem forçar texto de preenchimento.
    length_penalty: float = 1.3
    num_return_sequences: int = 1


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Proporções das partições de treino, validação e teste."""

    validation_size: float = 0.1
    test_size: float = 0.05
    seed: int = 42
    #: Culturas presentes na coleção; preenchido dinamicamente na extração.
    topics: list[str] = field(default_factory=list)
