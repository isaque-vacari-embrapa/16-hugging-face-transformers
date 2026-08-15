"""Catálogo dos modelos pré-treinados que o projeto sabe ajustar.

Inicialmente o pipeline foi desenvolvido a partir de um único modelo *encoder-decoder*
(``unicamp-dl/ptt5-v2-base``). Este módulo generaliza essa escolha: cada
modelo disponível é descrito por um :class:`ModelSpec`, que carrega tudo o
que as demais etapas precisam saber para tratá-lo — a família de
arquitetura, o formato do prompt, a estratégia de ajuste fino e os caminhos
dos seus artefatos.

Há duas famílias, e a diferença entre elas atravessa o pipeline inteiro:

``seq2seq``
    Modelos *encoder-decoder* (T5). A entrada vai para o codificador e a
    resposta é gerada pelo decodificador; rótulo e entrada são sequências
    separadas. Cabem inteiros na GPU de referência e são ajustados por
    completo.

``causal``
    Modelos *decoder-only* ajustados para instrução (Gemma). Prompt e
    resposta formam **uma única** sequência, e o rótulo é essa mesma
    sequência com a parte do prompt mascarada. Com 4B de parâmetros não
    cabem em 8 GB para ajuste completo, por isso são carregados
    quantizados em 4 bits e ajustados por LoRA.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from app.config import (
    MODELS_DIR,
    REPORTS_DIR,
    UNKNOWN_TOPIC,
    GenerationConfig,
    TrainingConfig,
    build_prompt,
)

#: Identificadores das duas famílias de arquitetura suportadas.
SEQ2SEQ = "seq2seq"
CAUSAL = "causal"

#: Instrução de sistema usada pelos modelos de instrução (família causal).
#: Ela substitui o papel do ``TASK_PREFIX`` do T5: condiciona o modelo ao
#: domínio e ao registro de linguagem esperado.
SYSTEM_INSTRUCTION = (
    "Você é um especialista da Embrapa. Responda à pergunta do produtor "
    "rural de forma técnica, objetiva e em português do Brasil, com base "
    "nas recomendações da pesquisa agropecuária brasileira."
)


@dataclass(frozen=True)
class ModelSpec:
    """Descrição de um modelo pré-treinado selecionável para o ajuste fino.

    Concentra as diferenças entre os modelos em um único objeto, de modo
    que treino, avaliação, inferência e registro no MLflow recebam a
    especificação e não precisem testar o nome do checkpoint.
    """

    key: str
    hf_name: str
    family: str
    label: str
    parameters: str
    summary: str
    #: Nome do diretório do modelo ajustado, dentro de ``models/``.
    dirname: str
    #: Nome do modelo no Model Registry do MLflow.
    registered_model: str
    #: Carrega os pesos quantizados em 4 bits (NF4) — obrigatório para os
    #: modelos que não cabem na GPU em precisão plena.
    load_in_4bit: bool = False
    #: Ajusta apenas adaptadores LoRA, mantendo os pesos base congelados.
    use_lora: bool = False
    #: Hiperparâmetros que substituem os padrões de ``TrainingConfig``.
    training_overrides: dict[str, object] = field(default_factory=dict)
    #: Parâmetros de decodificação que substituem os de ``GenerationConfig``.
    generation_overrides: dict[str, object] = field(default_factory=dict)

    # -- caminhos ---------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Diretório do modelo ajustado."""
        return MODELS_DIR / self.dirname

    @property
    def training_report_file(self) -> Path:
        """Relatório de treinamento deste modelo."""
        return REPORTS_DIR / f"training_report_{self.key}.json"

    @property
    def evaluation_report_file(self) -> Path:
        """Relatório de avaliação deste modelo."""
        return REPORTS_DIR / f"evaluation_report_{self.key}.json"

    @property
    def samples_file(self) -> Path:
        """Cenário de execução com perguntas de exemplo deste modelo."""
        return REPORTS_DIR / f"generation_samples_{self.key}.md"

    # -- família ----------------------------------------------------------

    @property
    def is_causal(self) -> bool:
        """Indica se o modelo é *decoder-only*."""
        return self.family == CAUSAL

    @property
    def is_seq2seq(self) -> bool:
        """Indica se o modelo é *encoder-decoder*."""
        return self.family == SEQ2SEQ


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------

PTT5 = ModelSpec(
    key="ptt5",
    hf_name="unicamp-dl/ptt5-v2-base",
    family=SEQ2SEQ,
    label="PTT5 v2 base",
    parameters="223M",
    summary=(
        "T5-base re-treinado para o português, com vocabulário SentencePiece "
        "próprio. Ajuste fino completo de todos os parâmetros."
    ),
    dirname="ptt5-qa-embrapa",
    registered_model="qa-embrapa-ptt5",
)

GAIA = ModelSpec(
    key="gaia",
    hf_name="CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it",
    family=CAUSAL,
    label="Gemma 3 Gaia PT-BR 4b it",
    parameters="4,3B",
    summary=(
        "Gemma 3 de 4B ajustado para instrução em português pelo CEIA-UFG. "
        "Carregado em 4 bits (NF4) e ajustado por LoRA, única forma de "
        "treiná-lo nos 8 GB da GPU de referência."
    ),
    dirname="gaia-qa-embrapa",
    registered_model="qa-embrapa-gaia",
    load_in_4bit=True,
    use_lora=True,
    training_overrides={
        # O checkpoint é multimodal (Gemma 3 + torre de visão SigLIP). Sem
        # esta âncora no caminho do módulo, os sufixos ``q_proj``/``v_proj``
        # também casariam com as 162 projeções da torre de visão, que a
        # tarefa de texto nunca exercita.
        "lora_target_regex": (
            r"model\.language_model\.layers\.\d+\."
            r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
        ),
        # A resposta mediana da coleção tem 109 tokens neste vocabulário e
        # 88% cabem em 224; somados aos ~70 do prompt de conversa, a janela
        # de 320 cobre o corpus sem estourar a VRAM no cabeçote de saída,
        # que materializa 262 mil logits por posição.
        "max_target_length": 224,
        "max_sequence_length": 320,
        # LoRA tolera (e precisa de) uma taxa de aprendizado maior que o
        # ajuste completo: só as matrizes de baixo posto são treinadas.
        "learning_rate": 2e-4,
        # Uma época. Medimos 0,58 amostra/s na GPU de referência (bfloat16
        # emulado somado ao custo de desquantizar os pesos a cada passo),
        # o que dá ~3,6 h por época contra 32 min das cinco épocas do T5.
        # São 470 passos de otimização — suficientes para LoRA fixar
        # domínio e formato em um modelo que já responde em português.
        "num_train_epochs": 1.0,
        # Prompt e resposta formam uma sequência só: o lote precisa ser 1
        # para o logit de 262 mil posições de vocabulário caber na VRAM.
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        # Sem isto, os estados do otimizador competem com os pesos pela
        # VRAM já escassa.
        "optim": "paged_adamw_8bit",
        "warmup_ratio": 0.03,
        "early_stopping_patience": 1,
        "logging_steps": 20,
    },
    generation_overrides={
        # Padrão do uso interativo (``ask``, ``samples``, playground). A
        # busca em feixe replica o cache de atenção por feixe, e num modelo
        # de 4B cada feixe a mais custa memória e tempo de resposta: dois
        # feixes é o ponto em que o playground continua responsivo. A
        # avaliação **não** usa este valor — ela aplica as mesmas
        # decodificações aos dois modelos, para que sejam comparáveis.
        "num_beams": 2,
        # O modelo de instrução responde em parágrafos completos, sem a
        # tendência do T5 ajustado a encurtar demais.
        "length_penalty": 1.0,
    },
)

#: Modelos que a aplicação sabe treinar e servir, por identificador curto.
AVAILABLE_MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in (PTT5, GAIA)}

#: Modelo usado quando nenhum é informado.
DEFAULT_MODEL_KEY = PTT5.key


def resolve_spec(reference: str | ModelSpec | None = None) -> ModelSpec:
    """Devolve a especificação a partir da chave curta ou do nome no Hub.

    Aceita ``"gaia"``, ``"CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"`` ou a própria
    especificação, para que a CLI, o playground e os testes possam usar a
    forma mais conveniente em cada caso.
    """
    if isinstance(reference, ModelSpec):
        return reference
    if reference is None:
        return AVAILABLE_MODELS[DEFAULT_MODEL_KEY]

    text = str(reference).strip()
    if text in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[text]
    for spec in AVAILABLE_MODELS.values():
        if text.lower() == spec.hf_name.lower():
            return spec
    conhecidos = ", ".join(
        f"{spec.key} ({spec.hf_name})" for spec in AVAILABLE_MODELS.values()
    )
    raise KeyError(f"Modelo desconhecido: {reference!r}. Disponíveis: {conhecidos}.")


def training_config_for(
    spec: str | ModelSpec | None = None, **overrides: object
) -> TrainingConfig:
    """Monta a configuração de treino do modelo, já com os seus padrões.

    A precedência é: padrões de ``TrainingConfig`` < padrões do modelo
    (``training_overrides``) < o que o chamador informar. É isso que
    permite à CLI aceitar ``--epochs`` sem desfazer o restante do ajuste
    específico de cada arquitetura.
    """
    spec = resolve_spec(spec)
    settings: dict[str, object] = {
        "base_model": spec.hf_name,
        "output_dir": spec.output_dir,
        "family": spec.family,
        "load_in_4bit": spec.load_in_4bit,
        "use_lora": spec.use_lora,
    }
    settings.update(spec.training_overrides)
    settings.update(overrides)
    return replace(TrainingConfig(), **settings)


def generation_config_for(
    spec: str | ModelSpec | None = None, **overrides: object
) -> GenerationConfig:
    """Monta a configuração de decodificação com os padrões do modelo."""
    spec = resolve_spec(spec)
    settings: dict[str, object] = dict(spec.generation_overrides)
    settings.update(overrides)
    return replace(GenerationConfig(), **settings)


def spec_for_directory(model_dir: Path | str) -> ModelSpec | None:
    """Descobre a especificação a partir do diretório de um modelo ajustado.

    Usado pela inferência, que recebe um caminho e precisa saber de que
    família ele é antes mesmo de abrir os pesos.
    """
    name = Path(str(model_dir)).name
    for spec in AVAILABLE_MODELS.values():
        if name == spec.dirname:
            return spec
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def normalize(question: str, topic: str | None = None) -> tuple[str, str]:
    """Padroniza espaços da pergunta e do tema antes de montar o prompt."""
    question = " ".join(str(question).split())
    topic = " ".join(str(topic or UNKNOWN_TOPIC).split()).lower()
    return question, topic


#: Prompt no estilo T5: prefixo de tarefa seguido de tema e pergunta.
seq2seq_prompt = build_prompt


def chat_messages(question: str, topic: str | None = None) -> list[dict[str, str]]:
    """Diálogo no formato de mensagens esperado por um modelo de instrução.

    A instrução de sistema entra como primeira mensagem: o *chat template*
    do Gemma a funde ao primeiro turno do usuário, que é como o modelo foi
    treinado.
    """
    question, topic = normalize(question, topic)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Tema: {topic}\nPergunta: {question}"},
    ]


def causal_prompt(tokenizer, question: str, topic: str | None = None) -> str:
    """Prompt textual do modelo causal, já com os marcadores de turno.

    Delega ao *chat template* do próprio tokenizador em vez de reproduzir
    os marcadores à mão: eles fazem parte do checkpoint e mudam entre
    famílias de modelo.
    """
    return tokenizer.apply_chat_template(
        chat_messages(question, topic),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_prompt(
    spec: ModelSpec, question: str, topic: str | None = None, tokenizer=None
) -> str:
    """Monta o prompt de acordo com a família do modelo.

    O tokenizador só é necessário para a família causal; para o T5 o prompt
    é apenas texto e pode ser montado sem carregar nada.
    """
    if spec.is_causal:
        if tokenizer is None:
            raise ValueError(
                "O prompt de um modelo causal depende do chat template do "
                "tokenizador; informe `tokenizer`."
            )
        return causal_prompt(tokenizer, question, topic)
    return seq2seq_prompt(question, topic)
