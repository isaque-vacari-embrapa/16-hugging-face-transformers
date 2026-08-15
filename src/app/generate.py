"""Inferência com o modelo ajustado.

Concentra o carregamento do modelo e a decodificação, usados tanto pela
linha de comando quanto pelo playground em Streamlit.

O :class:`QAGenerator` atende às duas famílias descritas em
:mod:`app.models` e esconde do chamador a diferença entre elas:

* um *encoder-decoder* recebe o prompt e devolve só a resposta;
* um *decoder-only* continua o próprio prompt, e é preciso descartar os
  tokens de entrada da saída — daí o recorte por ``input_ids.shape[1]`` e o
  preenchimento à esquerda, que alinha o início da geração em todo o lote.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import replace
from pathlib import Path

from app.config import (
    BASE_MODEL_NAME,
    DEFAULT_MODEL_DIR,
    GenerationConfig,
)
from app.models import CAUSAL, render_prompt, resolve_spec, spec_for_directory

logger = logging.getLogger(__name__)

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")


def resolve_device(device: str | None = None) -> str:
    """Escolhe o dispositivo de inferência (``cuda`` quando disponível)."""
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_generated(text: str) -> str:
    """Ajustes cosméticos na saída do modelo.

    O SentencePiece reintroduz espaços antes de pontuação ao decodificar.
    As quebras de linha são preservadas (apenas as sequências de linhas em
    branco são reduzidas): um modelo de instrução responde em parágrafos e
    listas, e achatar tudo em uma linha só tornaria a resposta ilegível no
    playground.
    """
    text = _HORIZONTAL_SPACE_RE.sub(" ", text.strip())
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text.strip()


def trim_to_last_sentence(text: str) -> str:
    """Descarta uma frase final incompleta, quando houver."""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in ".!?":
        return stripped
    matches = list(re.finditer(r"[.!?](?=\s|$)", stripped))
    if not matches:
        return stripped
    return stripped[: matches[-1].end()].strip()


def is_model_directory(path: Path) -> bool:
    """Indica se o diretório contém um modelo carregável.

    Um ajuste completo grava ``config.json``; um ajuste por LoRA grava
    ``adapter_config.json`` e nenhum ``config.json``, porque os pesos base
    continuam no Hub.
    """
    return (path / "config.json").exists() or (path / "adapter_config.json").exists()


#: Marcadores que encerram um turno do modelo em templates de conversa.
TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>")


def resolve_stop_token_ids(tokenizer) -> list[int]:
    """Tokens que devem interromper a geração de um modelo de conversa.

    Existe por um detalhe que custa caro se passar despercebido: o Gemma
    declara ``eos_token_id: [1, 106]`` — o ``<eos>`` e o ``<end_of_turn>``
    —, mas o tokenizador conhece apenas o ``1``. Ao carregar o modelo, o
    ``transformers`` **alinha** a configuração aos valores do tokenizador e
    reduz a lista a ``1``. O ``<end_of_turn>``, que é o token que o modelo
    de fato emite ao terminar a resposta (e o que o treino ensina a emitir),
    deixa de encerrar a geração: a resposta iria até o limite de tokens e
    continuaria com um turno inventado.

    Resolver isto a partir do tokenizador, e não da configuração carregada,
    torna a parada independente do comportamento de alinhamento.
    """
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for marker in TURN_END_TOKENS:
        identifier = tokenizer.convert_tokens_to_ids(marker)
        if identifier is not None and identifier != tokenizer.unk_token_id:
            ids.add(int(identifier))
    return sorted(ids)


def is_hub_checkpoint(reference: str | Path) -> bool:
    """Indica se a referência é o nome de um modelo do catálogo no Hub."""
    try:
        resolve_spec(str(reference))
    except KeyError:
        return False
    return not Path(str(reference)).exists()


def infer_spec(weights: str | Path):
    """Descobre a especificação do modelo a partir dos pesos informados.

    A ordem de tentativas vai do mais confiável ao mais frágil: o nome do
    checkpoint base gravado pelo LoRA, o nome do diretório de saída, o
    nome do próprio checkpoint e, por fim, a arquitetura declarada no
    ``config.json``.
    """
    import json

    path = Path(str(weights))
    adapter = path / "adapter_config.json"
    if adapter.exists():
        base = json.loads(adapter.read_text(encoding="utf-8")).get(
            "base_model_name_or_path", ""
        )
        try:
            return resolve_spec(base)
        except KeyError:
            logger.warning("Checkpoint base desconhecido no adaptador: %s", base)

    por_diretorio = spec_for_directory(path)
    if por_diretorio is not None:
        return por_diretorio

    try:
        return resolve_spec(str(weights))
    except KeyError:
        pass

    config_file = path / "config.json"
    if config_file.exists():
        arquiteturas = json.loads(config_file.read_text(encoding="utf-8")).get(
            "architectures", []
        )
        if any("Seq2Seq" in name or name.startswith("T5") for name in arquiteturas):
            return resolve_spec("ptt5")

    logger.warning(
        "Não foi possível identificar o modelo em %s; assumindo %s.",
        weights,
        BASE_MODEL_NAME,
    )
    return resolve_spec()


#: Serializa a montagem de modelos dentro do processo.
#:
#: Carregar dois modelos ao mesmo tempo corrompe um deles. O
#: ``transformers`` monta o esqueleto sob ``init_empty_weights()`` do
#: ``accelerate``, que substitui ``nn.Module.register_parameter``
#: **globalmente** para criar os tensores no dispositivo ``meta``; só
#: depois materializa os pesos do checkpoint. Um modelo montado em outra
#: thread durante essa janela nasce em ``meta`` e nunca é materializado,
#: e falha ao ser movido para o dispositivo com "Cannot copy out of meta
#: tensor".
#:
#: O playground é onde isso aparece: o Streamlit reexecuta o script em uma
#: nova thread a cada interação, e o ``st.cache_resource`` só serializa
#: chamadas de mesma chave — trocar de modelo enquanto outro ainda carrega
#: dispara as duas montagens em paralelo. A janela é grande justamente na
#: imagem de CPU, onde o Gemma leva dezenas de segundos em float32.
#:
#: Serializar não custa desempenho real: os dois carregamentos disputariam
#: a mesma memória de qualquer forma.
_LOAD_LOCK = threading.Lock()


def load_for_inference(weights: str, spec, device: str):
    """Carrega tokenizador e modelo prontos para gerar texto.

    Um modelo causal ajustado por LoRA é montado em duas partes: os pesos
    base quantizados, baixados do Hub, e os adaptadores gravados pelo
    treinamento. O ``PeftModel`` sobrepõe um ao outro sem materializar uma
    cópia dos pesos.

    A montagem inteira acontece sob :data:`_LOAD_LOCK`; ver a nota lá sobre
    por que dois carregamentos simultâneos se corrompem.
    """
    with _LOAD_LOCK:
        return _build_for_inference(weights, spec, device)


def _build_for_inference(weights: str, spec, device: str):
    """Monta tokenizador e modelo. Chame por :func:`load_for_inference`."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    if spec.family != CAUSAL:
        tokenizer = AutoTokenizer.from_pretrained(weights, legacy=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(weights)
        model.config.use_cache = True
        model.to(device)
        return tokenizer, model

    path = Path(weights)
    adapter = path / "adapter_config.json" if path.is_dir() else None
    base_name = spec.hf_name
    tokenizer_source = weights if path.is_dir() else spec.hf_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Obrigatório para gerar em lote com um modelo decoder-only: com
    # preenchimento à direita, as posições geradas ficariam depois do
    # padding e o modelo continuaria a partir de tokens vazios.
    tokenizer.padding_side = "left"

    quantization = None
    if spec.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=quantization,
        # bfloat16 mesmo em Turing: em float16 os logits do Gemma saem NaN.
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": 0} if device == "cuda" else None,
    )

    if adapter is not None and adapter.exists():
        from peft import PeftModel

        logger.info("Aplicando adaptadores LoRA de %s", path)
        model = PeftModel.from_pretrained(model, str(path))

    model.config.use_cache = True
    if quantization is None and device != "cuda":
        model.to(device)
    return tokenizer, model


class QAGenerator:
    """Gera respostas para perguntas de produtores rurais.

    O modelo é carregado uma única vez; ``answer`` pode ser chamado
    repetidamente com diferentes parâmetros de decodificação.
    """

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        device: str | None = None,
        fallback_to_base: bool = False,
        spec=None,
    ) -> None:
        import torch

        from app.model_registry import is_mlflow_uri, resolve_model_dir

        # Aceita um diretório local, um URI do MLflow (como
        # ``models:/qa-embrapa-ptt5@champion`` ou ``runs:/<id>/modelo``) e
        # o nome de um checkpoint do Hub — este último útil para medir o
        # modelo base, antes de qualquer ajuste, na mesma bancada.
        self.reference = str(model_dir)
        if is_mlflow_uri(model_dir):
            weights = str(resolve_model_dir(model_dir))
            self.is_fine_tuned = True
        else:
            model_path = Path(model_dir)
            weights = str(model_path)
            self.is_fine_tuned = is_model_directory(model_path)
            if not self.is_fine_tuned:
                if is_hub_checkpoint(model_dir):
                    logger.info(
                        "Carregando o checkpoint base %s, sem ajuste fino.",
                        model_dir,
                    )
                elif fallback_to_base:
                    fallback = (
                        resolve_spec(spec).hf_name
                        if spec is not None
                        else (spec_for_directory(model_path) or resolve_spec()).hf_name
                    )
                    logger.warning(
                        "Modelo ajustado ausente em %s; usando o modelo base %s.",
                        model_path,
                        fallback,
                    )
                    weights = fallback
                else:
                    raise FileNotFoundError(
                        f"Modelo ajustado não encontrado em {model_path}. "
                        "Execute `poetry run qa-embrapa train` primeiro."
                    )

        self.source = self.reference if is_mlflow_uri(model_dir) else weights
        self.spec = resolve_spec(spec) if spec is not None else infer_spec(weights)
        self.family = self.spec.family
        self.device = resolve_device(device)
        self.tokenizer, self.model = load_for_inference(weights, self.spec, self.device)
        self.model.eval()
        self.stop_token_ids = (
            resolve_stop_token_ids(self.tokenizer) if self.spec.is_causal else []
        )
        self._torch = torch
        logger.info(
            "Modelo carregado de %s (%s, família %s) em %s",
            self.source,
            self.spec.label,
            self.family,
            self.device,
        )

    @property
    def is_causal(self) -> bool:
        """Indica se o modelo carregado é *decoder-only*."""
        return self.family == CAUSAL

    def prompt_for(self, question: str, topic: str | None = None) -> str:
        """Prompt exatamente como o modelo o recebe — usado no playground."""
        return render_prompt(self.spec, question, topic, tokenizer=self.tokenizer)

    # -- geração ----------------------------------------------------------

    def _generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
        max_source_length: int = 128,
    ) -> list[list[str]]:
        """Executa a decodificação para uma lista de prompts."""
        if self.is_causal:
            # O prompt de conversa já traz o ``<bos>`` do template; deixar o
            # tokenizador acrescentar outro deslocaria a distribuição vista
            # no treino. O limite é maior porque prompt e resposta dividem
            # a mesma janela.
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(max_source_length, 256),
                add_special_tokens=False,
            ).to(self.device)
        else:
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            ).to(self.device)

        kwargs: dict[str, object] = {
            "max_new_tokens": config.max_new_tokens,
            "min_new_tokens": config.min_new_tokens,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "num_return_sequences": config.num_return_sequences,
        }
        if config.do_sample:
            kwargs.update(
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )
        else:
            kwargs.update(
                do_sample=False,
                num_beams=max(1, config.num_beams),
                length_penalty=config.length_penalty,
                early_stopping=config.num_beams > 1,
            )
        if self.is_causal:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            kwargs["eos_token_id"] = self.stop_token_ids

        with self._torch.inference_mode():
            outputs = self.model.generate(**encoded, **kwargs)

        if self.is_causal:
            # A saída de um decoder-only inclui o prompt; só o que vem
            # depois dele é a resposta.
            outputs = outputs[:, encoded["input_ids"].shape[1] :]

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        grouped: list[list[str]] = []
        per_prompt = config.num_return_sequences
        for index in range(len(prompts)):
            chunk = decoded[index * per_prompt : (index + 1) * per_prompt]
            grouped.append([trim_to_last_sentence(clean_generated(t)) for t in chunk])
        return grouped

    def answer(
        self,
        question: str,
        topic: str | None = None,
        config: GenerationConfig | None = None,
        **overrides,
    ) -> str:
        """Gera uma única resposta para a pergunta informada."""
        answers = self.answer_many([(question, topic)], config=config, **overrides)
        return answers[0][0] if answers and answers[0] else ""

    def answer_variants(
        self,
        question: str,
        topic: str | None = None,
        config: GenerationConfig | None = None,
        **overrides,
    ) -> list[str]:
        """Gera todas as variantes pedidas por ``num_return_sequences``."""
        answers = self.answer_many([(question, topic)], config=config, **overrides)
        return answers[0] if answers else []

    def answer_many(
        self,
        questions: list[tuple[str, str | None]] | list[str],
        config: GenerationConfig | None = None,
        batch_size: int = 8,
        **overrides,
    ) -> list[list[str]]:
        """Gera respostas em lote para uma lista de perguntas.

        Cada item pode ser apenas a pergunta ou uma tupla ``(pergunta, tema)``.
        """
        config = config or GenerationConfig()
        if overrides:
            config = replace(config, **overrides)

        pairs: list[tuple[str, str | None]] = [
            (item, None) if isinstance(item, str) else item for item in questions
        ]
        prompts = [self.prompt_for(question, topic) for question, topic in pairs]

        results: list[list[str]] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            results.extend(self._generate(chunk, config))
        return results
