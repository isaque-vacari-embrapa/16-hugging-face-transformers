"""Ajuste fino (fine-tuning) com o Hugging Face Trainer.

O modelo a ajustar é escolhido pelo usuário entre os descritos em
:mod:`app.models`, e a estratégia de treino muda com a família dele:

**Família ``seq2seq``** (``unicamp-dl/ptt5-v2-base``)
    Um T5-base (~223M parâmetros) re-treinado para o português com
    vocabulário SentencePiece próprio. O vocabulário importa: o do T5
    original fragmenta palavras acentuadas em muitos tokens, o que degrada
    a qualidade e o custo em textos técnicos em português. Cabe inteiro na
    GPU e é ajustado por completo, em ``float32`` — o T5 produz ``NaN`` em
    ``float16``, e a GPU de referência (Quadro RTX 4000, Turing) não tem
    ``bfloat16`` nativo.

**Família ``causal``** (``CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it``)
    Um Gemma 3 de 4,3B ajustado para instrução em português. Em precisão
    plena os pesos ocupariam 17 GB, contra os ~6,9 GB livres da placa; por
    isso é carregado quantizado em 4 bits (NF4) e apenas adaptadores LoRA
    são treinados. A precisão de cálculo é ``bfloat16`` mesmo em Turing,
    onde é emulado: medimos ``NaN`` em todos os logits com ``float16``,
    resultado do intervalo dinâmico com que os pesos do Gemma foram
    treinados.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from app.config import (
    REPORTS_DIR,
    SPLITS_DIR,
    TrainingConfig,
    ensure_dirs,
)
from app.dataset import load_splits, tokenize_splits

logger = logging.getLogger(__name__)

# Reduz a fragmentação do alocador da CUDA. Precisa estar definido antes da
# primeira alocação na GPU: com 8 GB de VRAM (e parte dela ocupada pelo
# servidor gráfico) a margem de memória é estreita.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


#: Capacidade de computação mínima para bfloat16 nativo (Ampere, sm_80).
MIN_BF16_CAPABILITY = (8, 0)


def resolve_precision(family: str = "seq2seq") -> dict[str, bool]:
    """Decide a precisão numérica de acordo com a GPU e a família do modelo.

    ``float16`` nunca é usado. Nenhuma das duas famílias o tolera: o T5
    produz ``NaN`` nessa precisão e o Gemma também — verificamos que, em
    ``float16``, *todos* os logits do Gemma 3 Gaia saem ``NaN`` e a geração
    devolve apenas tokens de preenchimento.

    A diferença está no que se faz quando a GPU não tem ``bfloat16``
    nativo (Turing, sm_75):

    * ``seq2seq`` cai para ``float32``. O ``bfloat16`` emulado da Turing é
      mais lento (medimos ~15 amostras/s contra ~21 em ``float32``) e
      numericamente pior, e o T5-base cabe em ``float32``.
    * ``causal`` fica em ``bfloat16`` de qualquer forma. Um modelo de 4B em
      ``float32`` não caberia na placa, e ``float16`` não é opção — resta o
      ``bfloat16`` emulado, que funciona.
    """
    import torch

    from app.models import CAUSAL

    if not torch.cuda.is_available():
        logger.info("Sem GPU disponível: treinando em float32 na CPU.")
        return {"bf16": False, "fp16": False}

    capability = torch.cuda.get_device_capability(0)
    if capability >= MIN_BF16_CAPABILITY:
        logger.info("GPU %s com bfloat16 nativo: treinando em bf16.", capability)
        return {"bf16": True, "fp16": False}
    if family == CAUSAL:
        logger.info(
            "GPU %s sem bfloat16 nativo, mas o modelo causal exige bf16 "
            "(fp16 produz NaN): treinando em bf16 emulado.",
            capability,
        )
        return {"bf16": True, "fp16": False}
    logger.info(
        "GPU %s sem bfloat16 nativo: treinando em float32 "
        "(fp16 é instável para T5).",
        capability,
    )
    return {"bf16": False, "fp16": False}


def estimate_total_steps(config: TrainingConfig, num_train_examples: int) -> int:
    """Estima o número total de passos de otimização do treinamento."""
    per_epoch = max(
        1,
        math.ceil(
            num_train_examples
            / (config.per_device_train_batch_size * config.gradient_accumulation_steps)
        ),
    )
    return max(1, int(per_epoch * config.num_train_epochs))


def build_training_arguments(
    config: TrainingConfig,
    num_train_examples: int,
    use_mlflow: bool = True,
    precision: dict[str, bool] | None = None,
):
    """Monta os ``Seq2SeqTrainingArguments`` a partir da configuração.

    Com ``use_mlflow``, o Trainer recebe ``report_to=["mlflow"]``: o
    ``MLflowCallback`` do Hugging Face reaproveita a execução já ativa e
    registra automaticamente os hiperparâmetros e as métricas de cada época.

    ``precision`` pode ser reaproveitada de uma chamada anterior a
    :func:`resolve_precision`, que emite log ao decidir.
    """
    from transformers import Seq2SeqTrainingArguments, TrainingArguments

    from app.models import CAUSAL

    precision = precision if precision is not None else resolve_precision(config.family)
    total_steps = estimate_total_steps(config, num_train_examples)
    common: dict[str, object] = dict(
        output_dir=str(config.output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_steps=int(total_steps * config.warmup_ratio),
        label_smoothing_factor=config.label_smoothing_factor,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="linear",
        optim=config.optim,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to=["mlflow"] if use_mlflow else [],
        disable_tqdm=False,
        **precision,
    )
    if config.family == CAUSAL:
        # ``TrainingArguments`` (e não a variante Seq2Seq): um modelo
        # decoder-only não tem ``decoder_input_ids`` nem geração no laço de
        # avaliação — a métrica de seleção é a perda de validação.
        return TrainingArguments(**common)
    return Seq2SeqTrainingArguments(predict_with_generate=False, **common)


def prepare_for_kbit_training(model, config: TrainingConfig) -> None:
    """Prepara um modelo quantizado para receber adaptadores treináveis.

    Faz o mesmo que ``peft.prepare_model_for_kbit_training``, **menos** a
    promoção de todos os parâmetros não quantizados para ``float32``. Essa
    promoção é o que inviabiliza o treino aqui: a matriz de *embeddings* do
    Gemma tem 677M parâmetros (vocabulário de 262 mil × 2560) e sozinha
    passaria de 1,35 GB em ``bfloat16`` para 2,7 GB em ``float32`` — mais
    de um quinto da VRAM da placa, gasto em pesos que permanecem
    congelados. Os adaptadores LoRA continuam em ``float32``, que é onde a
    precisão de fato importa para a estabilidade do otimizador.
    """
    for param in model.parameters():
        param.requires_grad = False

    if config.gradient_checkpointing:
        # Sem isto, o gradiente não atravessa o ponto de checkpoint: a
        # entrada da primeira camada vem de uma tabela de embeddings
        # congelada e chegaria ao backward sem ``requires_grad``.
        model.enable_input_require_grads()


def load_base_model(config: TrainingConfig):
    """Carrega o modelo pré-treinado conforme a família e a quantização.

    Para a família causal aplica a receita QLoRA: pesos congelados em 4
    bits (NF4 com dupla quantização), cálculo em ``bfloat16`` e adaptadores
    LoRA como únicos parâmetros treináveis. Ver
    :func:`prepare_for_kbit_training` para a diferença em relação ao
    utilitário equivalente do ``peft``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    from app.models import CAUSAL

    if config.family != CAUSAL:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
        model.config.use_cache = not config.gradient_checkpointing
        return model

    quantization = None
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info("Carregando %s quantizado em 4 bits (NF4).", config.base_model)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    if config.load_in_4bit:
        prepare_for_kbit_training(model, config)

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        # O peft trata uma string como expressão regular sobre o nome
        # completo do módulo, e uma lista como sufixos a casar.
        alvos = config.lora_target_regex or list(config.lora_target_modules)
        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=alvos,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA aplicado: %s parâmetros treináveis de %s (%.3f%%).",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )
    return model


def load_tokenizer(config: TrainingConfig):
    """Carrega o tokenizador do modelo base, ajustado para o treinamento."""
    from transformers import AutoTokenizer

    from app.models import CAUSAL

    if config.family == CAUSAL:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Na geração o preenchimento à esquerda é obrigatório; no treino,
        # o ``CausalCollator`` preenche à direita por conta própria.
        tokenizer.padding_side = "left"
        return tokenizer
    return AutoTokenizer.from_pretrained(config.base_model, legacy=False)


def describe_environment() -> dict[str, object]:
    """Coleta o ambiente de execução, registrado como parâmetros no MLflow."""
    import torch
    import transformers

    gpu = None
    capability = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "compute_capability": capability,
    }


def train(
    config: TrainingConfig | None = None,
    splits_dir: Path = SPLITS_DIR,
    max_train_examples: int | None = None,
    resume_from_checkpoint: bool = False,
    register_model: bool = True,
    run_name: str = "treino",
    report_path: Path | None = None,
    spec=None,
) -> dict[str, object]:
    """Executa o ajuste fino e grava o modelo final em ``config.output_dir``.

    Todo o treinamento acontece dentro de uma execução do MLflow: os
    hiperparâmetros e as métricas de cada época são registrados pelo
    ``MLflowCallback`` do Trainer, e ao final o modelo é empacotado como
    ``pyfunc`` e registrado no Model Registry (ver :mod:`app.model_registry`).

    Retorna um dicionário com as métricas de treino e validação, também
    gravado no relatório de treinamento do modelo escolhido.
    """
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Trainer,
    )

    from app import tracking
    from app.dataset import CausalCollator
    from app.model_registry import log_and_register_model
    from app.models import CAUSAL, resolve_spec

    config = config or TrainingConfig()
    spec = resolve_spec(spec if spec is not None else config.base_model)
    report_path = report_path or spec.training_report_file
    is_causal = config.family == CAUSAL

    ensure_dirs(config.output_dir, REPORTS_DIR)
    use_mlflow = tracking.configure()
    # O callback do Trainer enviaria os artefatos de cada checkpoint se esta
    # variável estivesse ligada; os pesos vão para o Model Registry uma única
    # vez no fim, o que evita duplicar quase 900 MB por época.
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")

    with tracking.start_run(
        run_name,
        tags={
            "modelo_base": config.base_model,
            "modelo": spec.key,
            "familia": config.family,
            "estrategia": "lora-4bit" if config.use_lora else "ajuste-completo",
        },
    ):
        logger.info("Carregando tokenizador e modelo base: %s", config.base_model)
        tokenizer = load_tokenizer(config)
        model = load_base_model(config)

        splits = load_splits(splits_dir)
        if max_train_examples:
            logger.warning(
                "Modo reduzido: usando apenas %d exemplos de treino.",
                max_train_examples,
            )
            splits["train"] = splits["train"].select(
                range(min(max_train_examples, len(splits["train"])))
            )
            splits["validation"] = splits["validation"].select(
                range(min(max(32, max_train_examples // 8), len(splits["validation"])))
            )

        environment = describe_environment()
        precision = resolve_precision(config.family)
        tracking.log_params(environment, prefix="ambiente.")
        tracking.log_params(
            {
                "modelo": spec.key,
                "familia": config.family,
                "precisao": "bf16" if precision["bf16"] else "float32",
                "quantizacao": "nf4-4bit" if config.load_in_4bit else "nenhuma",
                "adaptadores": "lora" if config.use_lora else "nenhum",
                "exemplos_treino": len(splits["train"]),
                "exemplos_validacao": len(splits["validation"]),
                "lote_efetivo": (
                    config.per_device_train_batch_size
                    * config.gradient_accumulation_steps
                ),
            }
        )
        tracking.log_dataset(
            splits_dir / "train.jsonl", "qa-embrapa-train", "training", "answer"
        )
        tracking.log_dataset(
            splits_dir / "validation.jsonl", "qa-embrapa-validation", "validation"
        )

        tokenized = tokenize_splits(splits, tokenizer, config)
        if is_causal:
            collator = CausalCollator(pad_token_id=tokenizer.pad_token_id)
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                padding="longest",
            )

        args = build_training_arguments(
            config,
            len(tokenized["train"]),
            use_mlflow=use_mlflow,
            precision=precision,
        )
        trainer_class = Trainer if is_causal else Seq2SeqTrainer
        trainer = trainer_class(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience
                )
            ],
        )

        logger.info(
            "Iniciando treinamento: %d exemplos, %s épocas, lote efetivo %d",
            len(tokenized["train"]),
            config.num_train_epochs,
            config.per_device_train_batch_size * config.gradient_accumulation_steps,
        )
        started = time.perf_counter()
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint or None
        )
        elapsed = time.perf_counter() - started

        eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])

        logger.info("Salvando modelo ajustado em %s", config.output_dir)
        model.config.use_cache = True
        # Com LoRA, ``save_model`` grava apenas os adaptadores (~100 MB) e o
        # ``adapter_config.json``, que aponta para o checkpoint base no Hub.
        trainer.save_model(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))

        report: dict[str, object] = {
            "model_key": spec.key,
            "model_label": spec.label,
            "base_model": config.base_model,
            "family": config.family,
            "strategy": "lora-4bit" if config.use_lora else "ajuste-completo",
            "output_dir": str(config.output_dir),
            "elapsed_seconds": round(elapsed, 1),
            "train_examples": len(tokenized["train"]),
            "validation_examples": len(tokenized["validation"]),
            "test_examples": len(tokenized["test"]),
            "environment": environment,
            "hyperparameters": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in asdict(config).items()
            },
            "train_metrics": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in train_result.metrics.items()
            },
            "validation_metrics": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in eval_metrics.items()
            },
            "log_history": trainer.state.log_history,
            # Definido antes da gravação: é o que liga este relatório à
            # execução correspondente no MLflow.
            "mlflow_run_id": tracking.active_run_id(),
        }

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["trainable_parameters"] = trainable
        report["total_parameters"] = sum(p.numel() for p in model.parameters())

        tracking.log_metrics(report["train_metrics"], prefix="final.")
        tracking.log_metrics(report["validation_metrics"], prefix="final.")
        tracking.log_metrics(
            {
                "tempo_treino_segundos": elapsed,
                "parametros_treinaveis": trainable,
                "parametros_totais": report["total_parameters"],
            }
        )

        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Relatório de treinamento gravado em %s", report_path)
        tracking.log_artifact(report_path, artifact_path="relatorios")

        if register_model:
            model_uri = log_and_register_model(
                config.output_dir,
                registered_model_name=spec.registered_model,
                extra_metadata={
                    "base_model": config.base_model,
                    "family": config.family,
                    "eval_loss": eval_metrics.get("eval_loss"),
                    "train_examples": len(tokenized["train"]),
                },
            )
            if model_uri:
                report["mlflow_model_uri"] = model_uri
                tracking.set_tags({"modelo_registrado": model_uri})
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        return report
