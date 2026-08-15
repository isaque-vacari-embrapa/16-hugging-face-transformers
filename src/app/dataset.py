"""Construção do conjunto de dados no formato esperado pelo Hugging Face.

Transforma o JSON Lines produzido por :mod:`app.extract` em um
``DatasetDict`` com as partições de treino, validação e teste, e faz a
tokenização no formato *sequence-to-sequence* (entrada = prompt com tema e
pergunta, saída = resposta).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import (
    QA_DATASET_FILE,
    SPLITS_DIR,
    DatasetSplitConfig,
    TrainingConfig,
    build_prompt,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "test")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def load_qa_dataset(path: Path = QA_DATASET_FILE):
    """Carrega o JSON Lines de pares pergunta/resposta como ``Dataset``."""
    from datasets import load_dataset

    if not path.exists():
        raise FileNotFoundError(
            f"Conjunto de dados não encontrado: {path}. "
            "Execute `poetry run qa-embrapa extract` primeiro."
        )
    dataset = load_dataset("json", data_files=str(path), split="train")
    logger.info("%d exemplos carregados de %s", len(dataset), path)
    return dataset


def split_dataset(dataset, config: DatasetSplitConfig | None = None):
    """Divide o conjunto em treino/validação/teste de forma reprodutível.

    A divisão é aleatória (com semente fixa) e não estratificada por tema:
    com ~500 pares por cultura e 20 culturas, a amostragem aleatória já
    distribui todos os temas nas três partições.
    """
    from datasets import DatasetDict

    config = config or DatasetSplitConfig()
    holdout = config.validation_size + config.test_size
    if not 0 < holdout < 1:
        raise ValueError("A soma de validation_size e test_size deve estar em (0, 1).")

    first = dataset.train_test_split(test_size=holdout, seed=config.seed)
    test_ratio = config.test_size / holdout
    second = first["test"].train_test_split(test_size=test_ratio, seed=config.seed)

    splits = DatasetDict(
        train=first["train"],
        validation=second["train"],
        test=second["test"],
    )
    logger.info(
        "Partições: treino=%d validação=%d teste=%d",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return splits


def save_splits(splits, output_dir: Path = SPLITS_DIR) -> dict[str, Path]:
    """Grava cada partição em um arquivo JSON Lines."""
    ensure_dirs(output_dir)
    paths: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        path = output_dir / f"{name}.jsonl"
        splits[name].to_json(str(path), force_ascii=False, lines=True)
        paths[name] = path
        logger.info("%s: %d exemplos -> %s", name, len(splits[name]), path)
    return paths


def load_splits(splits_dir: Path = SPLITS_DIR):
    """Recarrega as partições gravadas em ``splits_dir``."""
    from datasets import load_dataset

    files = {name: splits_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Partições não encontradas: "
            + ", ".join(missing)
            + ". Execute `poetry run qa-embrapa dataset` primeiro."
        )
    return load_dataset(
        "json", data_files={name: str(path) for name, path in files.items()}
    )


def build_splits(
    input_file: Path = QA_DATASET_FILE,
    splits_dir: Path = SPLITS_DIR,
    config: DatasetSplitConfig | None = None,
    report_file: Path | None = None,
    run_name: str = "particoes",
) -> dict[str, object]:
    """Cria as partições e rastreia a etapa no MLflow.

    Registra as proporções e a semente como parâmetros, os tamanhos como
    métricas e a linhagem de cada partição, para que o treinamento possa ser
    associado exatamente aos dados que consumiu.
    """
    from app import tracking

    config = config or DatasetSplitConfig()
    with tracking.start_run(run_name):
        tracking.log_params(
            {
                "entrada": input_file,
                "validation_size": config.validation_size,
                "test_size": config.test_size,
                "seed": config.seed,
            }
        )
        dataset = load_qa_dataset(input_file)
        splits = split_dataset(dataset, config)
        save_splits(splits, splits_dir)
        statistics = dataset_statistics(splits)

        tracking.log_metrics(
            {name: statistics[name]["examples"] for name in SPLIT_NAMES},
            prefix="exemplos.",
        )
        tracking.log_metrics({"temas": len(statistics["topics"])})
        for name in SPLIT_NAMES:
            tracking.log_dataset(
                splits_dir / f"{name}.jsonl", f"qa-embrapa-{name}", name
            )
        if report_file is not None:
            write_json(report_file, statistics)
            tracking.log_artifact(report_file, artifact_path="relatorios")
    return statistics


def truncate_to_sentence(text: str, tokenizer, max_tokens: int) -> str:
    """Encurta o texto no limite de tokens, respeitando o fim de frase.

    Truncar no meio de uma frase ensinaria o modelo a parar de gerar em
    posições arbitrárias. Aqui as frases são acumuladas enquanto couberem
    no orçamento de tokens; se nem a primeira frase couber, aplica-se o
    corte direto por tokens.
    """
    budget = max_tokens - 1  # reserva o token de fim de sequência
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    kept: list[str] = []
    used = 0
    for sentence in sentences:
        length = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if used + length > budget:
            break
        kept.append(sentence)
        used += length
    if not kept:
        # Nem a primeira frase cabe: resta o corte direto por tokens.
        return tokenizer.decode(ids[:budget], skip_special_tokens=True).strip()
    return " ".join(kept).strip()


def build_preprocess_fn(tokenizer, config: TrainingConfig):
    """Cria a função de tokenização *encoder-decoder* usada pelo ``map``.

    Entrada e rótulo são sequências independentes: o prompt vai para o
    codificador, a resposta é o alvo do decodificador.
    """

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        prompts = [
            build_prompt(question, topic)
            for question, topic in zip(batch["question"], batch["topic"])
        ]
        targets = [
            truncate_to_sentence(answer, tokenizer, config.max_target_length)
            for answer in batch["answer"]
        ]
        model_inputs = tokenizer(
            prompts,
            max_length=config.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def build_causal_preprocess_fn(tokenizer, config: TrainingConfig):
    """Cria a função de tokenização *decoder-only* usada pelo ``map``.

    Aqui prompt e resposta formam **uma única** sequência. O rótulo é essa
    mesma sequência com as posições do prompt marcadas com ``-100``, para
    que a perda seja calculada apenas sobre a resposta: sem essa máscara o
    modelo gastaria capacidade aprendendo a reproduzir o próprio enunciado.

    A resposta termina no marcador de fim de turno do modelo de instrução
    (``<end_of_turn>`` no Gemma) — é ele que ensina a geração a parar.
    """
    from app.models import causal_prompt

    end_of_turn = tokenizer.eos_token or ""
    # O template de conversa fecha cada turno com um marcador próprio, que
    # não é o ``eos_token`` padrão do tokenizador. Quando ele existe, é o
    # que o modelo deve aprender a emitir.
    for candidate in ("<end_of_turn>",):
        if tokenizer.convert_tokens_to_ids(candidate) not in (
            None,
            tokenizer.unk_token_id,
        ):
            end_of_turn = candidate
            break

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for question, topic, answer in zip(
            batch["question"], batch["topic"], batch["answer"]
        ):
            prompt = causal_prompt(tokenizer, question, topic)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target = truncate_to_sentence(answer, tokenizer, config.max_target_length)
            answer_ids = tokenizer(target + end_of_turn, add_special_tokens=False)[
                "input_ids"
            ]

            # O prompt é curto e previsível; se algo precisa ser cortado
            # para caber, é a resposta — cortar o prompt removeria a
            # pergunta que o modelo precisa ler.
            budget = max(1, config.max_sequence_length - len(prompt_ids))
            answer_ids = answer_ids[:budget]

            ids = (prompt_ids + answer_ids)[: config.max_sequence_length]
            mask = [-100] * len(prompt_ids) + answer_ids
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append(mask[: len(ids)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


def tokenize_splits(splits, tokenizer, config: TrainingConfig, num_proc: int = 1):
    """Tokeniza todas as partições, removendo as colunas de texto original.

    A função aplicada depende da família declarada em ``config.family``.
    """
    from app.models import CAUSAL

    if config.family == CAUSAL:
        preprocess = build_causal_preprocess_fn(tokenizer, config)
    else:
        preprocess = build_preprocess_fn(tokenizer, config)

    columns = list(splits["train"].column_names)
    tokenized = splits.map(
        preprocess,
        batched=True,
        remove_columns=columns,
        num_proc=num_proc if num_proc > 1 else None,
        desc="Tokenizando",
    )
    return tokenized


class CausalCollator:
    """Agrupa exemplos de comprimento variável em um lote para o Trainer.

    O ``DataCollatorForSeq2Seq`` não serve aqui: ele monta
    ``decoder_input_ids``, que um modelo *decoder-only* não tem. O
    preenchimento é sempre **à direita** — o tokenizador do Gemma vem
    configurado para preencher à esquerda (correto na geração, errado no
    treino, onde deslocaria o alinhamento entre entrada e rótulo).
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = longest - len(item["input_ids"])
            batch["input_ids"].append(
                list(item["input_ids"]) + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(list(item["attention_mask"]) + [0] * padding)
            # ``-100`` é ignorado pela função de perda do PyTorch.
            batch["labels"].append(list(item["labels"]) + [-100] * padding)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


def dataset_statistics(splits) -> dict[str, object]:
    """Resumo das partições para registro no relatório."""
    stats: dict[str, object] = {}
    for name in SPLIT_NAMES:
        subset = splits[name]
        topics = sorted({str(topic) for topic in subset["topic"]})
        stats[name] = {
            "examples": len(subset),
            "topics": len(topics),
        }
    stats["topics"] = sorted({str(topic) for topic in splits["train"]["topic"]})
    return stats


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Grava um relatório em JSON com acentuação preservada."""
    ensure_dirs(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
