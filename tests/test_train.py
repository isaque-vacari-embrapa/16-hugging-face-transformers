"""Testes das partes puras do módulo de treinamento.

Não executam ajuste fino: verificam o cálculo de passos, a escolha de
precisão numérica e a montagem dos ``Seq2SeqTrainingArguments`` — em especial
a integração com o MLflow via ``report_to``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import TrainingConfig
from app.train import (
    MIN_BF16_CAPABILITY,
    build_training_arguments,
    describe_environment,
    estimate_total_steps,
    resolve_precision,
)

# ---------------------------------------------------------------------------
# Estimativa de passos
# ---------------------------------------------------------------------------


def test_estimate_total_steps_com_lote_efetivo_16() -> None:
    config = TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5.0,
    )
    # 7511 exemplos / 16 = 470 passos por época (arredondado para cima).
    assert estimate_total_steps(config, 7511) == 470 * 5


def test_estimate_total_steps_arredonda_para_cima() -> None:
    config = TrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
    )
    assert estimate_total_steps(config, 5) == 3


def test_estimate_total_steps_nunca_e_zero() -> None:
    config = TrainingConfig(num_train_epochs=0.0)
    assert estimate_total_steps(config, 0) == 1


# ---------------------------------------------------------------------------
# Precisão numérica
# ---------------------------------------------------------------------------


def test_precisao_em_float32_sem_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_precision() == {"bf16": False, "fp16": False}


def test_precisao_em_float32_em_gpu_turing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turing (sm_75) só tem bfloat16 emulado: mais lento e pior."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (7, 5))
    assert resolve_precision() == {"bf16": False, "fp16": False}


def test_precisao_em_bfloat16_em_gpu_ampere(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 6))
    assert resolve_precision() == {"bf16": True, "fp16": False}


def test_float16_nunca_e_escolhido(monkeypatch: pytest.MonkeyPatch) -> None:
    """T5 produz NaN em float16; nenhuma combinação deve ligá-lo."""
    import torch

    for disponivel, capacidade in [(False, None), (True, (7, 5)), (True, (9, 0))]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda v=disponivel: v)
        if capacidade:
            monkeypatch.setattr(
                torch.cuda, "get_device_capability", lambda index=0, c=capacidade: c
            )
        assert resolve_precision()["fp16"] is False


def test_limiar_de_bfloat16_e_ampere() -> None:
    assert MIN_BF16_CAPABILITY == (8, 0)


# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------


def test_describe_environment_traz_as_versoes() -> None:
    ambiente = describe_environment()
    assert set(ambiente) == {
        "torch",
        "transformers",
        "cuda",
        "gpu",
        "compute_capability",
    }
    assert ambiente["torch"]
    assert ambiente["transformers"]


# ---------------------------------------------------------------------------
# Argumentos do Trainer
# ---------------------------------------------------------------------------


def test_argumentos_reportam_ao_mlflow_quando_ativo() -> None:
    args = build_training_arguments(
        TrainingConfig(),
        7511,
        use_mlflow=True,
        precision={"bf16": False, "fp16": False},
    )
    assert args.report_to == ["mlflow"]


def test_argumentos_nao_reportam_quando_mlflow_desligado() -> None:
    args = build_training_arguments(
        TrainingConfig(),
        7511,
        use_mlflow=False,
        precision={"bf16": False, "fp16": False},
    )
    assert args.report_to == []


def test_warmup_steps_derivam_da_proporcao_configurada() -> None:
    config = replace(TrainingConfig(), warmup_ratio=0.1, num_train_epochs=5.0)
    args = build_training_arguments(
        config, 7511, precision={"bf16": False, "fp16": False}
    )
    # 470 passos por época x 5 épocas x 10% = 235.
    assert args.warmup_steps == 235


def test_precisao_informada_e_respeitada() -> None:
    args = build_training_arguments(
        TrainingConfig(), 100, precision={"bf16": True, "fp16": False}
    )
    assert args.bf16 is True
    assert args.fp16 is False


def test_avaliacao_e_salvamento_por_epoca_com_melhor_modelo() -> None:
    args = build_training_arguments(
        TrainingConfig(), 100, precision={"bf16": False, "fp16": False}
    )
    assert args.eval_strategy.value == "epoch"
    assert args.save_strategy.value == "epoch"
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False


def test_checkpoints_ficam_em_subdiretorio_do_modelo() -> None:
    config = TrainingConfig()
    args = build_training_arguments(
        config, 100, precision={"bf16": False, "fp16": False}
    )
    assert args.output_dir == str(config.output_dir / "checkpoints")
