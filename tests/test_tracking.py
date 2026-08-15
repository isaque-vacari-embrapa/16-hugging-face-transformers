"""Testes da integração com o MLflow.

Cada teste usa um diretório temporário como backend, para não escrever no
``mlflow/mlflow.db`` do projeto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app import tracking
from app.config import GenerationConfig, TrainingConfig


@pytest.fixture(autouse=True)
def estado_limpo(monkeypatch: pytest.MonkeyPatch):
    """Reinicia o estado do módulo e o ambiente entre os testes."""
    monkeypatch.delenv(tracking.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(tracking, "_state", {"configured": False, "enabled": True})
    yield


@pytest.fixture()
def mlflow_temporario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Aponta o MLflow para um backend SQLite descartável."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking, "MLFLOW_DIR", tmp_path)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    return uri


# ---------------------------------------------------------------------------
# Liga/desliga
# ---------------------------------------------------------------------------


def test_rastreamento_ativo_por_padrao() -> None:
    assert tracking.is_enabled() is True


def test_disable_desliga_e_exporta_a_variavel_de_ambiente() -> None:
    import os

    tracking.disable()
    assert tracking.is_enabled() is False
    assert os.environ[tracking.DISABLE_ENV_VAR] == "1"


def test_variavel_de_ambiente_desliga_o_rastreamento(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tracking.DISABLE_ENV_VAR, "1")
    assert tracking.is_enabled() is False


def test_configure_devolve_falso_quando_desligado() -> None:
    tracking.disable()
    assert tracking.configure() is False


def test_start_run_nao_falha_com_rastreamento_desligado() -> None:
    tracking.disable()
    with tracking.start_run("etapa") as run:
        assert run is None
        # As funções de registro precisam ser inócuas nessa situação.
        tracking.log_params({"a": 1})
        tracking.log_metrics({"b": 2.0})
        tracking.log_dict({"c": 3}, "x.json")
        tracking.set_tags({"d": "4"})
    assert tracking.active_run_id() is None


# ---------------------------------------------------------------------------
# Achatamento de parâmetros
# ---------------------------------------------------------------------------


@dataclass
class _Aninhado:
    alfa: int = 1
    caminho: Path = Path("/tmp/x")


def test_flatten_achata_dicionarios_aninhados() -> None:
    plano = tracking.flatten({"a": {"b": {"c": 1}}, "d": 2})
    assert plano == {"a.b.c": 1, "d": 2}


def test_flatten_aceita_dataclass() -> None:
    plano = tracking.flatten(_Aninhado())
    assert plano["alfa"] == 1
    assert plano["caminho"] == "/tmp/x"


def test_flatten_converte_listas_em_texto() -> None:
    assert tracking.flatten({"temas": ["Soja", "Milho"]}) == {"temas": "Soja, Milho"}


def test_flatten_aplica_prefixo() -> None:
    assert tracking.flatten({"a": 1}, prefix="geracao.") == {"geracao.a": 1}


def test_flatten_de_configuracoes_reais_nao_perde_campos() -> None:
    treino = tracking.flatten(TrainingConfig())
    geracao = tracking.flatten(GenerationConfig())
    assert treino["num_train_epochs"] == TrainingConfig().num_train_epochs
    assert treino["output_dir"] == str(TrainingConfig().output_dir)
    assert geracao["length_penalty"] == GenerationConfig().length_penalty


# ---------------------------------------------------------------------------
# Registro efetivo
# ---------------------------------------------------------------------------


def test_execucao_registra_parametros_metricas_e_artefatos(
    mlflow_temporario: str, tmp_path: Path
) -> None:
    import mlflow

    assert tracking.configure(experiment_name="teste") is True

    arquivo = tmp_path / "relatorio.json"
    arquivo.write_text('{"ok": true}', encoding="utf-8")

    with tracking.start_run("etapa-teste", tags={"origem": "pytest"}) as run:
        run_id = run.info.run_id
        tracking.log_params({"config": {"epocas": 3}, "modelo": "ptt5"})
        tracking.log_metrics({"perda": 2.5, "texto_ignorado": "abc"})
        tracking.log_metrics({"rouge": 20.0}, prefix="beam.")
        tracking.log_artifact(arquivo, artifact_path="relatorios")
        tracking.log_dict({"resumo": "x"}, "resumo.json")
        tracking.set_tags({"decodificacao": "beam"})

    cliente = mlflow.MlflowClient(tracking_uri=mlflow_temporario)
    dados = cliente.get_run(run_id)
    assert dados.data.params["config.epocas"] == "3"
    assert dados.data.params["modelo"] == "ptt5"
    assert dados.data.metrics["perda"] == 2.5
    assert dados.data.metrics["beam.rouge"] == 20.0
    assert "texto_ignorado" not in dados.data.metrics
    assert dados.data.tags["origem"] == "pytest"
    assert dados.data.tags["etapa"] == "etapa-teste"
    assert dados.data.tags["decodificacao"] == "beam"
    artefatos = {item.path for item in cliente.list_artifacts(run_id)}
    assert "relatorios" in artefatos
    assert "resumo.json" in artefatos


def test_execucoes_aninhadas_ficam_associadas_a_mae(mlflow_temporario: str) -> None:
    import mlflow

    tracking.configure(experiment_name="teste-aninhado")
    with tracking.start_run("mae") as mae:
        with tracking.start_run("filha", nested=True) as filha:
            assert filha.info.run_id != mae.info.run_id
            filha_id = filha.info.run_id
        mae_id = mae.info.run_id

    cliente = mlflow.MlflowClient(tracking_uri=mlflow_temporario)
    filha_run = cliente.get_run(filha_id)
    assert filha_run.data.tags["mlflow.parentRunId"] == mae_id


def test_valores_longos_de_parametro_sao_truncados(mlflow_temporario: str) -> None:
    import mlflow

    tracking.configure(experiment_name="teste-truncado")
    with tracking.start_run("etapa") as run:
        tracking.log_params({"grande": "x" * (tracking.MAX_PARAM_LENGTH + 50)})
        run_id = run.info.run_id

    cliente = mlflow.MlflowClient(tracking_uri=mlflow_temporario)
    valor = cliente.get_run(run_id).data.params["grande"]
    assert len(valor) == tracking.MAX_PARAM_LENGTH


def test_log_dataset_registra_linhagem(mlflow_temporario: str, tmp_path: Path) -> None:
    import mlflow

    tracking.configure(experiment_name="teste-linhagem")
    jsonl = tmp_path / "dados.jsonl"
    jsonl.write_text(
        '{"question": "Q1?", "answer": "R1", "topic": "Soja"}\n'
        '{"question": "Q2?", "answer": "R2", "topic": "Milho"}\n',
        encoding="utf-8",
    )
    with tracking.start_run("etapa") as run:
        tracking.log_dataset(jsonl, "conjunto-teste", "training", targets="answer")
        run_id = run.info.run_id

    cliente = mlflow.MlflowClient(tracking_uri=mlflow_temporario)
    entradas = cliente.get_run(run_id).inputs.dataset_inputs
    assert len(entradas) == 1
    assert entradas[0].dataset.name == "conjunto-teste"


def test_log_table_aceita_lista_de_linhas(mlflow_temporario: str) -> None:
    """``mlflow.log_table`` só aceita DataFrame ou dicionário de colunas."""
    import mlflow

    tracking.configure(experiment_name="teste-tabela")
    with tracking.start_run("etapa") as run:
        tracking.log_table(
            [
                {"pergunta": "Q1?", "gerada": "R1", "palavras": 2},
                {"pergunta": "Q2?", "gerada": "R2", "palavras": 3},
            ],
            "exemplos.json",
        )
        run_id = run.info.run_id

    cliente = mlflow.MlflowClient(tracking_uri=mlflow_temporario)
    artefatos = {item.path for item in cliente.list_artifacts(run_id)}
    assert "exemplos.json" in artefatos
    tabela = mlflow.load_table("exemplos.json", run_ids=[run_id])
    assert list(tabela["pergunta"]) == ["Q1?", "Q2?"]


def test_log_table_ignora_lista_vazia(mlflow_temporario: str) -> None:
    tracking.configure(experiment_name="teste-tabela-vazia")
    with tracking.start_run("etapa"):
        tracking.log_table([], "vazio.json")  # não deve levantar exceção


def test_log_artifact_ignora_arquivo_inexistente(
    mlflow_temporario: str, tmp_path: Path
) -> None:
    tracking.configure(experiment_name="teste-ausente")
    with tracking.start_run("etapa"):
        # Não deve levantar exceção.
        tracking.log_artifact(tmp_path / "nao-existe.json")
