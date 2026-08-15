"""Integração com o MLflow: rastreamento de experimentos e artefatos.

Concentra a configuração do MLflow e oferece utilitários que funcionam
também quando o rastreamento está desligado (``--no-mlflow``), de modo que
as etapas do pipeline não precisem de ramificações condicionais.

O backend é um SQLite local (``mlflow/mlflow.db``) porque o **Model
Registry** do MLflow não funciona sobre o armazenamento em arquivos puro:
registrar versões, atribuir aliases e resolver URIs ``models:/...`` exige
um backend com banco de dados. Os artefatos ficam em ``mlflow/artifacts``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from app.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

#: Variável de ambiente que desliga o rastreamento em todo o processo.
DISABLE_ENV_VAR = "QA_EMBRAPA_NO_MLFLOW"

#: Limite de caracteres de um valor de parâmetro aceito pelo MLflow.
MAX_PARAM_LENGTH = 500

_state: dict[str, Any] = {"configured": False, "enabled": True}


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Indica se o rastreamento no MLflow está ativo neste processo."""
    if os.environ.get(DISABLE_ENV_VAR):
        return False
    return bool(_state["enabled"])


def disable() -> None:
    """Desliga o rastreamento (usado pela opção ``--no-mlflow`` da CLI)."""
    _state["enabled"] = False
    os.environ[DISABLE_ENV_VAR] = "1"


def configure(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> bool:
    """Prepara o MLflow e devolve se o rastreamento está utilizável.

    Também exporta ``MLFLOW_TRACKING_URI`` e ``MLFLOW_EXPERIMENT_NAME`` no
    ambiente, para que integrações que leem essas variáveis — como o
    ``MLflowCallback`` do Hugging Face Trainer — encontrem o mesmo destino.

    Os valores padrão são resolvidos aqui, e não na assinatura, para que os
    testes possam redirecionar o destino substituindo as constantes do
    módulo.
    """
    experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
    tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    if not is_enabled():
        return False
    if _state["configured"] == experiment_name:
        return True

    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow é dependência direta
        logger.warning("MLflow não está instalado; rastreamento desligado.")
        disable()
        return False

    ensure_dirs(MLFLOW_DIR, MLFLOW_ARTIFACT_ROOT)
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri()
        )
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    _state["configured"] = experiment_name
    logger.info(
        "MLflow configurado: experimento '%s' em %s", experiment_name, tracking_uri
    )
    return True


# ---------------------------------------------------------------------------
# Execuções (runs)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def start_run(
    run_name: str,
    tags: Mapping[str, str] | None = None,
    nested: bool = False,
    description: str | None = None,
) -> Iterator[Any]:
    """Abre uma execução do MLflow, ou não faz nada se estiver desligado.

    Devolve o objeto ``Run`` do MLflow ou ``None``, de modo que o chamador
    possa escrever o mesmo código nas duas situações.
    """
    if not configure():
        yield None
        return

    import mlflow

    etapa_tags = {"etapa": run_name, "projeto": "qa-embrapa"}
    etapa_tags.update(tags or {})
    with mlflow.start_run(
        run_name=run_name, nested=nested, tags=etapa_tags, description=description
    ) as run:
        logger.info("Execução MLflow '%s' iniciada (%s)", run_name, run.info.run_id)
        yield run


def active_run_id() -> str | None:
    """Identificador da execução ativa, se houver."""
    if not is_enabled():
        return None
    import mlflow

    run = mlflow.active_run()
    return run.info.run_id if run else None


# ---------------------------------------------------------------------------
# Registro de parâmetros, métricas e artefatos
# ---------------------------------------------------------------------------


def flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Achata dicionários e dataclasses aninhados em chaves ``a.b.c``.

    O MLflow aceita apenas pares chave/valor escalares como parâmetros.
    """
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)

    flat: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            composed = f"{prefix}{key}"
            if isinstance(value, Mapping) or (
                is_dataclass(value) and not isinstance(value, type)
            ):
                flat.update(flatten(value, prefix=f"{composed}."))
            elif isinstance(value, (list, tuple, set)):
                flat[composed] = ", ".join(str(item) for item in value)
            elif isinstance(value, Path):
                flat[composed] = str(value)
            else:
                flat[composed] = value
        return flat
    return {prefix.rstrip("."): payload}


def log_params(payload: Any, prefix: str = "") -> None:
    """Registra parâmetros, achatando estruturas e truncando valores longos."""
    if not is_enabled():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    params = {}
    for key, value in flatten(payload, prefix=prefix).items():
        text = str(value)
        params[key] = text[:MAX_PARAM_LENGTH] if len(text) > MAX_PARAM_LENGTH else value
    if params:
        mlflow.log_params(params)


def log_metrics(payload: Mapping[str, Any], prefix: str = "", step: int | None = None):
    """Registra métricas numéricas, ignorando valores não numéricos."""
    if not is_enabled():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    metrics: dict[str, float] = {}
    for key, value in flatten(payload, prefix=prefix).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[key.replace(" ", "_")] = float(value)
    if metrics:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    """Anexa um arquivo existente à execução ativa."""
    if not is_enabled() or not path.exists():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_dict(payload: Any, filename: str) -> None:
    """Anexa uma estrutura serializável como JSON à execução ativa."""
    if not is_enabled():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_dict(json.loads(json.dumps(payload, ensure_ascii=False)), filename)


def log_table(rows: list[dict[str, Any]], filename: str) -> None:
    """Anexa uma tabela de exemplos, visível na interface do MLflow.

    ``mlflow.log_table`` aceita apenas ``DataFrame`` ou dicionário de colunas,
    não uma lista de linhas — daí a conversão.
    """
    if not is_enabled() or not rows:
        return
    import pandas as pd

    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_table(data=pd.DataFrame(rows), artifact_file=filename)


def set_tags(tags: Mapping[str, str]) -> None:
    """Aplica etiquetas à execução ativa."""
    if not is_enabled():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.set_tags(dict(tags))


# ---------------------------------------------------------------------------
# Linhagem de dados
# ---------------------------------------------------------------------------


def log_dataset(
    path: Path, name: str, context: str, targets: str | None = None
) -> None:
    """Registra a linhagem de um arquivo JSON Lines usado como entrada.

    Usa ``mlflow.data`` para que a interface mostre origem, esquema e
    resumo estatístico do conjunto associado à execução.
    """
    if not is_enabled() or not path.exists():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    try:
        import pandas as pd

        frame = pd.read_json(path, lines=True)
        dataset = mlflow.data.from_pandas(
            frame, source=path.as_uri(), name=name, targets=targets
        )
        mlflow.log_input(dataset, context=context)
    except Exception as error:  # pragma: no cover - depende do ambiente
        logger.warning("Não foi possível registrar a linhagem de %s: %s", path, error)
