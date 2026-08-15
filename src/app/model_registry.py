"""Empacotamento do modelo ajustado como modelo do MLflow.

O flavor nativo ``mlflow.transformers`` **não** serve para este projeto: ele
constrói um ``pipeline`` do Hugging Face a partir do modelo, e a versão 5 da
biblioteca ``transformers`` removeu a tarefa ``text2text-generation``, a
única adequada a um modelo *encoder-decoder* como o T5. Usar
``text-generation`` no lugar dela produz saída incorreta — o modelo apenas
repete o prompt, porque o pipeline causal não passa a entrada pelo
codificador.

A alternativa é um modelo ``pyfunc`` próprio, que:

* carrega o tokenizador e os pesos no formato Hugging Face a partir dos
  artefatos da versão registrada;
* reaproveita o :class:`app.generate.QAGenerator`, ou seja, empacota junto
  o formato do prompt e a limpeza da saída — que são parte do contrato do
  modelo, não detalhes do chamador;
* aceita os parâmetros de decodificação como ``params`` da assinatura, de
  modo que um consumidor via REST (``mlflow models serve``) tenha os mesmos
  controles do playground.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.config import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_MODEL_ARTIFACT,
    MLFLOW_REGISTERED_MODEL,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

#: Esquemas de URI que o MLflow resolve (em vez de um diretório local).
MLFLOW_URI_PREFIXES = ("models:/", "runs:/", "mlflow-artifacts:/")

#: Colunas aceitas na entrada do modelo empacotado.
INPUT_COLUMNS = ("question", "topic")


def is_mlflow_uri(reference: str | Path) -> bool:
    """Indica se a referência é um URI do MLflow e não um caminho local."""
    return str(reference).startswith(MLFLOW_URI_PREFIXES)


def champion_uri(model_name: str = MLFLOW_REGISTERED_MODEL) -> str:
    """URI da versão apontada pelo alias de produção no Model Registry."""
    return f"models:/{model_name}@{MLFLOW_CHAMPION_ALIAS}"


class QAEmbrapaModel:
    """Modelo ``pyfunc`` que responde perguntas de produtores rurais.

    Herda de ``mlflow.pyfunc.PythonModel`` em tempo de execução; a herança é
    resolvida em :func:`_python_model` para que este módulo possa ser
    importado sem o MLflow instalado.
    """

    def load_context(self, context) -> None:
        """Carrega o modelo a partir dos artefatos da versão registrada."""
        from app.generate import QAGenerator

        model_dir = context.artifacts[MLFLOW_MODEL_ARTIFACT]
        self.generator = QAGenerator(model_dir)

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        """Gera uma resposta para cada linha da entrada.

        ``model_input`` pode ser uma lista de perguntas ou um ``DataFrame``
        com as colunas ``question`` e, opcionalmente, ``topic``.
        """
        from dataclasses import replace

        questions, topics = _normalize_input(model_input)
        config = GenerationConfig()
        if params:
            known = {
                key: value
                for key, value in params.items()
                if key in GenerationConfig.__dataclass_fields__
            }
            if known:
                config = replace(config, **known)

        answers = self.generator.answer_many(
            list(zip(questions, topics)), config=config
        )
        return [variants[0] if variants else "" for variants in answers]


def _normalize_input(model_input) -> tuple[list[str], list[str | None]]:
    """Converte as formas aceitas de entrada em listas de perguntas e temas."""
    if isinstance(model_input, str):
        return [model_input], [None]
    if hasattr(model_input, "columns"):  # pandas.DataFrame
        columns = list(model_input.columns)
        if "question" in columns:
            questions = [str(item) for item in model_input["question"]]
            topics = (
                [None if item is None else str(item) for item in model_input["topic"]]
                if "topic" in columns
                else [None] * len(questions)
            )
            return questions, topics
        first = model_input[columns[0]]
        return [str(item) for item in first], [None] * len(first)
    if isinstance(model_input, dict):
        questions = [str(item) for item in model_input["question"]]
        topics = model_input.get("topic") or [None] * len(questions)
        return questions, [None if t is None else str(t) for t in topics]
    questions = [str(item) for item in model_input]
    return questions, [None] * len(questions)


@contextlib.contextmanager
def staged_weights(model_dir: Path) -> Iterator[Path]:
    """Prepara um diretório com apenas os arquivos necessários à inferência.

    ``models/ptt5-qa-embrapa`` contém também ``checkpoints/``, que o Trainer
    usa para retomar o treino e guarda o estado do otimizador (2,5 GB). Sem
    esta separação, cada versão registrada carregaria quatro vezes o
    necessário: o modelo empacotado precisa dos pesos finais, do tokenizador
    e das configurações, nada além disso.
    """
    model_dir = Path(model_dir)
    with tempfile.TemporaryDirectory(prefix="qa-embrapa-modelo-") as temporary:
        staged = Path(temporary) / model_dir.name
        staged.mkdir(parents=True)
        for item in sorted(model_dir.iterdir()):
            if item.is_file():
                shutil.copy2(item, staged / item.name)
        logger.info(
            "Empacotando %d arquivo(s) do modelo (%.0f MB), sem os checkpoints.",
            len(list(staged.iterdir())),
            sum(f.stat().st_size for f in staged.iterdir()) / 2**20,
        )
        yield staged


def _python_model():
    """Cria a instância do ``PythonModel``, já com a herança do MLflow."""
    import mlflow.pyfunc

    concrete = type(
        "QAEmbrapaPythonModel", (QAEmbrapaModel, mlflow.pyfunc.PythonModel), {}
    )
    return concrete()


def build_signature():
    """Assinatura do modelo: entrada tabular e parâmetros de decodificação."""
    import pandas as pd
    from mlflow.models import infer_signature

    example = pd.DataFrame(
        {
            "question": ["Qual é a melhor época de semeadura da soja?"],
            "topic": ["Soja"],
        }
    )
    defaults = GenerationConfig()
    params = {
        "max_new_tokens": defaults.max_new_tokens,
        "num_beams": defaults.num_beams,
        "do_sample": defaults.do_sample,
        "temperature": defaults.temperature,
        "top_p": defaults.top_p,
        "length_penalty": defaults.length_penalty,
        "repetition_penalty": defaults.repetition_penalty,
    }
    signature = infer_signature(example, ["A melhor época depende da região."], params)
    return signature, example


def log_and_register_model(
    model_dir: Path,
    artifact_name: str = "modelo",
    registered_model_name: str | None = MLFLOW_REGISTERED_MODEL,
    alias: str | None = MLFLOW_CHAMPION_ALIAS,
    extra_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Empacota o modelo na execução ativa e registra uma versão.

    Devolve o URI do modelo registrado (``models:/nome/versão``) ou ``None``
    quando o rastreamento está desligado.
    """
    from app import tracking

    if not tracking.is_enabled():
        logger.info("Rastreamento desligado: modelo não empacotado no MLflow.")
        return None

    import mlflow

    if mlflow.active_run() is None:
        logger.warning("Nenhuma execução ativa; modelo não empacotado.")
        return None

    signature, example = build_signature()
    logger.info("Empacotando o modelo de %s como pyfunc do MLflow ...", model_dir)
    with staged_weights(model_dir) as weights:
        info = mlflow.pyfunc.log_model(
            name=artifact_name,
            python_model=_python_model(),
            artifacts={MLFLOW_MODEL_ARTIFACT: str(weights)},
            # Empacota o próprio pacote da aplicação: o formato do prompt e a
            # limpeza da saída viajam com o modelo.
            code_paths=[str(Path(__file__).resolve().parent)],
            signature=signature,
            input_example=example,
            registered_model_name=registered_model_name,
            metadata=extra_metadata or {},
            pip_requirements=[
                f"transformers=={_version('transformers')}",
                f"torch=={_version('torch')}",
                f"sentencepiece=={_version('sentencepiece')}",
                f"pandas=={_version('pandas')}",
            ],
        )

    model_uri = info.model_uri
    if registered_model_name:
        version = _latest_version(registered_model_name)
        if version is not None:
            model_uri = f"models:/{registered_model_name}/{version}"
            if alias:
                mlflow.MlflowClient().set_registered_model_alias(
                    registered_model_name, alias, version
                )
                logger.info(
                    "Versão %s registrada como '%s' de '%s'",
                    version,
                    alias,
                    registered_model_name,
                )
    logger.info("Modelo empacotado em %s", model_uri)
    return model_uri


def _version(package: str) -> str:
    """Versão instalada de um pacote, para fixar as dependências do modelo."""
    import importlib.metadata as metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0"


def _latest_version(model_name: str) -> str | None:
    """Número da última versão registrada de um modelo."""
    import mlflow

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        return None
    return str(max(int(item.version) for item in versions))


#: Arquivos que identificam um diretório de pesos carregável. Um ajuste
#: completo grava ``config.json``; um ajuste por LoRA grava apenas
#: ``adapter_config.json``, porque os pesos base continuam no Hub.
WEIGHT_MARKERS = ("config.json", "adapter_config.json")


def resolve_model_dir(reference: str | Path) -> Path:
    """Devolve um diretório local com os pesos, aceitando URIs do MLflow.

    Baixa a raiz do modelo registrado e localiza os pesos procurando por um
    dos :data:`WEIGHT_MARKERS`. O nome desse subdiretório **não** é a chave
    usada em ``artifacts={...}`` no momento do registro: o MLflow guarda
    cada artefato sob o nome base do caminho de origem —
    ``artifacts/ptt5-qa-embrapa``, no caso deste projeto. Procurar pelo
    arquivo evita depender desse detalhe de implementação.

    Procurar por **dois** marcadores, e não só pelo ``config.json``, é o que
    faz um modelo LoRA ser encontrado: ele não tem ``config.json``, porque
    o que foi empacotado são os adaptadores, e a arquitetura base é
    resolvida a partir do ``adapter_config.json``.

    Funciona com as três formas de URI: ``models:/nome/versão``,
    ``models:/nome@alias`` e ``models:/m-<id>``.
    """
    if not is_mlflow_uri(reference):
        return Path(reference)

    import mlflow
    from app import tracking

    tracking.configure()
    uri = str(reference).rstrip("/")
    logger.info("Baixando o modelo de %s ...", uri)
    local = Path(mlflow.artifacts.download_artifacts(uri))

    for marker in WEIGHT_MARKERS:
        candidates = sorted(local.glob(f"artifacts/*/{marker}"))
        if candidates:
            return candidates[0].parent
    for marker in WEIGHT_MARKERS:
        if (local / marker).exists():
            return local
    raise FileNotFoundError(
        f"Não foi possível localizar os pesos do modelo em {uri} "
        f"(baixado em {local}). Nenhum de {', '.join(WEIGHT_MARKERS)} "
        "foi encontrado."
    )


def list_registered_versions(
    model_name: str = MLFLOW_REGISTERED_MODEL,
) -> list[dict[str, Any]]:
    """Lista as versões do modelo no Model Registry, mais recentes primeiro."""
    from app import tracking

    if not tracking.configure():
        return []

    import mlflow

    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
        # ``search_model_versions`` não preenche os aliases de cada versão; o
        # mapeamento alias -> versão vem do modelo registrado.
        alias_map = client.get_registered_model(model_name).aliases or {}
    except Exception:  # pragma: no cover - modelo ainda não registrado
        return []

    por_versao: dict[str, list[str]] = {}
    for alias, version in alias_map.items():
        por_versao.setdefault(str(version), []).append(alias)

    resultado: list[dict[str, Any]] = []
    for item in sorted(versions, key=lambda v: int(v.version), reverse=True):
        resultado.append(
            {
                "version": item.version,
                "run_id": item.run_id,
                "aliases": sorted(por_versao.get(str(item.version), [])),
                "status": item.status,
                "created": item.creation_timestamp,
                "uri": f"models:/{model_name}/{item.version}",
            }
        )
    return resultado
