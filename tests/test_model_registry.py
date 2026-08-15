"""Testes do empacotamento do modelo para o MLflow.

Os testes de resolução de URI e normalização de entrada não carregam o
modelo. O teste de ida e volta pelo Model Registry carrega os pesos e por
isso é ignorado quando o modelo ajustado não existe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    DEFAULT_MODEL_DIR,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_MODEL_ARTIFACT,
    MLFLOW_REGISTERED_MODEL,
)
from app.model_registry import (
    QAEmbrapaModel,
    _normalize_input,
    champion_uri,
    is_mlflow_uri,
    resolve_model_dir,
)

modelo_ausente = not (DEFAULT_MODEL_DIR / "config.json").exists()
requer_modelo = pytest.mark.skipif(
    modelo_ausente,
    reason=(
        f"requer o modelo ajustado em {DEFAULT_MODEL_DIR}; "
        "execute `poetry run qa-embrapa train`"
    ),
)


# ---------------------------------------------------------------------------
# Referências e URIs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "referencia",
    [
        "models:/qa-embrapa-ptt5@champion",
        "models:/qa-embrapa-ptt5/3",
        "runs:/abc123/modelo",
        "mlflow-artifacts:/1/abc/artifacts/modelo",
    ],
)
def test_reconhece_uris_do_mlflow(referencia: str) -> None:
    assert is_mlflow_uri(referencia) is True


@pytest.mark.parametrize(
    "referencia",
    ["models/ptt5-qa-embrapa", "/tmp/modelo", "unicamp-dl/ptt5-v2-base"],
)
def test_nao_confunde_caminho_local_com_uri(referencia: str) -> None:
    assert is_mlflow_uri(referencia) is False


def test_is_mlflow_uri_aceita_path() -> None:
    assert is_mlflow_uri(Path("models/ptt5-qa-embrapa")) is False


def test_champion_uri_usa_o_alias_configurado() -> None:
    assert (
        champion_uri() == f"models:/{MLFLOW_REGISTERED_MODEL}@{MLFLOW_CHAMPION_ALIAS}"
    )
    assert champion_uri("outro") == f"models:/outro@{MLFLOW_CHAMPION_ALIAS}"


def test_resolve_model_dir_devolve_caminho_local_sem_tocar_no_mlflow(
    tmp_path: Path,
) -> None:
    assert resolve_model_dir(tmp_path) == tmp_path
    assert resolve_model_dir(str(tmp_path)) == tmp_path


# ---------------------------------------------------------------------------
# Normalização da entrada
# ---------------------------------------------------------------------------


def test_normalize_input_de_texto_simples() -> None:
    assert _normalize_input("Como semear?") == (["Como semear?"], [None])


def test_normalize_input_de_lista() -> None:
    perguntas, temas = _normalize_input(["Q1?", "Q2?"])
    assert perguntas == ["Q1?", "Q2?"]
    assert temas == [None, None]


def test_normalize_input_de_dataframe_com_tema() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"question": ["Q1?", "Q2?"], "topic": ["Soja", "Milho"]})
    perguntas, temas = _normalize_input(frame)
    assert perguntas == ["Q1?", "Q2?"]
    assert temas == ["Soja", "Milho"]


def test_normalize_input_de_dataframe_sem_tema() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"question": ["Q1?"]})
    perguntas, temas = _normalize_input(frame)
    assert perguntas == ["Q1?"]
    assert temas == [None]


def test_normalize_input_de_dataframe_com_outra_coluna() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"texto": ["Q1?", "Q2?"]})
    perguntas, temas = _normalize_input(frame)
    assert perguntas == ["Q1?", "Q2?"]
    assert temas == [None, None]


def test_normalize_input_de_dicionario() -> None:
    perguntas, temas = _normalize_input({"question": ["Q1?"], "topic": ["Soja"]})
    assert perguntas == ["Q1?"]
    assert temas == ["Soja"]


# ---------------------------------------------------------------------------
# Assinatura e contrato do modelo
# ---------------------------------------------------------------------------


def test_build_signature_expoe_os_parametros_de_decodificacao() -> None:
    pytest.importorskip("mlflow")
    from app.model_registry import build_signature

    signature, example = build_signature()
    entradas = [campo.name for campo in signature.inputs.inputs]
    assert entradas == ["question", "topic"]
    parametros = [campo.name for campo in signature.params.params]
    for esperado in ("num_beams", "do_sample", "temperature", "length_penalty"):
        assert esperado in parametros
    assert list(example.columns) == ["question", "topic"]


def test_staged_weights_deixa_os_checkpoints_de_fora(tmp_path: Path) -> None:
    """O modelo registrado não deve carregar o estado do otimizador."""
    from app.model_registry import staged_weights

    origem = tmp_path / "ptt5-qa-embrapa"
    origem.mkdir()
    for nome in ("config.json", "model.safetensors", "tokenizer.json"):
        (origem / nome).write_text("x", encoding="utf-8")
    checkpoints = origem / "checkpoints" / "checkpoint-2350"
    checkpoints.mkdir(parents=True)
    (checkpoints / "optimizer.pt").write_text("peso" * 1000, encoding="utf-8")

    with staged_weights(origem) as preparado:
        nomes = sorted(item.name for item in preparado.iterdir())
        assert nomes == ["config.json", "model.safetensors", "tokenizer.json"]
        assert not (preparado / "checkpoints").exists()
        # O nome do diretório é preservado: é ele que aparece nos artefatos.
        assert preparado.name == origem.name

    # O diretório temporário é removido ao sair do contexto.
    assert not preparado.exists()


def test_predict_repassa_apenas_parametros_conhecidos() -> None:
    """Parâmetros desconhecidos não devem quebrar ``replace``."""

    class GeradorFalso:
        def __init__(self) -> None:
            self.configuracoes = []

        def answer_many(self, pares, config=None, **kwargs):
            self.configuracoes.append(config)
            return [["resposta"] for _ in pares]

    modelo = QAEmbrapaModel()
    modelo.generator = GeradorFalso()
    saida = modelo.predict(
        None,
        ["Como semear?"],
        params={"num_beams": 7, "parametro_inexistente": 1},
    )
    assert saida == ["resposta"]
    assert modelo.generator.configuracoes[0].num_beams == 7


# ---------------------------------------------------------------------------
# Ida e volta pelo Model Registry
# ---------------------------------------------------------------------------


@requer_modelo
def test_registro_e_carregamento_pelo_model_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow
    from app import tracking
    from app.model_registry import list_registered_versions, log_and_register_model

    monkeypatch.delenv(tracking.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(tracking, "_state", {"configured": False, "enabled": True})
    monkeypatch.setattr(tracking, "MLFLOW_DIR", tmp_path)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(
        tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    tracking.configure(experiment_name="teste-registro")

    with tracking.start_run("treino-teste"):
        uri = log_and_register_model(
            DEFAULT_MODEL_DIR, registered_model_name="modelo-teste", alias="champion"
        )

    assert uri == "models:/modelo-teste/1"
    versoes = list_registered_versions("modelo-teste")
    assert len(versoes) == 1
    assert "champion" in versoes[0]["aliases"]

    # O alias resolve e o modelo empacotado responde de verdade.
    carregado = mlflow.pyfunc.load_model("models:/modelo-teste@champion")
    import pandas as pd

    resposta = carregado.predict(
        pd.DataFrame(
            {"question": ["O que é dormência de sementes?"], "topic": ["Sementes"]}
        ),
        params={"num_beams": 1, "max_new_tokens": 40},
    )
    assert len(resposta) == 1
    assert len(str(resposta[0]).split()) >= 3


@requer_modelo
def test_resolve_model_dir_localiza_os_pesos_pelo_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import tracking
    from app.model_registry import log_and_register_model

    monkeypatch.delenv(tracking.DISABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(tracking, "_state", {"configured": False, "enabled": True})
    monkeypatch.setattr(tracking, "MLFLOW_DIR", tmp_path)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(
        tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    tracking.configure(experiment_name="teste-artefatos")

    with tracking.start_run("treino-teste"):
        uri = log_and_register_model(
            DEFAULT_MODEL_DIR, registered_model_name=None, alias=None
        )

    # Sem nome registrado, o URI devolvido é o do "logged model"
    # (``models:/m-<id>``), que também precisa ser resolvível.
    assert uri.startswith("models:/m-")
    local = resolve_model_dir(uri)
    assert (local / "config.json").exists()
    # O MLflow nomeia o subdiretório com o nome base da origem, não com a
    # chave ``MLFLOW_MODEL_ARTIFACT`` usada em ``context.artifacts``.
    assert local.name == DEFAULT_MODEL_DIR.name
    assert local.parent.name == "artifacts"
    assert MLFLOW_MODEL_ARTIFACT == "model_dir"


def test_resolve_model_dir_encontra_pesos_de_lora_sem_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regressão: um modelo LoRA não tem ``config.json``.

    O que é empacotado são os adaptadores; a arquitetura base é resolvida
    a partir do ``adapter_config.json``, e os pesos base continuam no Hub.
    Procurar só pelo ``config.json`` fazia o playground falhar ao carregar
    o Gemma pelo Model Registry.
    """
    from app import model_registry

    baixado = tmp_path / "modelo"
    pesos = baixado / "artifacts" / "gaia-qa-embrapa"
    pesos.mkdir(parents=True)
    (pesos / "adapter_config.json").write_text("{}", encoding="utf-8")
    (pesos / "adapter_model.safetensors").write_bytes(b"")

    import mlflow

    monkeypatch.setattr(model_registry, "is_mlflow_uri", lambda referencia: True)
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda uri: str(baixado)
    )

    local = resolve_model_dir("models:/qa-embrapa-gaia@champion")
    assert local == pesos
    assert not (local / "config.json").exists()


def test_marcadores_de_peso_cobrem_as_duas_estrategias() -> None:
    from app.model_registry import WEIGHT_MARKERS

    assert "config.json" in WEIGHT_MARKERS
    assert "adapter_config.json" in WEIGHT_MARKERS
