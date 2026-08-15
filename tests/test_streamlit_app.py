"""Teste de integração do playground em Streamlit.

Usa ``streamlit.testing.v1.AppTest``, que executa o script da aplicação em
processo e expõe os elementos renderizados. O teste é ignorado quando o
modelo ajustado ainda não existe, porque carregá-lo é pré-requisito da
renderização e não faz sentido baixar o modelo base só para isto.
"""

from __future__ import annotations

import pytest

from app.config import DEFAULT_MODEL_DIR
from app.models import AVAILABLE_MODELS, DEFAULT_MODEL_KEY, resolve_spec

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = "src/app/streamlit_app.py"

modelo_ausente = not (DEFAULT_MODEL_DIR / "config.json").exists()
requer_modelo = pytest.mark.skipif(
    modelo_ausente,
    reason=(
        "requer o modelo ajustado em "
        f"{DEFAULT_MODEL_DIR}; execute `poetry run qa-embrapa train`"
    ),
)


@pytest.fixture(scope="module")
def app() -> AppTest:
    """Executa o script do playground uma única vez por módulo."""
    instancia = AppTest.from_file(APP_PATH, default_timeout=600)
    instancia.run()
    return instancia


@requer_modelo
def test_playground_renderiza_sem_excecao(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value.endswith("O produtor pergunta, a Embrapa responde")
    rotulos = [tab.label for tab in app.tabs]
    assert rotulos == [
        "Playground",
        "Avaliação",
        "Comparação",
        "MLflow",
        "Sobre o projeto",
    ]


@requer_modelo
def test_playground_expoe_os_controles_de_decodificacao(app: AppTest) -> None:
    assert len(app.sidebar.radio) == 1
    # modelo, origem dos pesos e tema
    rotulos = [item.label for item in app.sidebar.selectbox]
    assert rotulos == ["Modelo pré-treinado", "Origem dos pesos", "Cultura da pergunta"]
    # tamanho máximo, penalidade de repetição, alternativas e feixes/temperatura
    assert len(app.sidebar.slider) >= 4


@requer_modelo
def test_playground_permite_escolher_o_modelo(app: AppTest) -> None:
    """A seleção do modelo é o primeiro controle da barra lateral.

    O ``AppTest`` expõe as opções já formatadas por ``format_func`` — o
    rótulo legível, que é o que o usuário vê — mas ``value`` continua sendo
    a chave curta, que é o que o código recebe.
    """
    seletor = app.sidebar.selectbox[0]
    assert len(seletor.options) == len(AVAILABLE_MODELS)
    for spec in AVAILABLE_MODELS.values():
        assert any(spec.label in opcao for opcao in seletor.options)
    assert seletor.value == DEFAULT_MODEL_KEY


@requer_modelo
def test_playground_permite_escolher_a_origem_dos_pesos(app: AppTest) -> None:
    origem = app.sidebar.selectbox[1]
    assert "Diretório local" in origem.options
    assert any("MLflow" in opcao for opcao in origem.options)
    # O campo de texto reflete a origem escolhida.
    assert app.sidebar.text_input[0].label == "Referência do modelo"


@requer_modelo
def test_origem_dos_pesos_acompanha_o_modelo_escolhido(app: AppTest) -> None:
    """Trocar de modelo precisa trocar o diretório oferecido por padrão."""
    esperado = str(resolve_spec(app.sidebar.selectbox[0].value).output_dir)
    assert app.sidebar.text_input[0].value == esperado


@requer_modelo
def test_playground_gera_resposta_ao_clicar_no_botao() -> None:
    instancia = AppTest.from_file(APP_PATH, default_timeout=600)
    instancia.run()
    instancia.text_area[0].set_value("Quais os sintomas do mal-do-panamá na bananeira?")
    botao = next(item for item in instancia.button if item.label == "Gerar resposta")
    botao.click().run()

    assert not instancia.exception
    assert instancia.success, "nenhuma resposta foi exibida"
    assert len(instancia.success[0].value.split()) >= 5


@requer_modelo
def test_playground_avisa_quando_a_pergunta_esta_vazia() -> None:
    instancia = AppTest.from_file(APP_PATH, default_timeout=600)
    instancia.run()
    instancia.text_area[0].set_value("   ")
    botao = next(item for item in instancia.button if item.label == "Gerar resposta")
    botao.click().run()

    assert not instancia.exception
    assert any("Escreva uma pergunta" in item.value for item in instancia.warning)
