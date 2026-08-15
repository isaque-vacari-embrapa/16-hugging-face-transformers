"""Configuração compartilhada dos testes.

As etapas do pipeline abrem execuções no MLflow. Sem isolamento, rodar a
suíte poluiria o ``mlflow/mlflow.db`` do projeto com execuções sintéticas —
o mesmo problema que já havia acontecido com os relatórios em ``reports/``.
Por isso o rastreamento é desligado por padrão em todos os testes; quem
precisa dele (``tests/test_tracking.py``) reativa apontando para um banco
temporário.
"""

from __future__ import annotations

import pytest

from app import tracking


@pytest.fixture(autouse=True, scope="session")
def mlflow_desligado():
    """Desliga o rastreamento no MLflow durante toda a sessão de testes.

    O escopo precisa ser de sessão, e não de função: fixtures de escopo mais
    amplo (como a que instancia o playground em ``test_streamlit_app.py``)
    são preparadas antes das de função e, com escopo menor aqui, executariam
    com o rastreamento ainda ligado.
    """
    patch = pytest.MonkeyPatch()
    patch.setenv(tracking.DISABLE_ENV_VAR, "1")
    yield
    patch.undo()
