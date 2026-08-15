"""Testes de fumaça da interface de linha de comando.

Não executam conversão, treinamento nem inferência: verificam apenas que os
subcomandos estão registrados, que a ajuda é gerada e que as etapas de dados
funcionam de ponta a ponta em um diretório temporário.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from app.cli import SAMPLE_QUESTIONS, cli

COMANDOS_ESPERADOS = {
    "ask",
    "compare",
    "convert",
    "dataset",
    "evaluate",
    "extract",
    "info",
    "pipeline",
    "playground",
    "samples",
    "train",
}

RESPOSTA = (
    "A melhor época de semeadura depende da região, do ciclo da cultivar e "
    "da disponibilidade de água no solo durante a floração."
)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_ajuda_lista_todos_os_subcomandos(runner: CliRunner) -> None:
    resultado = runner.invoke(cli, ["--help"])
    assert resultado.exit_code == 0
    assert COMANDOS_ESPERADOS <= set(resultado.output.split())


@pytest.mark.parametrize("comando", sorted(COMANDOS_ESPERADOS))
def test_ajuda_de_cada_subcomando(runner: CliRunner, comando: str) -> None:
    resultado = runner.invoke(cli, [comando, "--help"])
    assert resultado.exit_code == 0
    assert "Usage:" in resultado.output


def test_extract_e_dataset_em_diretorio_temporario(
    runner: CliRunner, tmp_path: Path
) -> None:
    interim = tmp_path / "interim"
    interim.mkdir()
    conteudo = "\n\n".join(
        f"## {numero} Pergunta número {numero} sobre o cultivo?\n\n{RESPOSTA}"
        for numero in range(1, 41)
    )
    (interim / "soja.md").write_text(conteudo, encoding="utf-8")

    saida = tmp_path / "qa.jsonl"
    # ``--report`` mantém o teste hermético: sem ele, a extração sobrescreveria
    # o diagnóstico real do projeto em reports/.
    resultado = runner.invoke(
        cli,
        [
            "extract",
            "--interim-dir",
            str(interim),
            "--output",
            str(saida),
            "--report",
            str(tmp_path / "extraction.json"),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    assert (tmp_path / "extraction.json").exists()
    registros = [
        json.loads(linha)
        for linha in saida.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]
    assert len(registros) == 40
    assert registros[0]["topic"] == "Soja"

    splits = tmp_path / "splits"
    resultado = runner.invoke(
        cli,
        [
            "dataset",
            "--input",
            str(saida),
            "--splits-dir",
            str(splits),
            "--validation-size",
            "0.2",
            "--test-size",
            "0.1",
            "--report",
            str(tmp_path / "dataset.json"),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    for nome in ("train", "validation", "test"):
        assert (splits / f"{nome}.jsonl").exists()
    assert (tmp_path / "dataset.json").exists()


def test_extract_falha_com_mensagem_clara_se_faltar_o_markdown(
    runner: CliRunner, tmp_path: Path
) -> None:
    resultado = runner.invoke(
        cli,
        [
            "extract",
            "--interim-dir",
            str(tmp_path / "inexistente"),
            "--output",
            str(tmp_path / "qa.jsonl"),
            "--report",
            str(tmp_path / "extraction.json"),
        ],
    )
    assert resultado.exit_code != 0
    assert isinstance(resultado.exception, FileNotFoundError)
    assert "conversão" in str(resultado.exception)


def test_dataset_falha_com_mensagem_clara_se_faltar_o_jsonl(
    runner: CliRunner, tmp_path: Path
) -> None:
    resultado = runner.invoke(cli, ["dataset", "--input", str(tmp_path / "nada.jsonl")])
    assert resultado.exit_code != 0
    assert "extract" in str(resultado.exception)


def test_cenario_de_exemplo_cobre_temas_variados() -> None:
    temas = {tema for tema, _ in SAMPLE_QUESTIONS}
    assert len(SAMPLE_QUESTIONS) >= 8
    assert len(temas) >= 6
    assert all(pergunta.endswith("?") for _, pergunta in SAMPLE_QUESTIONS)
