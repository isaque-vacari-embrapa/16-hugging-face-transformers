"""Testes do pipeline de conversão PDF -> Markdown.

A Docling é substituída por um conversor falso: o objetivo aqui é verificar
a lógica de nomes de arquivo, o reaproveitamento de conversões anteriores e
o relatório, não o modelo de análise de layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.convert import (
    ConversionResult,
    convert_all,
    convert_pdf,
    list_pdfs,
    markdown_path_for,
)


@dataclass
class DocumentoFalso:
    markdown: str

    def export_to_markdown(self) -> str:
        return self.markdown


@dataclass
class ResultadoFalso:
    document: DocumentoFalso


class ConversorFalso:
    """Substitui o ``DocumentConverter`` da Docling nos testes."""

    def __init__(self) -> None:
        self.chamadas: list[Path] = []

    def convert(self, pdf_path: Path) -> ResultadoFalso:
        self.chamadas.append(Path(pdf_path))
        nome = Path(pdf_path).stem
        return ResultadoFalso(
            DocumentoFalso(f"## 1 Pergunta de {nome}?\n\nResposta.\n")
        )


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    diretorio = tmp_path / "raw"
    diretorio.mkdir()
    for nome in ("Soja", "Gado-de-Leite"):
        (diretorio / f"{nome}-o-produtor-pergunta-a-embrapa-responde.pdf").write_bytes(
            b"%PDF-1.4"
        )
    return diretorio


def test_markdown_path_for_usa_o_slug_do_tema(tmp_path: Path) -> None:
    destino = markdown_path_for(
        Path("Gado-de-Leite-o-produtor-pergunta-a-embrapa-responde.pdf"), tmp_path
    )
    assert destino == tmp_path / "gado-de-leite.md"


def test_list_pdfs_em_ordem_alfabetica(raw_dir: Path) -> None:
    nomes = [caminho.name for caminho in list_pdfs(raw_dir)]
    assert nomes == sorted(nomes)
    assert len(nomes) == 2


def test_list_pdfs_falha_se_o_diretorio_nao_existe(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_pdfs(tmp_path / "inexistente")


def test_convert_pdf_grava_markdown_e_relata(raw_dir: Path, tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    conversor = ConversorFalso()
    pdf = raw_dir / "Soja-o-produtor-pergunta-a-embrapa-responde.pdf"

    resultado = convert_pdf(pdf, interim_dir=interim, converter=conversor)

    assert isinstance(resultado, ConversionResult)
    assert resultado.skipped is False
    assert resultado.topic_slug == "soja"
    assert resultado.topic_label == "Soja"
    destino = interim / "soja.md"
    assert destino.read_text(encoding="utf-8").startswith("## 1 Pergunta de Soja")
    assert resultado.as_dict()["markdown"] == "soja.md"


def test_convert_pdf_reaproveita_conversao_existente(
    raw_dir: Path, tmp_path: Path
) -> None:
    interim = tmp_path / "interim"
    interim.mkdir()
    (interim / "soja.md").write_text("conteúdo anterior", encoding="utf-8")
    conversor = ConversorFalso()
    pdf = raw_dir / "Soja-o-produtor-pergunta-a-embrapa-responde.pdf"

    resultado = convert_pdf(pdf, interim_dir=interim, converter=conversor)

    assert resultado.skipped is True
    assert conversor.chamadas == []
    assert (interim / "soja.md").read_text(encoding="utf-8") == "conteúdo anterior"


def test_convert_pdf_reconverte_com_overwrite(raw_dir: Path, tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    interim.mkdir()
    (interim / "soja.md").write_text("conteúdo anterior", encoding="utf-8")
    conversor = ConversorFalso()
    pdf = raw_dir / "Soja-o-produtor-pergunta-a-embrapa-responde.pdf"

    resultado = convert_pdf(
        pdf, interim_dir=interim, converter=conversor, overwrite=True
    )

    assert resultado.skipped is False
    assert conversor.chamadas == [pdf]
    assert "Pergunta de Soja" in (interim / "soja.md").read_text(encoding="utf-8")


def test_convert_all_percorre_todos_os_pdfs(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversor = ConversorFalso()
    monkeypatch.setattr("app.convert.build_converter", lambda **_: conversor)
    interim = tmp_path / "interim"

    resultados = list(convert_all(raw_dir=raw_dir, interim_dir=interim))

    assert {item.topic_slug for item in resultados} == {"soja", "gado-de-leite"}
    assert sorted(caminho.name for caminho in interim.glob("*.md")) == [
        "gado-de-leite.md",
        "soja.md",
    ]


def test_convert_all_nao_carrega_a_docling_quando_tudo_esta_convertido(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interim = tmp_path / "interim"
    interim.mkdir()
    for nome in ("soja.md", "gado-de-leite.md"):
        (interim / nome).write_text("já convertido", encoding="utf-8")

    def falhar(**_):  # pragma: no cover - não deve ser chamado
        raise AssertionError("build_converter não deveria ser chamado")

    monkeypatch.setattr("app.convert.build_converter", falhar)
    resultados = list(convert_all(raw_dir=raw_dir, interim_dir=interim))
    assert all(item.skipped for item in resultados)


def test_convert_all_falha_sem_pdfs(tmp_path: Path) -> None:
    vazio = tmp_path / "raw"
    vazio.mkdir()
    with pytest.raises(FileNotFoundError):
        list(convert_all(raw_dir=vazio, interim_dir=tmp_path / "interim"))
