"""Testes da normalização dos temas (culturas) da coleção."""

from __future__ import annotations

import pytest

from app.topics import (
    slugify,
    topic_label,
    topic_label_from_filename,
    topic_slug_from_filename,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Soja-o-produtor-pergunta-a-embrapa-responde.pdf", "soja"),
        ("Gado-de-Leite-o-produtor-pergunta-a-embrapa-responde.pdf", "gado-de-leite"),
        (
            "Producao-Organica-de-Hortalicas-o-produtor-pergunta-a-embrapa-responde.pdf",
            "producao-organica-de-hortalicas",
        ),
        ("soja.md", "soja"),
    ],
)
def test_topic_slug_from_filename(filename: str, expected: str) -> None:
    assert topic_slug_from_filename(filename) == expected


def test_topic_label_usa_acentuacao_conhecida() -> None:
    assert topic_label("algodao") == "Algodão"
    assert topic_label("producao-organica-de-hortalicas") == (
        "Produção Orgânica de Hortaliças"
    )


def test_topic_label_corrige_erro_de_grafia_do_arquivo() -> None:
    # O nome do PDF original traz "Caprionos" em vez de "Caprinos".
    assert (
        topic_label_from_filename(
            "Caprionos-e-Ovinos-de-Corte-o-produtor-pergunta-a-embrapa-responde.pdf"
        )
        == "Caprinos e Ovinos de Corte"
    )


def test_topic_label_gera_rotulo_para_tema_desconhecido() -> None:
    assert topic_label("cana-de-acucar") == "Cana De Acucar"


def test_slugify_remove_acentos_e_pontuacao() -> None:
    assert slugify("Produção Orgânica, de Hortaliças!") == (
        "producao-organica-de-hortalicas"
    )


def test_topic_slug_de_nome_sem_sufixo_conhecido() -> None:
    assert topic_slug_from_filename("arquivo-qualquer.pdf") == "arquivo-qualquer"
