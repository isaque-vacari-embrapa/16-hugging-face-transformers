"""Testes da extração dos pares pergunta/resposta.

Cada teste reproduz um defeito real de diagramação observado no Markdown
que a Docling produz a partir dos PDFs da coleção.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extract import (
    assemble_answer,
    clean_text,
    dedupe_pairs,
    extract_from_markdown,
    join_fragments,
    normalize_question,
    parse_blocks,
    summarize,
)

RESPOSTA = (
    "Dormência é uma condição fisiológica que impede a germinação de "
    "sementes inteiras e viáveis, mesmo em presença de todos os fatores "
    "ambientais favoráveis à germinação."
)


def escrever(tmp_path: Path, nome: str, conteudo: str) -> Path:
    """Grava um Markdown temporário e devolve o caminho."""
    caminho = tmp_path / nome
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Limpeza de texto
# ---------------------------------------------------------------------------


def test_clean_text_remove_comentarios_enfase_e_espacos() -> None:
    assert clean_text("<!-- image --> **Texto**   com   espaços") == (
        "Texto com espaços"
    )


def test_clean_text_remove_hifen_opcional_com_e_sem_espaco() -> None:
    # "germi<soft hyphen>nação" e "prin<soft hyphen> cipais"
    assert clean_text("germi\u00adnação") == "germinação"
    assert clean_text("prin\u00ad cipais") == "principais"


def test_clean_text_desfaz_entidades_html() -> None:
    assert clean_text("Salsola &amp; Asch.") == "Salsola & Asch."


def test_join_fragments_reconstroi_palavra_partida_por_hifen() -> None:
    assert join_fragments(["para a agri-", "cultura?"]) == "para a agricultura?"


def test_join_fragments_usa_espaco_quando_nao_ha_hifen() -> None:
    assert join_fragments(["Sementes dormentes têm", "maior longevidade?"]) == (
        "Sementes dormentes têm maior longevidade?"
    )


def test_normalize_question_capitaliza_e_garante_interrogacao() -> None:
    assert normalize_question("qual o espaçamento ideal") == (
        "Qual o espaçamento ideal?"
    )


# ---------------------------------------------------------------------------
# Blocos e respostas
# ---------------------------------------------------------------------------


def test_parse_blocks_classifica_titulos_itens_e_paragrafos() -> None:
    blocos = parse_blocks(
        "## 1 Pergunta?\n\nParágrafo.\n\n-  Primeiro item\n\n| a | b |\n"
    )
    assert [(bloco.kind, bloco.is_bullet) for bloco in blocos] == [
        ("heading", False),
        ("body", False),
        ("body", True),
    ]


def test_assemble_answer_transforma_itens_em_frases() -> None:
    blocos = parse_blocks(
        "Os cuidados são:\n\n-  escolher o herbicida\n\n-  usar a dose correta\n"
    )
    assert assemble_answer(blocos) == (
        "Os cuidados são: escolher o herbicida. usar a dose correta."
    )


def test_assemble_answer_descarta_referencias_bibliograficas() -> None:
    blocos = parse_blocks(
        f"{RESPOSTA}\n\nSMITH, H. Phytochromes. Nature, v. 407, 2000. "
        "DOI: 10.1038/35036500.\n"
    )
    resposta = assemble_answer(blocos)
    assert "DOI" not in resposta
    assert resposta.startswith("Dormência é uma condição")


# ---------------------------------------------------------------------------
# Padrões de numeração
# ---------------------------------------------------------------------------


def test_numeracao_a_esquerda(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 147 O que é dormência de sementes?\n\n{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 1
    assert pares[0].number == 147
    assert pares[0].question == "O que é dormência de sementes?"
    assert pares[0].id == "sementes-147"
    assert pares[0].topic == "Sementes"


def test_numeracao_a_direita(tmp_path: Path) -> None:
    # Padrão do volume da Banana: o número é impresso após o enunciado.
    caminho = escrever(
        tmp_path,
        "banana.md",
        f"## Onde se originou a maioria das cultivares de bananeira? 2\n\n{RESPOSTA}\n",
    )
    pares, _ = extract_from_markdown(caminho)
    assert pares[0].number == 2
    assert pares[0].question == (
        "Onde se originou a maioria das cultivares de bananeira?"
    )


def test_numero_isolado_em_titulo_proprio(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 183\n\n## Por que sementes dormentes precisam de oxigênio?\n\n{RESPOSTA}\n",
    )
    pares, _ = extract_from_markdown(caminho)
    assert pares[0].number == 183
    assert pares[0].question == ("Por que sementes dormentes precisam de oxigênio?")


# ---------------------------------------------------------------------------
# Defeitos de diagramação
# ---------------------------------------------------------------------------


def test_enunciado_partido_em_dois_titulos(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        "## 193 Sementes que apresentam dormência também apresentam\n\n"
        f"## maior longevidade?\n\n{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 1
    assert pares[0].number == 193
    assert pares[0].question == (
        "Sementes que apresentam dormência também apresentam maior longevidade?"
    )


def test_enunciado_invertido_pela_ordem_de_leitura(tmp_path: Path) -> None:
    # O fragmento com o número vem primeiro por causa da diagramação.
    caminho = escrever(
        tmp_path,
        "sementes.md",
        "## 194 semente e a dormência?\n\n"
        "## Há alguma correlação entre caracteres morfológicos da\n\n"
        f"{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 1
    assert pares[0].number == 194
    assert pares[0].question == (
        "Há alguma correlação entre caracteres morfológicos da semente e a dormência?"
    )
    assert pares[0].repaired is True


def test_inicio_do_enunciado_caido_no_corpo_do_texto(tmp_path: Path) -> None:
    # "... para a agri-" ficou como parágrafo e "192 cultura?" como título.
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 191 Pergunta anterior qualquer?\n\n{RESPOSTA}\n\n"
        "Qual é a importância de conhecer a dormência de ervas daninhas "
        "para a agri-\n\n"
        f"## 192 cultura?\n\n{RESPOSTA}\n",
    )
    pares, _ = extract_from_markdown(caminho)
    numeros = {par.number: par.question for par in pares}
    assert numeros[192] == (
        "Qual é a importância de conhecer a dormência de ervas daninhas "
        "para a agricultura?"
    )


def test_enunciado_deixado_no_corpo_do_texto_e_recuperado(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 189 Primeira pergunta do capítulo?\n\n{RESPOSTA}\n\n"
        "A dormência deve ser considerada em programas de melhoramento?\n\n"
        f"{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 2
    assert stats.recovered_from_body == 1
    assert pares[1].question == (
        "A dormência deve ser considerada em programas de melhoramento?"
    )
    # O número é inferido a partir do anterior da sequência.
    assert pares[1].number == 190
    assert pares[1].number_inferred is True


def test_recuperacao_pode_ser_desligada(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 189 Primeira pergunta do capítulo?\n\n{RESPOSTA}\n\n"
        "A dormência deve ser considerada em programas de melhoramento?\n\n"
        f"{RESPOSTA}\n",
    )
    pares, _ = extract_from_markdown(caminho, recover_inline_questions=False)
    assert len(pares) == 1


# ---------------------------------------------------------------------------
# Rejeições
# ---------------------------------------------------------------------------


def test_titulo_de_capitulo_nao_vira_pergunta(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 3 Dormência\n\nTexto introdutório do capítulo, sem pergunta.\n\n"
        f"## 147 O que é dormência de sementes?\n\n{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 1
    assert stats.rejected_no_question_mark == 1
    assert pares[0].number == 147


def test_titulo_de_capitulo_colado_ao_enunciado_e_descartado(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "caju.md",
        "## Aproveitamento Industrial\n\n"
        "## Deborah dos Santos Garruti Janice Ribeiro Lima Antonio Calixto Lima "
        "Francisco Fábio de Assis Paiva\n\n"
        f"## 250 Como se faz a castanha de caju torrada?\n\n{RESPOSTA}\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert stats.accepted == 1
    assert pares[0].question == "Como se faz a castanha de caju torrada?"


def test_bibliografia_final_nao_gera_par(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "sementes.md",
        f"## 147 O que é dormência de sementes?\n\n{RESPOSTA}\n\n"
        "## Referências\n\n"
        "SMITH, H. Phytochromes. Nature, v. 407, p. 585-591, 2000.\n",
    )
    pares, _ = extract_from_markdown(caminho)
    assert len(pares) == 1
    assert "Phytochromes" not in pares[0].answer


def test_resposta_curta_que_so_remete_a_tabela_e_rejeitada(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "algodao.md",
        "## 10 Qual o teor médio de gossipol nos subprodutos do algodão?\n\n"
        "Esses teores encontram-se na Tabela 9, a seguir.\n",
    )
    pares, stats = extract_from_markdown(caminho)
    assert pares == []
    assert stats.rejected_exhibit_only == 1


def test_pergunta_longa_demais_e_rejeitada(tmp_path: Path) -> None:
    enunciado = "Palavra " * 60
    caminho = escrever(
        tmp_path, "soja.md", f"## 10 {enunciado.strip()}?\n\n{RESPOSTA}\n"
    )
    pares, stats = extract_from_markdown(caminho)
    assert pares == []
    assert stats.rejected_long_question == 1


def test_resposta_ausente_e_rejeitada(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path, "soja.md", "## 10 Uma pergunta sem resposta?\n\nSim.\n"
    )
    pares, stats = extract_from_markdown(caminho)
    assert pares == []
    assert stats.rejected_short_answer == 1


# ---------------------------------------------------------------------------
# Agregações
# ---------------------------------------------------------------------------


def test_dedupe_pairs_remove_pergunta_repetida_no_mesmo_tema(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path,
        "soja.md",
        f"## 10 Qual a melhor época de semeadura?\n\n{RESPOSTA}\n\n"
        f"## 11 Qual a melhor época de semeadura?\n\n{RESPOSTA}\n",
    )
    pares, _ = extract_from_markdown(caminho)
    assert len(pares) == 2
    assert len(dedupe_pairs(pares)) == 1


def test_summarize_conta_temas_e_tamanhos(tmp_path: Path) -> None:
    caminho = escrever(
        tmp_path, "soja.md", f"## 10 Qual a melhor época de semeadura?\n\n{RESPOSTA}\n"
    )
    pares, _ = extract_from_markdown(caminho)
    resumo = summarize(pares)
    assert resumo["total"] == 1
    assert resumo["topics"] == 1
    assert resumo["pairs_per_topic"] == {"Soja": 1}


def test_summarize_de_lista_vazia() -> None:
    assert summarize([]) == {"total": 0}


@pytest.mark.parametrize("numero", [1, 42, 501])
def test_id_do_par_combina_tema_e_numero(tmp_path: Path, numero: int) -> None:
    caminho = escrever(
        tmp_path, "milho.md", f"## {numero} Qual o espaçamento ideal?\n\n{RESPOSTA}\n"
    )
    pares, _ = extract_from_markdown(caminho)
    assert pares[0].id == f"milho-{numero}"
