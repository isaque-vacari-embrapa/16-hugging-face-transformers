"""Testes das métricas de originalidade e da limpeza da saída gerada."""

from __future__ import annotations

from app.evaluation import (
    OriginalityIndex,
    distinct_n,
    ngrams,
    repetition_rate,
    tokenize_words,
)
from app.generate import clean_generated, trim_to_last_sentence

CORPUS = [
    "A melhor época de semeadura da soja depende da região e da cultivar.",
    "A adubação nitrogenada é imprescindível às lavouras irrigadas de arroz.",
]


def test_tokenize_words_normaliza_para_minusculas_sem_pontuacao() -> None:
    assert tokenize_words("Adubação, nitrogenada!") == ["adubação", "nitrogenada"]


def test_ngrams_de_texto_curto_e_vazio() -> None:
    assert ngrams(["a", "b"], 4) == []
    assert ngrams(["a", "b", "c", "d"], 4) == [("a", "b", "c", "d")]


def test_novel_ngram_ratio_zero_para_copia_literal() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    assert index.novel_ngram_ratio(CORPUS[0]) == 0.0


def test_novel_ngram_ratio_um_para_texto_inedito() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    novo = "O controle da lagarta exige monitoramento semanal das armadilhas."
    assert index.novel_ngram_ratio(novo) == 1.0


def test_novel_ngram_ratio_intermediario_para_texto_parcialmente_copiado() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    misto = (
        "A melhor época de semeadura da soja depende do monitoramento "
        "semanal das armadilhas instaladas."
    )
    taxa = index.novel_ngram_ratio(misto)
    assert 0.0 < taxa < 1.0


def test_novel_ngram_ratio_de_texto_vazio() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    assert index.novel_ngram_ratio("") == 0.0


def test_longest_copied_span_detecta_trecho_literal() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    texto = "Sabemos que a melhor época de semeadura da soja depende da região."
    # "a melhor época de semeadura da soja depende da região" = 10 palavras.
    assert index.longest_copied_span(texto) == 10


def test_longest_copied_span_zero_para_texto_inedito() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    assert index.longest_copied_span("Monitoramento semanal de armadilhas.") == 0


def test_longest_copied_span_de_texto_menor_que_o_ngrama() -> None:
    index = OriginalityIndex.from_texts(CORPUS)
    assert index.longest_copied_span("A soja") == 0


def test_distinct_n_detecta_respostas_repetidas() -> None:
    identicas = ["mesma resposta genérica"] * 4
    variadas = ["primeira resposta", "segunda alternativa", "terceira opção"]
    assert distinct_n(identicas, 2) < distinct_n(variadas, 2)


def test_distinct_n_de_lista_vazia() -> None:
    assert distinct_n([], 2) == 0.0


def test_repetition_rate_detecta_loop_de_geracao() -> None:
    repetido = "a b c d " * 5
    assert repetition_rate(repetido) > 0.5
    assert repetition_rate("uma frase sem qualquer repetição interna aqui") == 0.0


# ---------------------------------------------------------------------------
# Limpeza da saída do modelo
# ---------------------------------------------------------------------------


def test_clean_generated_remove_espaco_antes_da_pontuacao() -> None:
    assert clean_generated("A adubação é necessária , sim .") == (
        "A adubação é necessária, sim."
    )


def test_clean_generated_capitaliza_a_inicial() -> None:
    assert clean_generated("depende da região.") == "Depende da região."


def test_clean_generated_ajusta_parenteses() -> None:
    assert clean_generated("O pH ideal ( entre 5 e 6 ) varia.") == (
        "O pH ideal (entre 5 e 6) varia."
    )


def test_trim_to_last_sentence_descarta_frase_incompleta() -> None:
    texto = "A semeadura ocorre em outubro. Já a colheita depende da"
    assert trim_to_last_sentence(texto) == "A semeadura ocorre em outubro."


def test_trim_to_last_sentence_preserva_texto_completo() -> None:
    texto = "A semeadura ocorre em outubro."
    assert trim_to_last_sentence(texto) == texto


def test_trim_to_last_sentence_preserva_texto_sem_pontuacao_alguma() -> None:
    texto = "resposta sem pontuação final"
    assert trim_to_last_sentence(texto) == texto
