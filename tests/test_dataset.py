"""Testes da preparação do conjunto de dados e do prompt do modelo."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    TASK_PREFIX,
    DatasetSplitConfig,
    TrainingConfig,
    build_prompt,
    ensure_dirs,
)
from app.dataset import (
    SPLIT_NAMES,
    build_preprocess_fn,
    dataset_statistics,
    split_dataset,
    truncate_to_sentence,
    write_json,
)


class TokenizadorFalso:
    """Tokenizador mínimo: um token por palavra.

    Evita baixar o SentencePiece do modelo real nos testes, mantendo o
    comportamento relevante (contagem de tokens e truncamento).
    """

    def __call__(
        self,
        text=None,
        text_target=None,
        max_length=None,
        truncation=False,
        add_special_tokens=True,
    ):
        entrada = text if text is not None else text_target
        textos = [entrada] if isinstance(entrada, str) else list(entrada)
        ids = [item.split() for item in textos]
        if truncation and max_length:
            ids = [item[:max_length] for item in ids]
        ids = [[abs(hash(token)) % 1000 for token in item] for item in ids]
        resultado = {"input_ids": ids[0] if isinstance(entrada, str) else ids}
        if not isinstance(entrada, str):
            resultado["attention_mask"] = [[1] * len(item) for item in ids]
        return resultado

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(item) for item in ids)


@pytest.fixture()
def tokenizador() -> TokenizadorFalso:
    return TokenizadorFalso()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_build_prompt_inclui_tema_e_pergunta() -> None:
    prompt = build_prompt("Qual a melhor época de semeadura?", "Soja")
    assert prompt == (
        f"{TASK_PREFIX}tema: soja | pergunta: Qual a melhor época de semeadura?"
    )


def test_build_prompt_usa_tema_padrao_quando_ausente() -> None:
    assert "tema: geral |" in build_prompt("Como irrigar?")


def test_build_prompt_normaliza_espacos() -> None:
    prompt = build_prompt("Como   irrigar\n a lavoura?", "  Gado de Leite ")
    assert "tema: gado de leite | pergunta: Como irrigar a lavoura?" in prompt


# ---------------------------------------------------------------------------
# Truncamento
# ---------------------------------------------------------------------------


def test_truncate_to_sentence_mantem_texto_curto(tokenizador) -> None:
    texto = "Uma resposta curta."
    assert truncate_to_sentence(texto, tokenizador, 20) == texto


def test_truncate_to_sentence_corta_no_fim_da_frase(tokenizador) -> None:
    texto = "Primeira frase com cinco palavras. Segunda frase bem mais longa aqui."
    # Orçamento suficiente apenas para a primeira frase (6 palavras).
    resultado = truncate_to_sentence(texto, tokenizador, 8)
    assert resultado == "Primeira frase com cinco palavras."


def test_truncate_to_sentence_corta_por_tokens_se_a_frase_nao_couber(
    tokenizador,
) -> None:
    texto = "uma frase única bastante longa que não cabe no orçamento de tokens."
    resultado = truncate_to_sentence(texto, tokenizador, 4)
    assert resultado
    assert len(resultado.split()) <= 3


# ---------------------------------------------------------------------------
# Tokenização em lote
# ---------------------------------------------------------------------------


def test_build_preprocess_fn_gera_entradas_e_rotulos(tokenizador) -> None:
    config = TrainingConfig(max_source_length=32, max_target_length=16)
    preprocess = build_preprocess_fn(tokenizador, config)
    lote = {
        "question": ["Qual a melhor época de semeadura?"],
        "topic": ["Soja"],
        "answer": ["A melhor época depende da região e da cultivar."],
    }
    saida = preprocess(lote)
    assert set(saida) == {"input_ids", "attention_mask", "labels"}
    assert len(saida["input_ids"]) == 1
    assert len(saida["labels"]) == 1
    assert len(saida["input_ids"][0]) <= 32
    assert len(saida["labels"][0]) <= 16


# ---------------------------------------------------------------------------
# Partições
# ---------------------------------------------------------------------------


def _dataset_falso(n: int = 100):
    datasets = pytest.importorskip("datasets")
    return datasets.Dataset.from_dict(
        {
            "question": [f"Pergunta {i}?" for i in range(n)],
            "answer": [f"Resposta {i}." for i in range(n)],
            "topic": ["Soja" if i % 2 else "Milho" for i in range(n)],
        }
    )


def test_split_dataset_respeita_as_proporcoes() -> None:
    splits = split_dataset(
        _dataset_falso(100),
        DatasetSplitConfig(validation_size=0.1, test_size=0.05, seed=1),
    )
    assert set(splits) == set(SPLIT_NAMES)
    # O arredondamento é da própria biblioteca datasets; o que importa é que
    # nenhum exemplo seja perdido ou duplicado e as proporções sejam respeitadas.
    assert sum(len(splits[name]) for name in SPLIT_NAMES) == 100
    assert len(splits["validation"]) == pytest.approx(10, abs=1)
    assert len(splits["test"]) == pytest.approx(5, abs=1)
    assert len(splits["train"]) == pytest.approx(85, abs=2)


def test_split_dataset_nao_vaza_exemplos_entre_particoes() -> None:
    splits = split_dataset(_dataset_falso(100), DatasetSplitConfig(seed=5))
    conjuntos = [set(splits[name]["question"]) for name in SPLIT_NAMES]
    assert conjuntos[0].isdisjoint(conjuntos[1])
    assert conjuntos[0].isdisjoint(conjuntos[2])
    assert conjuntos[1].isdisjoint(conjuntos[2])


def test_split_dataset_e_reprodutivel_com_a_mesma_semente() -> None:
    config = DatasetSplitConfig(seed=7)
    primeiro = split_dataset(_dataset_falso(60), config)
    segundo = split_dataset(_dataset_falso(60), config)
    assert primeiro["test"]["question"] == segundo["test"]["question"]


def test_split_dataset_rejeita_proporcoes_invalidas() -> None:
    with pytest.raises(ValueError):
        split_dataset(
            _dataset_falso(20), DatasetSplitConfig(validation_size=0.9, test_size=0.2)
        )


def test_dataset_statistics_conta_exemplos_e_temas() -> None:
    splits = split_dataset(_dataset_falso(100), DatasetSplitConfig(seed=3))
    estatisticas = dataset_statistics(splits)
    assert estatisticas["train"]["examples"] == len(splits["train"])
    assert estatisticas["train"]["topics"] == 2
    assert set(estatisticas["topics"]) == {"Milho", "Soja"}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def test_write_json_cria_diretorio_e_preserva_acentos(tmp_path: Path) -> None:
    destino = tmp_path / "sub" / "relatorio.json"
    write_json(destino, {"tema": "Produção Orgânica"})
    conteudo = destino.read_text(encoding="utf-8")
    assert "Produção Orgânica" in conteudo


def test_ensure_dirs_e_idempotente(tmp_path: Path) -> None:
    alvo = tmp_path / "a" / "b"
    ensure_dirs(alvo)
    ensure_dirs(alvo)
    assert alvo.is_dir()
