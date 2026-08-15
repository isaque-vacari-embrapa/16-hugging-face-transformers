"""Testes da tokenização e do agrupamento em lote da família causal.

O ponto crítico é a máscara de rótulos: um modelo *decoder-only* recebe
prompt e resposta na mesma sequência, e treinar sobre o prompt ensinaria o
modelo a reproduzir o enunciado em vez de respondê-lo. Estes testes usam um
tokenizador falso, para não depender de baixar 8 GB de pesos.
"""

from __future__ import annotations

import pytest

from app.dataset import CausalCollator, build_causal_preprocess_fn
from app.models import training_config_for


class TokenizadorFalso:
    """Tokenizador mínimo: um token por palavra, marcadores próprios.

    Reproduz o suficiente da interface do Hugging Face para exercitar a
    tokenização: ``apply_chat_template``, chamada direta e ``decode``.
    """

    pad_token_id = 0
    unk_token_id = 1
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.vocabulario: dict[str, int] = {"<pad>": 0, "<unk>": 1, "<end_of_turn>": 2}

    def _id(self, palavra: str) -> int:
        return self.vocabulario.setdefault(palavra, len(self.vocabulario) + 10)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocabulario.get(token, self.unk_token_id)

    def apply_chat_template(
        self, mensagens, tokenize=False, add_generation_prompt=False
    ) -> str:
        corpo = " ".join(item["content"] for item in mensagens)
        return f"<inicio> {corpo} <modelo>"

    def _segmentar(self, texto: str) -> list[int]:
        """Separa o marcador de fim de turno, como faz um tokenizador real.

        O ``<end_of_turn>`` é um token especial do vocabulário do Gemma:
        ele é reconhecido mesmo colado à última palavra da resposta.
        """
        ids: list[int] = []
        for parte in texto.split("<end_of_turn>"):
            ids.extend(self._id(palavra) for palavra in parte.split())
            ids.append(self.vocabulario["<end_of_turn>"])
        return ids[:-1]

    def __call__(self, texto, add_special_tokens=True, **kwargs):
        if isinstance(texto, str):
            return {"input_ids": self._segmentar(texto)}
        return {"input_ids": [self._segmentar(item) for item in texto]}

    def decode(self, ids, skip_special_tokens=False) -> str:
        inverso = {valor: chave for chave, valor in self.vocabulario.items()}
        return " ".join(inverso.get(item, "?") for item in ids)


@pytest.fixture()
def tokenizador() -> TokenizadorFalso:
    return TokenizadorFalso()


@pytest.fixture()
def lote() -> dict[str, list[str]]:
    return {
        "question": ["Como plantar soja?"],
        "topic": ["Soja"],
        "answer": ["Prepare o solo antes da semeadura."],
    }


# ---------------------------------------------------------------------------
# Máscara de rótulos
# ---------------------------------------------------------------------------


def test_prompt_e_mascarado_e_apenas_a_resposta_tem_rotulo(
    tokenizador: TokenizadorFalso, lote: dict[str, list[str]]
) -> None:
    config = training_config_for("gaia")
    saida = build_causal_preprocess_fn(tokenizador, config)(lote)

    ids, rotulos = saida["input_ids"][0], saida["labels"][0]
    assert len(ids) == len(rotulos) == len(saida["attention_mask"][0])
    assert -100 in rotulos, "o prompt precisa estar mascarado"

    # As posições não mascaradas reproduzem exatamente a entrada.
    for indice, rotulo in enumerate(rotulos):
        if rotulo != -100:
            assert rotulo == ids[indice]

    aprendido = tokenizador.decode([r for r in rotulos if r != -100])
    assert "Prepare" in aprendido
    assert "Como" not in aprendido, "a pergunta não pode entrar na perda"


def test_resposta_termina_no_marcador_de_fim_de_turno(
    tokenizador: TokenizadorFalso, lote: dict[str, list[str]]
) -> None:
    """É o marcador que ensina a geração a parar."""
    config = training_config_for("gaia")
    saida = build_causal_preprocess_fn(tokenizador, config)(lote)
    rotulos = [r for r in saida["labels"][0] if r != -100]
    assert rotulos[-1] == tokenizador.convert_tokens_to_ids("<end_of_turn>")


def test_atencao_cobre_toda_a_sequencia(
    tokenizador: TokenizadorFalso, lote: dict[str, list[str]]
) -> None:
    saida = build_causal_preprocess_fn(tokenizador, training_config_for("gaia"))(lote)
    assert set(saida["attention_mask"][0]) == {1}


def test_sequencia_respeita_o_limite_da_janela(
    tokenizador: TokenizadorFalso,
) -> None:
    config = training_config_for("gaia", max_sequence_length=12)
    lote = {
        "question": ["Pergunta"],
        "topic": ["Soja"],
        "answer": [" ".join(f"palavra{n}" for n in range(200))],
    }
    saida = build_causal_preprocess_fn(tokenizador, config)(lote)
    assert len(saida["input_ids"][0]) <= 12
    assert len(saida["labels"][0]) == len(saida["input_ids"][0])


def test_varios_exemplos_no_mesmo_lote(tokenizador: TokenizadorFalso) -> None:
    lote = {
        "question": ["Pergunta curta?", "Outra pergunta diferente?"],
        "topic": ["Milho", "Soja"],
        "answer": ["Resposta.", "Outra resposta um pouco mais longa."],
    }
    saida = build_causal_preprocess_fn(tokenizador, training_config_for("gaia"))(lote)
    assert len(saida["input_ids"]) == len(saida["labels"]) == 2


# ---------------------------------------------------------------------------
# Agrupamento em lote
# ---------------------------------------------------------------------------


def test_colator_preenche_a_direita_e_ignora_o_preenchimento() -> None:
    """Preencher à esquerda deslocaria o alinhamento entre entrada e rótulo."""
    colator = CausalCollator(pad_token_id=0)
    lote = colator(
        [
            {
                "input_ids": [5, 6, 7],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 6, 7],
            },
            {"input_ids": [8], "attention_mask": [1], "labels": [8]},
        ]
    )
    assert lote["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert lote["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    # ``-100`` é o valor ignorado pela função de perda do PyTorch.
    assert lote["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]


def test_colator_devolve_tensores_inteiros() -> None:
    import torch

    lote = CausalCollator(pad_token_id=0)(
        [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}]
    )
    assert all(tensor.dtype == torch.long for tensor in lote.values())


# ---------------------------------------------------------------------------
# Tokens de parada
# ---------------------------------------------------------------------------


def test_parada_inclui_o_fim_de_turno_alem_do_eos(
    tokenizador: TokenizadorFalso,
) -> None:
    """Regressão: o ``<end_of_turn>`` não pode ficar de fora da parada.

    O Gemma declara ``eos_token_id: [1, 106]``, mas o tokenizador conhece
    apenas o ``1`` — e o ``transformers`` alinha a configuração ao
    tokenizador ao carregar, descartando o ``106``. Sem este resgate, a
    geração ignoraria o marcador que o treino ensina o modelo a emitir e
    seguiria até o limite de tokens, inventando um novo turno.
    """
    from app.generate import resolve_stop_token_ids

    tokenizador.vocabulario["<eos>"] = 7
    tokenizador.eos_token_id = 7

    parada = resolve_stop_token_ids(tokenizador)
    assert 7 in parada, "o eos do tokenizador precisa continuar valendo"
    assert tokenizador.convert_tokens_to_ids("<end_of_turn>") in parada


def test_parada_ignora_marcadores_ausentes_do_vocabulario(
    tokenizador: TokenizadorFalso,
) -> None:
    """Um modelo sem marcador de turno para apenas no próprio ``eos``."""
    from app.generate import resolve_stop_token_ids

    del tokenizador.vocabulario["<end_of_turn>"]
    tokenizador.eos_token_id = 7
    assert resolve_stop_token_ids(tokenizador) == [7]
