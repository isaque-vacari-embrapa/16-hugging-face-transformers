"""Testes da comparação entre modelos e do critério de recomendação.

Trabalham sobre dicionários montados à mão, no mesmo formato dos
relatórios de avaliação: o objetivo é fixar a regra de escolha, não
reexecutar a avaliação.
"""

from __future__ import annotations

from app.comparison import (
    QUALITY_WEIGHTS,
    build_summary,
    quality_index,
    recommend,
    render_markdown,
)


def modelo(
    chave: str,
    rougeL: float,
    rouge1: float = 20.0,
    rouge2: float = 5.0,
    bleu: float = 1.0,
    chrf: float = 15.0,
    segundos: float = 0.5,
    presets: dict | None = None,
) -> dict:
    """Monta a estrutura que ``collect`` produziria para um modelo.

    ``presets`` permite montar um modelo cuja melhor decodificação não é a
    de referência — o caso que motivou comparar melhor-contra-melhor.
    """
    qualidade = {
        "rougeL": rougeL,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "bleu": bleu,
        "chrf": chrf,
    }
    custo = {"seconds_per_answer": segundos, "seconds_total": segundos * 200}
    if presets is None:
        presets = {
            "beam_search": {
                "quality": qualidade,
                "length": {"words_mean": 40.0},
                "cost": custo,
            }
        }
    return {
        "key": chave,
        "label": chave.upper(),
        "checkpoint": f"org/{chave}",
        "family": "seq2seq",
        "parameters": "223M",
        "strategy": "ajuste-completo",
        "test_examples": 200,
        "training": {
            "elapsed_seconds": 1920.0,
            "epochs": 5.0,
            "trainable_parameters": 223_000_000,
            "total_parameters": 223_000_000,
            "validation_loss": 2.7,
        },
        "reference_preset": "beam_search",
        "quality": qualidade,
        "originality": {
            "novel_4gram_ratio_mean": 0.72,
            "longest_copied_span_mean": 5.5,
            "distinct_2": 0.62,
            "self_repetition_4gram_mean": 0.002,
        },
        "length": {"words_mean": 40.0, "empty_answers": 0},
        "cost": custo,
        "presets": presets,
        "examples": [
            {
                "topic": "Soja",
                "question": "Como plantar?",
                "reference": "Prepare o solo.",
                "generated": f"Resposta do {chave}.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Índice de qualidade
# ---------------------------------------------------------------------------


def test_pesos_do_indice_somam_um() -> None:
    assert round(sum(QUALITY_WEIGHTS.values()), 6) == 1.0


def test_indice_e_media_ponderada_das_metricas() -> None:
    indice = quality_index(
        {
            "rougeL": 100.0,
            "rouge1": 100.0,
            "rouge2": 100.0,
            "bleu": 100.0,
            "chrf": 100.0,
        }
    )
    assert indice == 100.0


def test_indice_ignora_metricas_ausentes() -> None:
    """Um relatório parcial não pode ser punido como se tivesse zerado."""
    assert quality_index({"rougeL": 50.0}) == 50.0


def test_indice_e_zero_sem_metricas() -> None:
    assert quality_index({}) == 0.0


# ---------------------------------------------------------------------------
# Recomendação
# ---------------------------------------------------------------------------


def test_recomenda_o_de_maior_indice_quando_a_diferenca_e_clara() -> None:
    escolha = recommend([modelo("a", rougeL=10.0), modelo("b", rougeL=40.0)])
    assert escolha["key"] == "b"
    assert escolha["runner_up"] == "a"


def test_empate_tecnico_com_custo_alto_favorece_o_barato() -> None:
    """Meio ponto de qualidade não paga um modelo 20x mais lento."""
    caro = modelo("grande", rougeL=20.4, segundos=10.0)
    barato = modelo("pequeno", rougeL=20.0, segundos=0.5)
    escolha = recommend([caro, barato])
    assert escolha["key"] == "pequeno"
    assert "custo" in escolha["reason"]


def test_vantagem_grande_vence_mesmo_sendo_caro() -> None:
    caro = modelo("grande", rougeL=45.0, segundos=10.0)
    barato = modelo("pequeno", rougeL=20.0, segundos=0.5)
    assert recommend([caro, barato])["key"] == "grande"


def _com_presets(chave: str, por_preset: dict[str, tuple[float, float]]) -> dict:
    """Modelo cujo índice e custo variam entre decodificações."""
    return modelo(
        chave,
        rougeL=0.0,
        presets={
            nome: {
                "quality": {"rougeL": indice},
                "length": {"words_mean": 40.0},
                "cost": {"seconds_per_answer": custo},
            }
            for nome, (indice, custo) in por_preset.items()
        },
    )


def test_recomendacao_usa_a_melhor_decodificacao_de_cada_modelo() -> None:
    """Regressão do caso real medido nesta coleção.

    Em ``beam_search`` o modelo pequeno vence; na melhor configuração de
    cada um, o grande vence. Arbitrar por uma decodificação fixada de
    antemão escolheria a conclusão junto com o critério.
    """
    pequeno = _com_presets(
        "pequeno", {"beam_search": (18.2, 0.2), "amostragem": (18.4, 0.2)}
    )
    grande = _com_presets("grande", {"beam_search": (17.8, 5.3), "greedy": (20.0, 5.7)})

    escolha = recommend([pequeno, grande])
    assert escolha["key"] == "grande"
    assert escolha["preset"] == "greedy"
    # E o relatório precisa expor que o resultado não foi unânime.
    assert escolha["per_preset_winners"]["beam_search"] == "pequeno"


def test_vencedores_por_decodificacao_saem_no_resultado() -> None:
    pequeno = _com_presets("pequeno", {"a": (10.0, 0.2), "b": (30.0, 0.2)})
    grande = _com_presets("grande", {"a": (20.0, 1.0), "b": (20.0, 1.0)})
    vencedores = recommend([pequeno, grande])["per_preset_winners"]
    assert vencedores == {"a": "grande", "b": "pequeno"}


def test_melhor_preset_de_um_modelo_sem_decodificacoes() -> None:
    """Sem presets, cai na qualidade agregada do topo do relatório."""
    from app.comparison import best_preset

    item = modelo("solo", rougeL=12.0)
    item["presets"] = {}
    nome, indice = best_preset(item)
    assert nome is None
    assert indice == quality_index(item["quality"])


def test_modelo_unico_e_recomendado_sem_desempate() -> None:
    escolha = recommend([modelo("a", rougeL=10.0)])
    assert escolha["key"] == "a"
    assert escolha["reason"] == "único modelo avaliado"


def test_sem_modelos_nao_ha_recomendacao() -> None:
    assert recommend([])["key"] is None


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------


def test_resumo_tem_uma_entrada_por_modelo() -> None:
    resumo = build_summary([modelo("a", rougeL=10.0), modelo("b", rougeL=40.0)])
    assert set(resumo) == {"a", "b"}
    assert resumo["b"]["indice_qualidade"] > resumo["a"]["indice_qualidade"]
    assert resumo["a"]["treino_minutos"] == 32.0


def test_markdown_traz_recomendacao_metricas_e_exemplos() -> None:
    modelos = [modelo("a", rougeL=10.0), modelo("b", rougeL=40.0)]
    texto = render_markdown(
        {
            "reference_preset": "beam_search",
            "quality_weights": QUALITY_WEIGHTS,
            "models": modelos,
            "summary": build_summary(modelos),
            "recommendation": recommend(modelos),
        }
    )
    assert "# Comparação entre os modelos ajustados" in texto
    assert "## Recomendação" in texto
    assert "rougeL" in texto
    assert "Resposta do b." in texto
    # A perda de validação precisa vir com a ressalva de não ser comparável.
    assert "não comparável" in texto


def test_markdown_nao_quebra_sem_modelos() -> None:
    texto = render_markdown(
        {
            "reference_preset": "beam_search",
            "quality_weights": QUALITY_WEIGHTS,
            "models": [],
            "summary": {},
            "recommendation": recommend([]),
        }
    )
    assert "# Comparação entre os modelos ajustados" in texto
