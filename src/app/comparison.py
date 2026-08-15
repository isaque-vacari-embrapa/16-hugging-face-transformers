"""Comparação entre os modelos treinados e recomendação de um deles.

Lê os relatórios de treino e avaliação já gravados por modelo, alinha as
métricas em uma tabela e aplica um critério explícito de escolha. A leitura
a partir dos relatórios — e não de uma execução conjunta — é deliberada: os
dois modelos não cabem juntos na GPU, então são treinados e avaliados em
momentos distintos, e a comparação precisa funcionar sobre o que ficou
gravado.

**O que é comparável e o que não é.** As métricas de qualidade
(``ROUGE``/``BLEU``/``chrF``) e de originalidade são calculadas sobre as
mesmas perguntas do mesmo conjunto de teste, com as mesmas referências:
são diretamente comparáveis. A perda de validação **não é** — cada modelo
a calcula sobre o próprio vocabulário (32 mil contra 262 mil posições) e
sobre segmentações diferentes do mesmo texto, de modo que os valores não
estão na mesma escala. Por isso ela aparece no relatório apenas dentro da
coluna de cada modelo, nunca como critério de desempate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import (
    COMPARISON_MARKDOWN_FILE,
    COMPARISON_REPORT_FILE,
    ensure_dirs,
)
from app.models import AVAILABLE_MODELS, resolve_spec

logger = logging.getLogger(__name__)

#: Decodificação usada como referência na comparação. É a configuração que
#: o playground oferece por padrão e a que maximiza fidelidade ao material
#: técnico, que é o objetivo do sistema.
REFERENCE_PRESET = "beam_search"

#: Métricas de qualidade, na ordem em que aparecem nas tabelas.
QUALITY_METRICS = ("rougeL", "rouge1", "rouge2", "bleu", "chrf")

#: Peso de cada métrica de qualidade no índice agregado. ROUGE-L pesa mais
#: por medir a maior subsequência comum, que captura ordem e cobertura;
#: BLEU pesa menos porque penaliza duramente a paráfrase legítima, comum
#: quando o modelo reescreve a recomendação técnica com outras palavras.
QUALITY_WEIGHTS = {
    "rougeL": 0.35,
    "rouge1": 0.25,
    "chrf": 0.25,
    "rouge2": 0.10,
    "bleu": 0.05,
}


def load_report(path: Path) -> dict[str, Any] | None:
    """Carrega um relatório JSON, ou ``None`` se ele ainda não existe."""
    if not path.exists():
        logger.info("Relatório ausente: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect(model_keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Reúne, por modelo, o que os relatórios de treino e avaliação dizem.

    Modelos ainda não avaliados são silenciosamente omitidos: a comparação
    deve funcionar mesmo quando só parte do catálogo foi executada.
    """
    chaves = model_keys or list(AVAILABLE_MODELS)
    reunidos: list[dict[str, Any]] = []

    for chave in chaves:
        spec = resolve_spec(chave)
        avaliacao = load_report(spec.evaluation_report_file)
        if avaliacao is None:
            continue
        treino = load_report(spec.training_report_file) or {}
        presets = avaliacao.get("presets", {})
        referencia = presets.get(REFERENCE_PRESET) or next(iter(presets.values()), {})

        reunidos.append(
            {
                "key": spec.key,
                "label": spec.label,
                "checkpoint": spec.hf_name,
                "family": spec.family,
                "parameters": spec.parameters,
                "strategy": treino.get(
                    "strategy", "lora-4bit" if spec.use_lora else "ajuste-completo"
                ),
                "test_examples": avaliacao.get("test_examples"),
                "training": {
                    "elapsed_seconds": treino.get("elapsed_seconds"),
                    "epochs": (treino.get("hyperparameters") or {}).get(
                        "num_train_epochs"
                    ),
                    "trainable_parameters": treino.get("trainable_parameters"),
                    "total_parameters": treino.get("total_parameters"),
                    "validation_loss": (treino.get("validation_metrics") or {}).get(
                        "eval_loss"
                    ),
                },
                "reference_preset": (
                    REFERENCE_PRESET
                    if REFERENCE_PRESET in presets
                    else next(iter(presets), None)
                ),
                "quality": referencia.get("quality", {}),
                "originality": referencia.get("originality", {}),
                "length": referencia.get("length", {}),
                "cost": referencia.get("cost", {}),
                "presets": {
                    nome: {
                        "quality": dados.get("quality", {}),
                        "originality": dados.get("originality", {}),
                        "length": dados.get("length", {}),
                        "cost": dados.get("cost", {}),
                    }
                    for nome, dados in presets.items()
                },
                "examples": referencia.get("examples", []),
            }
        )
    return reunidos


def quality_index(quality: dict[str, Any]) -> float:
    """Média ponderada das métricas de qualidade, em uma escala de 0 a 100.

    Serve para ordenar os modelos sem depender de uma única métrica, que
    isoladamente é fácil de enganar: um modelo que copia trechos inteiros
    do treino sobe em ``BLEU`` sem ser mais útil.
    """
    total = 0.0
    peso_usado = 0.0
    for metrica, peso in QUALITY_WEIGHTS.items():
        valor = quality.get(metrica)
        if isinstance(valor, (int, float)):
            total += float(valor) * peso
            peso_usado += peso
    return round(total / peso_usado, 2) if peso_usado else 0.0


def build_summary(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Tabela resumida por modelo, com o índice de qualidade agregado."""
    resumo: dict[str, Any] = {}
    for item in models:
        resumo[item["key"]] = {
            "modelo": item["label"],
            "parametros": item["parameters"],
            "estrategia": item["strategy"],
            "indice_qualidade": quality_index(item["quality"]),
            **{
                metrica: item["quality"].get(metrica)
                for metrica in QUALITY_METRICS
                if metrica in item["quality"]
            },
            "4gramas_ineditos": item["originality"].get("novel_4gram_ratio_mean"),
            "maior_trecho_copiado": item["originality"].get("longest_copied_span_mean"),
            "palavras_media": item["length"].get("words_mean"),
            "segundos_por_resposta": item["cost"].get("seconds_per_answer"),
            "treino_minutos": (
                round(item["training"]["elapsed_seconds"] / 60, 1)
                if item["training"].get("elapsed_seconds")
                else None
            ),
        }
    return resumo


def best_preset(model: dict[str, Any]) -> tuple[str | None, float]:
    """Melhor decodificação do modelo e o índice de qualidade que ela atinge.

    Comparar cada modelo na *sua* melhor configuração é o que responde à
    pergunta prática — qual dos dois entrega mais, dado que o operador vai
    escolher a decodificação que funciona melhor para o modelo que
    implantar. Fixar uma decodificação única como árbitro produziria uma
    conclusão que muda conforme a decodificação escolhida a priori.
    """
    presets = model.get("presets") or {}
    if not presets:
        return None, quality_index(model.get("quality", {}))
    nome, dados = max(
        presets.items(), key=lambda item: quality_index(item[1].get("quality", {}))
    )
    return nome, quality_index(dados.get("quality", {}))


def preset_wins(models: list[dict[str, Any]]) -> dict[str, str]:
    """Vencedor de cada decodificação, quando todos os modelos a têm.

    É a evidência de que a comparação não depende de uma única escolha de
    decodificação — ou, quando depende, é o que torna essa dependência
    visível em vez de escondida atrás de um número só.
    """
    if len(models) < 2:
        return {}
    comuns = set(models[0].get("presets") or {})
    for item in models[1:]:
        comuns &= set(item.get("presets") or {})

    vencedores: dict[str, str] = {}
    for nome in sorted(comuns):
        campeao = max(
            models, key=lambda m: quality_index(m["presets"][nome].get("quality", {}))
        )
        vencedores[nome] = campeao["key"]
    return vencedores


def recommend(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Escolhe o modelo recomendado e explica o critério aplicado.

    O critério é o índice de qualidade que cada modelo atinge na **sua
    melhor decodificação**, e não em uma decodificação fixada de antemão:
    medimos que o ranking se inverte entre decodificações, de modo que
    eleger uma delas como árbitro seria escolher a conclusão junto com o
    critério.

    O custo de inferência entra como desempate. Uma diferença de qualidade
    menor que um ponto está dentro do ruído de um conjunto de teste de
    algumas centenas de exemplos e não justifica um modelo várias vezes
    mais caro por resposta.
    """
    if not models:
        return {"key": None, "label": None, "reason": "nenhum modelo avaliado"}

    def custo(model: dict[str, Any], preset: str | None) -> float:
        presets = model.get("presets") or {}
        origem = presets.get(preset, {}) if preset else model
        return (origem.get("cost") or {}).get("seconds_per_answer") or 0.0

    pontuados = []
    for item in models:
        nome, indice = best_preset(item)
        pontuados.append((item, nome, indice, custo(item, nome)))
    pontuados.sort(key=lambda linha: linha[2], reverse=True)

    melhor, preset_melhor, indice_melhor, custo_melhor = pontuados[0]
    base = {
        "criterion": (
            "índice de qualidade agregado na melhor decodificação de cada "
            "modelo, com o custo de inferência como desempate"
        ),
        "per_preset_winners": preset_wins(models),
    }

    if len(pontuados) == 1:
        return {
            **base,
            "key": melhor["key"],
            "label": melhor["label"],
            "quality_index": indice_melhor,
            "preset": preset_melhor,
            "reason": "único modelo avaliado",
        }

    segundo, preset_segundo, indice_segundo, custo_segundo = pontuados[1]
    diferenca = round(indice_melhor - indice_segundo, 2)
    razao_custo = (custo_melhor / custo_segundo) if custo_segundo else 1.0

    # Empate técnico com custo desproporcional: fica o mais barato.
    if diferenca < 1.0 and razao_custo > 3.0:
        return {
            **base,
            "key": segundo["key"],
            "label": segundo["label"],
            "quality_index": indice_segundo,
            "preset": preset_segundo,
            "runner_up": melhor["key"],
            "reason": (
                f"empate técnico em qualidade ({diferenca:+.2f} ponto a favor "
                f"de {melhor['label']}, dentro do ruído do conjunto de teste) "
                f"com custo de inferência {razao_custo:.1f}× menor"
            ),
        }

    return {
        **base,
        "key": melhor["key"],
        "label": melhor["label"],
        "quality_index": indice_melhor,
        "preset": preset_melhor,
        "runner_up": segundo["key"],
        "cost_ratio": round(razao_custo, 1),
        "reason": (
            f"maior índice de qualidade agregado na melhor configuração de "
            f"cada um ({indice_melhor:.2f} em `{preset_melhor}` contra "
            f"{indice_segundo:.2f} de {segundo['label']} em "
            f"`{preset_segundo}`, {diferenca:+.2f}), ao custo de "
            f"{razao_custo:.0f}× mais tempo por resposta"
        ),
    }


def compare_models(
    model_keys: list[str] | None = None,
    report_path: Path | None = COMPARISON_REPORT_FILE,
    markdown_path: Path | None = COMPARISON_MARKDOWN_FILE,
    run_name: str = "comparacao",
) -> dict[str, Any]:
    """Compara os modelos avaliados e grava os relatórios da comparação."""
    from app import tracking

    modelos = collect(model_keys)
    resultado: dict[str, Any] = {
        "reference_preset": REFERENCE_PRESET,
        "quality_weights": QUALITY_WEIGHTS,
        "models": modelos,
        "summary": build_summary(modelos),
        "recommendation": recommend(modelos),
    }
    if not modelos:
        return resultado

    with tracking.start_run(run_name, tags={"etapa": "comparacao"}):
        tracking.log_params(
            {
                "modelos": [item["key"] for item in modelos],
                "decodificacao_referencia": REFERENCE_PRESET,
                "recomendado": resultado["recommendation"]["key"],
            }
        )
        for item in modelos:
            tracking.log_metrics(
                {
                    "indice_qualidade": quality_index(item["quality"]),
                    **{
                        chave: valor
                        for chave, valor in item["quality"].items()
                        if isinstance(valor, (int, float))
                    },
                },
                prefix=f"{item['key']}.",
            )
        if report_path is not None:
            ensure_dirs(report_path.parent)
            report_path.write_text(
                json.dumps(resultado, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tracking.log_artifact(report_path, artifact_path="relatorios")
        if markdown_path is not None:
            ensure_dirs(markdown_path.parent)
            markdown_path.write_text(render_markdown(resultado), encoding="utf-8")
            tracking.log_artifact(markdown_path, artifact_path="relatorios")
        resultado["mlflow_run_id"] = tracking.active_run_id()

    logger.info("Comparação gravada em %s e %s", report_path, markdown_path)
    return resultado


# ---------------------------------------------------------------------------
# Relatório em Markdown
# ---------------------------------------------------------------------------


def _cell(value: Any, casas: int = 2, sufixo: str = "") -> str:
    """Formata um valor numérico para a tabela, ou ``—`` quando ausente."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{casas}f}{sufixo}"
    return f"{value}{sufixo}"


def _table(cabecalho: list[str], linhas: list[list[str]]) -> list[str]:
    """Monta uma tabela em Markdown."""
    return [
        "| " + " | ".join(cabecalho) + " |",
        "| " + " | ".join("---" for _ in cabecalho) + " |",
        *["| " + " | ".join(linha) + " |" for linha in linhas],
    ]


def render_markdown(resultado: dict[str, Any]) -> str:
    """Gera o relatório comparativo legível a partir dos dados coletados."""
    modelos: list[dict[str, Any]] = resultado["models"]
    recomendacao = resultado["recommendation"]
    nomes = [item["label"] for item in modelos]

    linhas: list[str] = [
        "# Comparação entre os modelos ajustados",
        "",
        "Relatório gerado por `poetry run qa-embrapa compare`. Os números "
        "vêm dos relatórios de avaliação de cada modelo, calculados sobre "
        "**as mesmas perguntas do mesmo conjunto de teste**.",
        "",
    ]

    if modelos:
        exemplos = modelos[0].get("test_examples")
        linhas += [
            f"- Conjunto de teste: **{exemplos} perguntas** não vistas no treino.",
            f"- Modelos comparados: {', '.join(nomes)}.",
            "",
            "O relatório traz **duas leituras**, que respondem a perguntas "
            "diferentes e podem apontar para modelos diferentes:",
            "",
            f"1. **Decodificação idêntica** (`{resultado['reference_preset']}` "
            "nos dois) — isola a diferença entre os modelos, mantendo a "
            "decodificação como variável controlada. É a base das seções "
            "*Qualidade*, *Originalidade* e *Custo*.",
            "2. **Melhor configuração de cada um** — é o que um operador "
            "obteria na prática, já que a decodificação é escolhida junto "
            "com o modelo. É a base da **recomendação**.",
            "",
        ]

    # -- recomendação -------------------------------------------------------
    linhas += [
        "## Recomendação",
        "",
        f"**{recomendacao['label']}** — {recomendacao['reason']}.",
        "",
        f"Critério: {recomendacao.get('criterion', 'índice de qualidade')}.",
        "",
    ]

    vencedores = recomendacao.get("per_preset_winners") or {}
    if len(set(vencedores.values())) > 1:
        detalhe = ", ".join(f"`{nome}` → {chave}" for nome, chave in vencedores.items())
        linhas += [
            "> **A comparação depende da decodificação.** Os vencedores por "
            f"decodificação não são unânimes ({detalhe}). É por isso que o "
            "critério compara cada modelo na sua melhor configuração, e não "
            "em uma decodificação eleita de antemão — que produziria a "
            "conclusão que se escolhesse.",
            "",
        ]

    # -- identificação ------------------------------------------------------
    linhas += ["## Os modelos", ""]
    linhas += _table(
        ["", *nomes],
        [
            ["Checkpoint", *[f"`{i['checkpoint']}`" for i in modelos]],
            ["Arquitetura", *[i["family"] for i in modelos]],
            ["Parâmetros", *[i["parameters"] for i in modelos]],
            ["Estratégia de ajuste", *[i["strategy"] for i in modelos]],
            [
                "Parâmetros treinados",
                *[
                    _cell(
                        (
                            f"{i['training']['trainable_parameters']:,}".replace(
                                ",", "."
                            )
                            if i["training"].get("trainable_parameters")
                            else None
                        )
                    )
                    for i in modelos
                ],
            ],
            [
                "Épocas",
                *[_cell(i["training"].get("epochs"), casas=0) for i in modelos],
            ],
            [
                "Tempo de treino",
                *[
                    _cell(
                        (
                            i["training"]["elapsed_seconds"] / 60
                            if i["training"].get("elapsed_seconds")
                            else None
                        ),
                        casas=0,
                        sufixo=" min",
                    )
                    for i in modelos
                ],
            ],
        ],
    )
    linhas += [""]

    # -- qualidade ----------------------------------------------------------
    referencia = resultado["reference_preset"]
    linhas += [
        f"## Qualidade — decodificação idêntica (`{referencia}`)",
        "",
        "Aderência à resposta original da Embrapa. Todas as métricas vão de "
        "0 a 100 e **quanto maior, melhor**.",
        "",
        "> Estes números são da decodificação de referência, a mesma nos "
        "dois modelos. Não são a base da recomendação — para isso, ver "
        "*Qualidade por decodificação*, mais abaixo.",
        "",
    ]
    linhas += _table(
        ["Métrica", *nomes],
        [
            [
                "**Índice agregado**",
                *[f"**{quality_index(i['quality']):.2f}**" for i in modelos],
            ],
            *[
                [metrica, *[_cell(i["quality"].get(metrica)) for i in modelos]]
                for metrica in QUALITY_METRICS
            ],
        ],
    )
    pesos = ", ".join(f"{k} {v:.0%}" for k, v in QUALITY_WEIGHTS.items())
    linhas += ["", f"Pesos do índice agregado: {pesos}.", ""]

    # -- originalidade ------------------------------------------------------
    linhas += [
        f"## Originalidade — decodificação idêntica (`{referencia}`)",
        "",
        "O conteúdo é novo ou é cópia do material de treino?",
        "",
    ]
    linhas += _table(
        ["Métrica", *nomes],
        [
            [
                "4-gramas inéditos (↑ mais original)",
                *[
                    _cell(
                        (
                            i["originality"]["novel_4gram_ratio_mean"] * 100
                            if i["originality"].get("novel_4gram_ratio_mean")
                            is not None
                            else None
                        ),
                        casas=1,
                        sufixo="%",
                    )
                    for i in modelos
                ],
            ],
            [
                "Maior trecho copiado (↓ melhor)",
                *[
                    _cell(
                        i["originality"].get("longest_copied_span_mean"),
                        casas=1,
                        sufixo=" palavras",
                    )
                    for i in modelos
                ],
            ],
            [
                "distinct-2 (↑ mais diverso)",
                *[_cell(i["originality"].get("distinct_2"), casas=3) for i in modelos],
            ],
            [
                "Repetição interna (↓ melhor)",
                *[
                    _cell(i["originality"].get("self_repetition_4gram_mean"), casas=3)
                    for i in modelos
                ],
            ],
        ],
    )
    linhas += [""]

    # -- custo --------------------------------------------------------------
    linhas += [f"## Custo e formato — decodificação idêntica (`{referencia}`)", ""]
    linhas += _table(
        ["Métrica", *nomes],
        [
            [
                "Segundos por resposta",
                *[
                    _cell(i["cost"].get("seconds_per_answer"), casas=2, sufixo=" s")
                    for i in modelos
                ],
            ],
            [
                "Palavras por resposta",
                *[_cell(i["length"].get("words_mean"), casas=1) for i in modelos],
            ],
            [
                "Respostas vazias",
                *[_cell(i["length"].get("empty_answers"), casas=0) for i in modelos],
            ],
            [
                "Perda de validação *(não comparável entre modelos)*",
                *[
                    _cell(i["training"].get("validation_loss"), casas=3)
                    for i in modelos
                ],
            ],
        ],
    )
    linhas += [
        "",
        "> A perda de validação aparece apenas como registro interno de cada "
        "modelo. Ela **não** é comparável entre os dois: cada um a calcula "
        "sobre o próprio vocabulário e a própria segmentação do texto.",
        "",
    ]

    # -- por decodificação --------------------------------------------------
    todas = sorted({nome for i in modelos for nome in i["presets"]})
    if todas:
        linhas += [
            "## Qualidade por decodificação",
            "",
            "Índice agregado em cada estratégia, com a mesma decodificação "
            "aplicada aos dois modelos. A última coluna mostra onde cada um "
            "leva vantagem — é esta tabela que revela que o resultado não é "
            "uniforme.",
            "",
        ]
        corpo = []
        for nome in todas:
            indices = [
                (
                    quality_index(i["presets"][nome]["quality"])
                    if nome in i["presets"]
                    else None
                )
                for i in modelos
            ]
            validos = [valor for valor in indices if valor is not None]
            if len(validos) == len(modelos) and len(modelos) > 1:
                topo = max(validos)
                campeao = modelos[indices.index(topo)]["label"]
                margem = topo - sorted(validos)[-2]
                vencedor = f"{campeao} (+{margem:.2f})"
            else:
                vencedor = "—"
            corpo.append([f"`{nome}`", *[_cell(valor) for valor in indices], vencedor])
        linhas += _table(["Decodificação", *nomes, "Melhor"], corpo)
        linhas += [""]

        melhores = [best_preset(i) for i in modelos]
        linhas += [
            "Melhor configuração de cada modelo — a base da recomendação:",
            "",
        ]
        linhas += _table(
            ["Modelo", "Decodificação", "Índice", "Palavras", "Segundos"],
            [
                [
                    item["label"],
                    f"`{nome}`" if nome else "—",
                    f"**{indice:.2f}**",
                    _cell(
                        (item["presets"].get(nome, {}).get("length") or {}).get(
                            "words_mean"
                        ),
                        casas=1,
                    ),
                    _cell(
                        (item["presets"].get(nome, {}).get("cost") or {}).get(
                            "seconds_per_answer"
                        ),
                        casas=2,
                        sufixo=" s",
                    ),
                ]
                for item, (nome, indice) in zip(modelos, melhores)
            ],
        )
        linhas += [""]

    # -- exemplos -----------------------------------------------------------
    linhas += render_examples(modelos)
    return "\n".join(linhas) + "\n"


def render_examples(modelos: list[dict[str, Any]], quantos: int = 3) -> list[str]:
    """Trechos lado a lado: a mesma pergunta respondida por cada modelo."""
    if not modelos:
        return []
    base = modelos[0].get("examples") or []
    if not base:
        return []

    linhas = [
        "## Respostas lado a lado",
        "",
        "As mesmas perguntas do conjunto de teste, na decodificação de " "referência.",
        "",
    ]
    for posicao, exemplo in enumerate(base[:quantos]):
        linhas += [
            f"### {posicao + 1}. {exemplo.get('topic', '—')} — "
            f"{exemplo.get('question', '')}",
            "",
            "**Resposta original da Embrapa**",
            "",
            f"> {_quote(exemplo.get('reference', ''))}",
            "",
        ]
        for item in modelos:
            exemplos = item.get("examples") or []
            gerada = (
                exemplos[posicao].get("generated", "")
                if posicao < len(exemplos)
                else ""
            )
            linhas += [
                f"**{item['label']}**",
                "",
                f"> {_quote(gerada) or '(vazio)'}",
                "",
            ]
    return linhas


def _quote(text: str, limite: int = 700) -> str:
    """Prepara um texto para caber dentro de uma citação em Markdown."""
    text = " ".join(str(text).split())
    if len(text) > limite:
        text = text[:limite].rsplit(" ", 1)[0] + " […]"
    return text
