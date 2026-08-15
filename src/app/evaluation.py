"""Avaliação da qualidade e da originalidade do conteúdo gerado.

A avaliação combina duas famílias de métricas, porque uma sozinha é
enganosa neste tipo de tarefa:

**Qualidade** (aderência à resposta de referência da Embrapa)
    ``ROUGE-1/2/L`` e ``BLEU``/``chrF``. Medem sobreposição de n-gramas com
    a resposta original. Valores altos indicam fidelidade técnica.

**Originalidade** (o texto é novo ou é cópia do material de treino?)
    - *n-gramas inéditos*: proporção de 4-gramas gerados que não aparecem
      em nenhuma resposta do conjunto de treino. Um modelo que apenas
      memorizou tem valor próximo de zero.
    - *maior trecho copiado*: tamanho da maior sequência de palavras
      idêntica a algum trecho do treino, em número de palavras.
    - *distinct-1/2*: diversidade lexical do conjunto de respostas geradas,
      usada para detectar respostas genéricas repetidas.

As duas famílias estão em tensão: decodificação por feixes maximiza a
qualidade, amostragem aumenta a originalidade. O relatório registra as duas
para permitir a escolha consciente dos parâmetros no playground.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from app.config import (
    DEFAULT_MODEL_DIR,
    EVALUATION_REPORT_FILE,
    SPLITS_DIR,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    """Tokenização simples em palavras minúsculas, para as métricas de texto."""
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    """Lista os n-gramas de tamanho ``size`` presentes em ``tokens``."""
    if len(tokens) < size:
        return []
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


# ---------------------------------------------------------------------------
# Originalidade
# ---------------------------------------------------------------------------


@dataclass
class OriginalityIndex:
    """Índice do corpus de treino usado para medir originalidade.

    Guarda duas representações: o conjunto de n-gramas (consulta O(1) para a
    taxa de n-gramas inéditos) e o corpus normalizado como uma única string
    (busca de subcadeia para o maior trecho copiado).
    """

    ngram_size: int
    ngram_set: set[tuple[str, ...]]
    corpus_text: str

    @classmethod
    def from_texts(cls, texts: list[str], ngram_size: int = 4) -> OriginalityIndex:
        """Constrói o índice a partir das respostas do conjunto de treino."""
        ngram_set: set[tuple[str, ...]] = set()
        lines: list[str] = []
        for text in texts:
            tokens = tokenize_words(text)
            ngram_set.update(ngrams(tokens, ngram_size))
            lines.append(" ".join(tokens))
        logger.info(
            "Índice de originalidade: %d textos, %d %d-gramas distintos",
            len(texts),
            len(ngram_set),
            ngram_size,
        )
        return cls(
            ngram_size=ngram_size,
            ngram_set=ngram_set,
            corpus_text="\n".join(lines),
        )

    def novel_ngram_ratio(self, text: str) -> float:
        """Fração de n-gramas do texto que não existem no corpus de treino."""
        tokens = tokenize_words(text)
        grams = ngrams(tokens, self.ngram_size)
        if not grams:
            return 0.0
        novel = sum(1 for gram in grams if gram not in self.ngram_set)
        return novel / len(grams)

    def longest_copied_span(self, text: str, cap: int = 80) -> int:
        """Tamanho (em palavras) do maior trecho literal presente no corpus.

        Para cada posição inicial, só tenta trechos maiores que o melhor
        resultado já encontrado e para de estender na primeira falha. Isso
        mantém o número de buscas próximo do número de palavras do texto,
        em vez de quadrático.
        """
        tokens = tokenize_words(text)
        if len(tokens) < self.ngram_size:
            return 0
        longest = 0
        limit = min(len(tokens), cap)
        for start in range(len(tokens)):
            size = max(longest + 1, self.ngram_size)
            while start + size <= len(tokens) and size <= limit:
                if " ".join(tokens[start : start + size]) in self.corpus_text:
                    longest = size
                    size += 1
                else:
                    break
        return longest


def distinct_n(texts: list[str], size: int) -> float:
    """Razão entre n-gramas distintos e n-gramas totais de um conjunto."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        grams = ngrams(tokenize_words(text), size)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repetition_rate(text: str, size: int = 4) -> float:
    """Proporção de n-gramas repetidos dentro do próprio texto gerado."""
    grams = ngrams(tokenize_words(text), size)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


# ---------------------------------------------------------------------------
# Qualidade
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Calcula ROUGE, BLEU e chrF com a biblioteca ``evaluate``."""
    import evaluate as hf_evaluate

    metrics: dict[str, float] = {}

    rouge = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False,  # o stemmer disponível é para inglês
    )
    for key in ("rouge1", "rouge2", "rougeL"):
        metrics[key] = round(float(rouge_scores[key]) * 100, 2)

    sacrebleu = hf_evaluate.load("sacrebleu")
    bleu_scores = sacrebleu.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    metrics["bleu"] = round(float(bleu_scores["score"]), 2)

    chrf = hf_evaluate.load("chrf")
    chrf_scores = chrf.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )
    metrics["chrf"] = round(float(chrf_scores["score"]), 2)
    return metrics


# ---------------------------------------------------------------------------
# Avaliação completa
# ---------------------------------------------------------------------------

#: Configurações de decodificação comparadas no relatório.
#:
#: São idênticas para todos os modelos, de propósito: comparar dois modelos
#: exige que a decodificação seja a mesma variável controlada, e não mais
#: uma diferença entre eles. Os padrões por modelo em
#: ``ModelSpec.generation_overrides`` valem para o uso interativo, onde o
#: tempo de resposta importa mais que a comparabilidade.
DECODING_PRESETS: dict[str, dict[str, object]] = {
    "beam_search": {"do_sample": False, "num_beams": 4, "length_penalty": 1.3},
    "beam_search_curto": {
        "do_sample": False,
        "num_beams": 4,
        "length_penalty": 1.0,
    },
    "greedy": {"do_sample": False, "num_beams": 1},
    "amostragem_criativa": {
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
    },
}


def evaluate_model(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    splits_dir: Path = SPLITS_DIR,
    limit: int | None = 200,
    batch_size: int = 8,
    presets: dict[str, dict[str, object]] | None = None,
    base_config: GenerationConfig | None = None,
    run_name: str = "avaliacao",
    logged_examples: int = 25,
    report_path: Path | None = EVALUATION_REPORT_FILE,
    spec=None,
) -> dict[str, object]:
    """Avalia o modelo no conjunto de teste sob várias decodificações.

    ``limit`` restringe o número de exemplos avaliados (a geração é a etapa
    mais lenta); ``None`` avalia o conjunto de teste completo. A seleção é
    sempre os ``limit`` primeiros exemplos da partição de teste — é o que
    garante que dois modelos avaliados com o mesmo limite respondam
    exatamente às mesmas perguntas.

    Cada decodificação vira uma execução aninhada no MLflow, com seus
    próprios parâmetros e métricas — é isso que permite compará-las na
    interface. As métricas também são replicadas na execução-mãe com o
    prefixo do nome da decodificação, para uma visão consolidada.
    """
    import time

    from app import tracking
    from app.dataset import load_splits
    from app.generate import QAGenerator
    from app.models import generation_config_for, resolve_spec

    spec = resolve_spec(spec) if spec is not None else None
    presets = presets or DECODING_PRESETS
    if base_config is None:
        base_config = (
            generation_config_for(spec) if spec is not None else GenerationConfig()
        )

    splits = load_splits(splits_dir)
    test = splits["test"]
    if limit:
        test = test.select(range(min(limit, len(test))))

    questions = list(test["question"])
    topics = list(test["topic"])
    references = list(test["answer"])
    train_answers = list(splits["train"]["answer"])

    index = OriginalityIndex.from_texts(train_answers, ngram_size=4)
    generator = QAGenerator(model_dir, spec=spec)
    spec = spec or generator.spec

    results: dict[str, object] = {
        "model": generator.source,
        "model_key": spec.key,
        "model_label": spec.label,
        "base_model": spec.hf_name,
        "family": spec.family,
        "parameters": spec.parameters,
        "fine_tuned": generator.is_fine_tuned,
        "test_examples": len(questions),
        "presets": {},
    }

    with tracking.start_run(
        run_name,
        tags={
            "modelo": spec.key,
            "modelo_base": spec.hf_name,
            "familia": spec.family,
            "pesos": str(model_dir),
        },
    ):
        tracking.log_params(
            {
                "modelo": spec.key,
                "pesos": generator.source,
                "modelo_base": spec.hf_name,
                "familia": spec.family,
                "exemplos_teste": len(questions),
                "decodificacoes": list(presets),
                "ngrama_originalidade": index.ngram_size,
            }
        )
        tracking.log_dataset(splits_dir / "test.jsonl", "qa-embrapa-test", "evaluation")

        for name, overrides in presets.items():
            logger.info("Gerando respostas com a decodificação '%s' ...", name)
            config = replace(base_config, num_return_sequences=1, **overrides)

            with tracking.start_run(f"{run_name}-{name}", nested=True):
                tracking.log_params(config, prefix="geracao.")
                tracking.set_tags({"decodificacao": name})

                started = time.perf_counter()
                generated = generator.answer_many(
                    list(zip(questions, topics)),
                    config=config,
                    batch_size=batch_size,
                )
                elapsed = time.perf_counter() - started
                predictions = [
                    variants[0] if variants else "" for variants in generated
                ]

                quality = compute_quality_metrics(predictions, references)
                novel = [index.novel_ngram_ratio(text) for text in predictions]
                copied = [index.longest_copied_span(text) for text in predictions]
                lengths = [len(tokenize_words(text)) for text in predictions]
                repetitions = [repetition_rate(text) for text in predictions]

                preset_result = {
                    "generation": {
                        key: (str(value) if isinstance(value, Path) else value)
                        for key, value in overrides.items()
                    },
                    "quality": quality,
                    "originality": {
                        "novel_4gram_ratio_mean": round(sum(novel) / len(novel), 3),
                        "longest_copied_span_mean": round(sum(copied) / len(copied), 1),
                        "longest_copied_span_max": max(copied),
                        "distinct_1": round(distinct_n(predictions, 1), 3),
                        "distinct_2": round(distinct_n(predictions, 2), 3),
                        "self_repetition_4gram_mean": round(
                            sum(repetitions) / len(repetitions), 3
                        ),
                    },
                    "length": {
                        "words_mean": round(sum(lengths) / len(lengths), 1),
                        "words_min": min(lengths),
                        "words_max": max(lengths),
                        "empty_answers": sum(
                            1 for text in predictions if not text.strip()
                        ),
                    },
                    # O custo de inferência é parte da comparação entre
                    # modelos: um ganho de qualidade que custe uma ordem de
                    # grandeza em tempo de resposta não é indiferente.
                    "cost": {
                        "seconds_total": round(elapsed, 1),
                        "seconds_per_answer": round(elapsed / len(predictions), 2),
                    },
                    "examples": [
                        {
                            "topic": topics[i],
                            "question": questions[i],
                            "reference": references[i],
                            "generated": predictions[i],
                            "novel_4gram_ratio": round(novel[i], 3),
                        }
                        for i in range(min(5, len(predictions)))
                    ],
                }
                results["presets"][name] = preset_result

                tracking.log_metrics(quality)
                tracking.log_metrics(preset_result["originality"])
                tracking.log_metrics(preset_result["length"])
                tracking.log_metrics(preset_result["cost"])
                tracking.log_table(
                    [
                        {
                            "tema": topics[i],
                            "pergunta": questions[i],
                            "referencia": references[i],
                            "gerada": predictions[i],
                            "palavras_geradas": lengths[i],
                            "4gramas_ineditos": round(novel[i], 3),
                            "maior_trecho_copiado": copied[i],
                        }
                        for i in range(min(logged_examples, len(predictions)))
                    ],
                    "exemplos_gerados.json",
                )

            # Consolida na execução-mãe para comparação em uma única tela.
            tracking.log_metrics(quality, prefix=f"{name}.")
            tracking.log_metrics(preset_result["originality"], prefix=f"{name}.")
            tracking.log_metrics(preset_result["length"], prefix=f"{name}.")
            tracking.log_metrics(preset_result["cost"], prefix=f"{name}.")
            logger.info("'%s': %s", name, quality)

        results["mlflow_run_id"] = tracking.active_run_id()
        if report_path is not None:
            write_report(results, report_path)
            tracking.log_artifact(report_path, artifact_path="relatorios")

    return results


def write_report(
    report: dict[str, object], path: Path = EVALUATION_REPORT_FILE
) -> Path:
    """Grava o relatório de avaliação em JSON."""
    from app.dataset import write_json

    write_json(path, report)
    logger.info("Relatório de avaliação gravado em %s", path)
    return path
