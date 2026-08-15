"""Extração dos pares pergunta/resposta a partir do Markdown da Docling.

A Coleção 500 Perguntas 500 Respostas da Embrapa tem um formato regular: cada item é
numerado e o enunciado aparece destacado, o que a biblioteca Docling converte em um título de seção.

    ## 147 O que é dormência de sementes?

    Dormência é uma condição fisiológica que impede a germinação ...

Na prática, porém, o texto original é diagramado em colunas e caixas, e
isso produz três defeitos recorrentes que este módulo corrige:

1. **Enunciado partido em dois títulos** — ``## 193 Sementes que
   apresentam dormência também apresentam`` seguido de
   ``## maior longevidade?``.
2. **Enunciado invertido** — o fragmento com o número vem depois na ordem
   de leitura: ``## 194 semente e a dormência?`` seguido de
   ``## Há alguma correlação entre caracteres morfológicos da``.
3. **Enunciado rebaixado a parágrafo** — a numeração não foi reconhecida e
   a pergunta acabou no corpo do texto, junto da resposta anterior.

O resultado é gravado em JSON Lines (``data/processed``), formato escolhido
por ser lido diretamente pela biblioteca ``datasets`` do Hugging Face.
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import INTERIM_DIR, QA_DATASET_FILE, ensure_dirs
from app.topics import topic_label, topic_slug_from_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limiares de aceitação
# ---------------------------------------------------------------------------

MIN_QUESTION_CHARS = 15
MIN_QUESTION_WORDS = 3
#: Nenhum enunciado da coleção passa de ~45 palavras; acima disso o texto
#: certamente arrastou título de capítulo ou lista de autores.
MAX_QUESTION_WORDS = 45
MIN_ANSWER_CHARS = 40
MIN_ANSWER_WORDS = 6

#: Faixa de tamanho aceita para uma pergunta "escondida" no corpo do texto.
INLINE_QUESTION_MIN_WORDS = 6
INLINE_QUESTION_MAX_WORDS = 45

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.*)$")
_BULLET_RE = re.compile(r"^\s{0,4}(?:[-*+]|•|\d{1,2}[.)])\s+(?P<text>.*)$")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_LEADING_NUMBER_RE = re.compile(r"^(?P<number>\d{1,4})\s*[.)\]-]?\s+(?P<rest>.*)$")
_ONLY_NUMBER_RE = re.compile(r"^(?P<number>\d{1,4})\s*[.)\]-]?$")
#: Alguns volumes da coleção (Banana, por exemplo) imprimem a numeração à
#: direita do enunciado: ``## Onde se originou a bananeira? 2``.
_TRAILING_NUMBER_RE = re.compile(r"^(?P<rest>.*\?)\s+(?P<number>\d{1,4})$")
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|`)")
_REFERENCE_RE = re.compile(
    r"(?:\bDOI:\s*\S)|(?:^[A-ZÀ-Ü][A-ZÀ-Ü\-']{1,}(?:,\s*[A-ZÀ-Ü]\.){1,})"
)
#: Assinatura de créditos de capítulo: "Nome Sobrenome - Embrapa Unidade".
_AFFILIATION_RE = re.compile(r"\s[-–]\s*(?:Embrapa|Universidade|Instituto|Fundação)\b")
#: Fragmento terminado em palavra funcional (ou hífen) indica frase
#: interrompida no fim da linha, ou seja, enunciado que continua adiante.
_CONTINUATION_END_RE = re.compile(
    r"(?:\b(?:de|do|da|dos|das|em|no|na|nos|nas|para|por|com|a|o|as|os|ao|à|às|"
    r"aos|e|ou|que|se|sobre|entre|sem|até|como|um|uma|seu|sua|seus|suas|qual|"
    r"quais|quando|onde|mais|menos|muito|todo|toda|não|ser|é)$)|[-–]$",
    re.IGNORECASE,
)
#: Respostas que apenas remetem a uma tabela/figura não convertida.
_ONLY_REFERS_TO_EXHIBIT_RE = re.compile(r"\b(?:Tabela|Figura|Quadro|Anexo)s?\s+\d")
_TERMINAL_PUNCT = ".!?:;"
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;:])\s+")


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    """Um par pergunta/resposta pronto para o conjunto de dados."""

    id: str
    topic_slug: str
    topic: str
    number: int | None
    question: str
    answer: str
    source_pdf: str
    source_markdown: str
    number_inferred: bool = False
    repaired: bool = False
    recovered_from_body: bool = False

    def to_json(self) -> str:
        """Serializa o registro em uma linha JSON (UTF-8 legível)."""
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ExtractionStats:
    """Contadores de diagnóstico de uma extração."""

    markdown_file: str = ""
    topic_slug: str = ""
    topic: str = ""
    headings_seen: int = 0
    candidates: int = 0
    accepted: int = 0
    rejected_no_question_mark: int = 0
    rejected_short_question: int = 0
    rejected_long_question: int = 0
    rejected_short_answer: int = 0
    rejected_exhibit_only: int = 0
    repaired_questions: int = 0
    recovered_from_body: int = 0
    inferred_numbers: int = 0
    number_range: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serializa as estatísticas para o relatório em JSON."""
        return asdict(self)


@dataclass
class _Block:
    """Bloco elementar do Markdown: título ou corpo de texto."""

    kind: str  # "heading" | "body"
    text: str
    is_bullet: bool = False


@dataclass
class _Candidate:
    """Agrupamento intermediário: fragmentos de enunciado + blocos da resposta."""

    fragments: list[str] = field(default_factory=list)
    number: int | None = None
    answer_blocks: list[_Block] = field(default_factory=list)
    repaired: bool = False

    @property
    def has_question_mark(self) -> bool:
        """Indica se algum fragmento encerra com sinal de interrogação."""
        return any(frag.rstrip().endswith("?") for frag in self.fragments)


# ---------------------------------------------------------------------------
# Limpeza de texto
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normaliza um trecho de Markdown para texto simples.

    Remove comentários HTML, marcações de ênfase, hifens de separação
    silábica (``\\u00ad``) e espaços redundantes — a justificação de texto
    do PDF original gera sequências longas de espaços dentro das frases.
    """
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    # Hífen opcional (soft hyphen) de translineação, às vezes seguido de
    # espaço: "prin­ cipais" deve voltar a ser "principais".
    text = re.sub("­\\s*", "", text)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = _EMPHASIS_RE.sub("", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def join_fragments(fragments: list[str]) -> str:
    """Concatena fragmentos de enunciado respeitando hifenização.

    Um fragmento terminado em hífen indica palavra partida no fim da linha
    (``... para a agri-`` + ``cultura?`` -> ``... para a agricultura?``).
    """
    out = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if not out:
            out = fragment
        elif out.endswith("-"):
            out = out[:-1] + fragment
        else:
            out = f"{out} {fragment}"
    return re.sub(r"\s+", " ", out).strip()


def _starts_lowercase(text: str) -> bool:
    """Indica que o texto começa em minúscula, sinal de enunciado truncado."""
    for char in text:
        if char.isalpha():
            return char.islower()
    return False


def _looks_like_reference(text: str) -> bool:
    """Heurística para descartar entradas de bibliografia."""
    return bool(_REFERENCE_RE.search(text))


def _looks_like_author_line(text: str) -> bool:
    """Heurística para os créditos de autoria que abrem cada capítulo.

    A Docling reconhece esses blocos como títulos, e eles podem acabar
    grudados no primeiro enunciado do capítulo. São identificados pela
    afiliação ("- Embrapa ...") ou por serem sequências longas de nomes
    próprios sem nenhuma palavra funcional.
    """
    if text.rstrip().endswith("?"):
        return False
    if _AFFILIATION_RE.search(text):
        return True
    words = text.split()
    if len(words) < 6:
        return False
    capitalized = sum(1 for word in words if word[:1].isupper())
    return capitalized / len(words) >= 0.7


def _ends_sentence(text: str) -> bool:
    """Indica se o texto termina com pontuação de fim de frase."""
    return bool(text) and text.rstrip()[-1:] in _TERMINAL_PUNCT


# ---------------------------------------------------------------------------
# Leitura do Markdown
# ---------------------------------------------------------------------------


def parse_blocks(markdown: str) -> list[_Block]:
    """Converte o Markdown em uma sequência linear de títulos e parágrafos."""
    blocks: list[_Block] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("|"):  # linhas de tabela
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            text = clean_text(heading.group("text"))
            if text:
                blocks.append(_Block(kind="heading", text=text))
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            text = clean_text(bullet.group("text"))
            if text:
                blocks.append(_Block(kind="body", text=text, is_bullet=True))
            continue
        text = clean_text(line)
        if text:
            blocks.append(_Block(kind="body", text=text))
    return blocks


def _split_heading_run(run: list[str]) -> list[_Candidate]:
    """Divide uma sequência de títulos consecutivos em candidatos.

    Um novo candidato começa quando aparece um segundo fragmento numerado
    (caso típico de um título de capítulo colado ao primeiro enunciado) e
    termina quando um fragmento encerra com ``?``.
    """
    candidates: list[_Candidate] = []
    current = _Candidate()
    for fragment in run:
        number = None
        only = _ONLY_NUMBER_RE.match(fragment)
        leading = _LEADING_NUMBER_RE.match(fragment)
        trailing = _TRAILING_NUMBER_RE.match(fragment)
        if only:
            number, text = int(only.group("number")), ""
        elif trailing:
            number, text = int(trailing.group("number")), trailing.group("rest")
        elif leading:
            number, text = int(leading.group("number")), leading.group("rest")
        else:
            text = fragment

        if number is not None and current.number is not None:
            candidates.append(current)
            current = _Candidate()

        if number is not None:
            current.number = number
        if text.strip() and not _looks_like_author_line(text):
            current.fragments.append(text.strip())

        if text.rstrip().endswith("?"):
            candidates.append(current)
            current = _Candidate()

    if current.number is not None or current.fragments:
        candidates.append(current)
    return candidates


def _drop_unrelated_prefixes(candidate: _Candidate) -> None:
    """Descarta títulos de capítulo colados ao início do enunciado.

    Um enunciado partido só é continuação legítima do fragmento anterior se
    o fragmento final começa em minúscula (``... superar a`` +
    ``dormência primária?``) ou se o anterior termina em palavra funcional
    (``... para o desenvolvimento das`` + ``Bananeiras comerciais?``).
    Fora desses casos, o fragmento com ``?`` já é a pergunta completa e o
    que vem antes é título de seção.
    """
    fragments = candidate.fragments
    if len(fragments) < 2:
        return
    last = fragments[-1].strip()
    if not last.endswith("?") or _starts_lowercase(last):
        return
    if _CONTINUATION_END_RE.search(fragments[-2].strip()):
        return
    candidate.fragments = [last]


def build_candidates(blocks: list[_Block]) -> list[_Candidate]:
    """Agrupa os blocos em candidatos (enunciado + blocos de resposta)."""
    candidates: list[_Candidate] = []
    run: list[str] = []
    index = 0
    total = len(blocks)

    while index < total:
        block = blocks[index]
        if block.kind == "heading":
            run.append(block.text)
            index += 1
            continue
        if run:
            candidates.extend(_split_heading_run(run))
            run = []
        if candidates:
            candidates[-1].answer_blocks.append(block)
        index += 1

    if run:
        candidates.extend(_split_heading_run(run))
    return candidates


def _merge_orphan_fragments(candidates: list[_Candidate]) -> list[_Candidate]:
    """Reúne enunciados invertidos com o fragmento órfão que os antecede.

    ``## 194 semente e a dormência?`` seguido de
    ``## Há alguma correlação entre caracteres morfológicos da`` é um único
    enunciado cuja ordem de leitura foi invertida pela diagramação.
    """
    merged: list[_Candidate] = []
    index = 0
    while index < len(candidates):
        current = candidates[index]
        following = candidates[index + 1] if index + 1 < len(candidates) else None
        is_orphan = (
            following is not None
            and following.number is None
            and not following.has_question_mark
            and bool(following.fragments)
        )
        needs_prefix = current.has_question_mark and _starts_lowercase(
            join_fragments(current.fragments)
        )
        if is_orphan and needs_prefix:
            current.fragments = following.fragments + current.fragments
            current.answer_blocks = following.answer_blocks + current.answer_blocks
            current.repaired = True
            index += 2
        else:
            index += 1
        merged.append(current)
    return merged


def _borrow_prefix_from_previous_answer(candidates: list[_Candidate]) -> None:
    """Recupera o começo do enunciado que caiu no fim da resposta anterior.

    Ocorre quando a Docling reconhece apenas a última linha do enunciado
    como título: ``... para a agri-`` fica no corpo do texto e
    ``## 192 cultura?`` vira o título.
    """
    for index in range(1, len(candidates)):
        current = candidates[index]
        if not current.has_question_mark:
            continue
        question = join_fragments(current.fragments)
        if not _starts_lowercase(question):
            continue
        previous = candidates[index - 1]
        if not previous.answer_blocks:
            continue
        tail = previous.answer_blocks[-1]
        if tail.is_bullet or _ends_sentence(tail.text):
            continue
        if len(tail.text.split()) > INLINE_QUESTION_MAX_WORDS:
            continue
        previous.answer_blocks.pop()
        current.fragments = [tail.text] + current.fragments
        current.repaired = True


def _recover_inline_questions(candidates: list[_Candidate]) -> list[_Candidate]:
    """Extrai enunciados que a Docling deixou no corpo do texto.

    Um parágrafo (não item de lista) que termina com ``?``, tem tamanho de
    pergunta e é seguido por outro parágrafo é tratado como um novo item da
    coleção; o que vem depois dele passa a ser sua resposta.
    """
    result: list[_Candidate] = []
    for candidate in candidates:
        blocks = candidate.answer_blocks
        current = candidate
        current.answer_blocks = []
        for position, block in enumerate(blocks):
            is_question_like = (
                not block.is_bullet
                and block.text.rstrip().endswith("?")
                and INLINE_QUESTION_MIN_WORDS
                <= len(block.text.split())
                <= INLINE_QUESTION_MAX_WORDS
                and not _starts_lowercase(block.text)
                and position < len(blocks) - 1
                and bool(current.answer_blocks)
            )
            if is_question_like:
                result.append(current)
                current = _Candidate(fragments=[block.text], repaired=True)
                continue
            current.answer_blocks.append(block)
        result.append(current)
    return result


def assemble_answer(blocks: list[_Block]) -> str:
    """Monta o texto da resposta a partir dos blocos.

    Os itens de lista são convertidos em frases (o tokenizador SentencePiece
    do T5 descarta quebras de linha, portanto a resposta é entregue ao
    modelo como um único parágrafo coeso). Blocos que aparentam ser
    referências bibliográficas são descartados.
    """
    parts: list[str] = []
    for block in blocks:
        text = block.text.strip()
        if not text or _looks_like_reference(text):
            continue
        if block.is_bullet and not _ends_sentence(text):
            text = f"{text}."
        parts.append(text)

    # Remove referências residuais no fim do documento.
    while parts and _looks_like_reference(parts[-1]):
        parts.pop()

    answer = join_fragments(parts)
    return re.sub(r"\s+([,.;:!?])", r"\1", answer).strip()


def normalize_question(text: str) -> str:
    """Padroniza o enunciado: espaços, inicial maiúscula e ``?`` final."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text.strip())
    text = re.sub(r"\s+", " ", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and not text.endswith("?"):
        text = f"{text.rstrip('.')}?"
    return text


# ---------------------------------------------------------------------------
# Extração por arquivo
# ---------------------------------------------------------------------------


def extract_from_markdown(
    markdown_path: Path,
    source_pdf: str | None = None,
    recover_inline_questions: bool = True,
) -> tuple[list[QAPair], ExtractionStats]:
    """Extrai os pares pergunta/resposta de um arquivo Markdown."""
    markdown = markdown_path.read_text(encoding="utf-8")
    slug = topic_slug_from_filename(markdown_path)
    label = topic_label(slug)
    stats = ExtractionStats(
        markdown_file=markdown_path.name, topic_slug=slug, topic=label
    )

    blocks = parse_blocks(markdown)
    stats.headings_seen = sum(1 for block in blocks if block.kind == "heading")

    candidates = build_candidates(blocks)
    candidates = _merge_orphan_fragments(candidates)
    _borrow_prefix_from_previous_answer(candidates)
    for candidate in candidates:
        _drop_unrelated_prefixes(candidate)
    if recover_inline_questions:
        before = len(candidates)
        candidates = _recover_inline_questions(candidates)
        stats.recovered_from_body = max(0, len(candidates) - before)

    stats.candidates = len(candidates)
    pairs: list[QAPair] = []
    last_number: int | None = None
    used_numbers: set[int] = set()

    for candidate in candidates:
        question = normalize_question(join_fragments(candidate.fragments))
        if not candidate.has_question_mark:
            stats.rejected_no_question_mark += 1
            continue
        words = question.split()
        if len(question) < MIN_QUESTION_CHARS or len(words) < MIN_QUESTION_WORDS:
            stats.rejected_short_question += 1
            continue
        if len(words) > MAX_QUESTION_WORDS:
            stats.rejected_long_question += 1
            continue
        answer = assemble_answer(candidate.answer_blocks)
        answer_words = answer.split()
        if len(answer) < MIN_ANSWER_CHARS or len(answer_words) < MIN_ANSWER_WORDS:
            stats.rejected_short_answer += 1
            continue
        # Respostas curtas que apenas remetem a uma tabela ou figura não
        # convertida do PDF não têm conteúdo aproveitável para o modelo.
        if len(answer_words) < 15 and _ONLY_REFERS_TO_EXHIBIT_RE.search(answer):
            stats.rejected_exhibit_only += 1
            continue

        number = candidate.number
        inferred = False
        if number is None and last_number is not None:
            guess = last_number + 1
            if guess not in used_numbers:
                number, inferred = guess, True
        if number is not None:
            if number in used_numbers:
                number, inferred = None, False
            else:
                used_numbers.add(number)
                last_number = number

        suffix = str(number) if number is not None else f"x{len(pairs) + 1:03d}"
        pairs.append(
            QAPair(
                id=f"{slug}-{suffix}",
                topic_slug=slug,
                topic=label,
                number=number,
                question=question,
                answer=answer,
                source_pdf=source_pdf or f"{markdown_path.stem}.pdf",
                source_markdown=markdown_path.name,
                number_inferred=inferred,
                repaired=candidate.repaired,
            )
        )
        stats.accepted += 1
        if inferred:
            stats.inferred_numbers += 1
        if candidate.repaired:
            stats.repaired_questions += 1

    numbers = sorted(n for n in (pair.number for pair in pairs) if n is not None)
    stats.number_range = [numbers[0], numbers[-1]] if numbers else []
    return pairs, stats


def list_markdown_files(interim_dir: Path = INTERIM_DIR) -> list[Path]:
    """Lista os arquivos Markdown convertidos, em ordem alfabética."""
    if not interim_dir.is_dir():
        raise FileNotFoundError(
            f"Diretório de Markdown não encontrado: {interim_dir}. "
            "Execute a etapa de conversão primeiro."
        )
    return sorted(interim_dir.glob("*.md"))


def dedupe_pairs(pairs: list[QAPair]) -> list[QAPair]:
    """Remove pares repetidos (mesmo tema e mesma pergunta normalizada)."""
    seen: set[tuple[str, str]] = set()
    unique: list[QAPair] = []
    for pair in pairs:
        key = (pair.topic_slug, " ".join(pair.question.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def extract_all(
    interim_dir: Path = INTERIM_DIR,
    output_file: Path = QA_DATASET_FILE,
    recover_inline_questions: bool = True,
    report_file: Path | None = None,
    run_name: str = "extracao",
) -> tuple[list[QAPair], list[ExtractionStats]]:
    """Extrai os pares de todos os Markdown e grava o arquivo JSON Lines.

    A etapa é rastreada no MLflow: os limiares de aceitação entram como
    parâmetros, os contadores agregados como métricas e o diagnóstico por
    arquivo como artefato, junto da linhagem do conjunto gerado.
    """
    from app import tracking
    from app.dataset import write_json

    all_pairs: list[QAPair] = []
    all_stats: list[ExtractionStats] = []

    with tracking.start_run(run_name):
        tracking.log_params(
            {
                "interim_dir": interim_dir,
                "recuperar_enunciados_no_corpo": recover_inline_questions,
                "min_palavras_pergunta": MIN_QUESTION_WORDS,
                "max_palavras_pergunta": MAX_QUESTION_WORDS,
                "min_caracteres_resposta": MIN_ANSWER_CHARS,
                "min_palavras_resposta": MIN_ANSWER_WORDS,
            }
        )

        for markdown_path in list_markdown_files(interim_dir):
            pairs, stats = extract_from_markdown(
                markdown_path, recover_inline_questions=recover_inline_questions
            )
            logger.info(
                "%s: %d pares aceitos (%d candidatos, %d reparados)",
                markdown_path.name,
                stats.accepted,
                stats.candidates,
                stats.repaired_questions,
            )
            all_pairs.extend(pairs)
            all_stats.append(stats)

        all_pairs = dedupe_pairs(all_pairs)
        ensure_dirs(output_file.parent)
        with output_file.open("w", encoding="utf-8") as handle:
            for pair in all_pairs:
                handle.write(f"{pair.to_json()}\n")
        logger.info("%d pares gravados em %s", len(all_pairs), output_file)

        summary = summarize(all_pairs)
        tracking.log_metrics(
            {key: value for key, value in summary.items() if key != "pairs_per_topic"}
        )
        tracking.log_metrics(summary.get("pairs_per_topic", {}), prefix="pares_tema.")
        tracking.log_metrics(
            {
                "arquivos_processados": len(all_stats),
                "candidatos": sum(item.candidates for item in all_stats),
                "rejeitados_sem_interrogacao": sum(
                    item.rejected_no_question_mark for item in all_stats
                ),
                "rejeitados_resposta_curta": sum(
                    item.rejected_short_answer for item in all_stats
                ),
                "recuperados_do_corpo": sum(
                    item.recovered_from_body for item in all_stats
                ),
            }
        )
        if report_file is not None:
            write_json(
                report_file,
                {
                    "summary": summary,
                    "per_file": [item.as_dict() for item in all_stats],
                },
            )
            tracking.log_artifact(report_file, artifact_path="relatorios")
        tracking.log_dataset(output_file, "qa-embrapa-completo", "extraction", "answer")

    return all_pairs, all_stats


def summarize(pairs: list[QAPair]) -> dict[str, object]:
    """Resume o conjunto extraído (contagens e tamanhos médios)."""
    if not pairs:
        return {"total": 0}
    question_words = [len(pair.question.split()) for pair in pairs]
    answer_words = [len(pair.answer.split()) for pair in pairs]
    per_topic = Counter(pair.topic for pair in pairs)
    return {
        "total": len(pairs),
        "topics": len(per_topic),
        "pairs_per_topic": dict(sorted(per_topic.items())),
        "question_words_mean": round(sum(question_words) / len(question_words), 1),
        "question_words_max": max(question_words),
        "answer_words_mean": round(sum(answer_words) / len(answer_words), 1),
        "answer_words_median": sorted(answer_words)[len(answer_words) // 2],
        "answer_words_max": max(answer_words),
        "repaired": sum(1 for pair in pairs if pair.repaired),
        "inferred_numbers": sum(1 for pair in pairs if pair.number_inferred),
    }


def load_pairs(path: Path = QA_DATASET_FILE) -> list[QAPair]:
    """Recarrega os pares gravados em JSON Lines."""
    if not path.exists():
        raise FileNotFoundError(
            f"Conjunto de dados não encontrado: {path}. "
            "Execute a etapa de extração primeiro."
        )
    pairs: list[QAPair] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(QAPair(**json.loads(line)))
    return pairs
