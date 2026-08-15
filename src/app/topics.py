"""Normalização dos temas (culturas) da coleção da Embrapa.

Os nomes dos arquivos PDF seguem o padrão
``<Tema>-o-produtor-pergunta-a-embrapa-responde.pdf``. Este módulo
converte esse trecho em um identificador estável (``slug``) e em um rótulo
legível e acentuado, usado no relatório e no playground.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

#: Sufixo comum a todos os arquivos da coleção.
FILENAME_SUFFIX = "-o-produtor-pergunta-a-embrapa-responde"

#: Rótulos legíveis. As chaves são os ``slugs`` extraídos do nome do arquivo.
#: Corrige também a grafia de ``caprionos`` (erro presente no nome original).
TOPIC_LABELS: dict[str, str] = {
    "abacaxi": "Abacaxi",
    "algodao": "Algodão",
    "arroz": "Arroz",
    "banana": "Banana",
    "caju": "Caju",
    "caprionos-e-ovinos-de-corte": "Caprinos e Ovinos de Corte",
    "citros": "Citros",
    "coco": "Coco",
    "feijao": "Feijão",
    "feijao-caupi": "Feijão-Caupi",
    "gado-de-leite": "Gado de Leite",
    "hortas": "Hortas",
    "mamao": "Mamão",
    "mamona": "Mamona",
    "mandioca": "Mandioca",
    "milho": "Milho",
    "producao-organica-de-hortalicas": "Produção Orgânica de Hortaliças",
    "sementes": "Sementes",
    "sistema-plantio-direto": "Sistema Plantio Direto",
    "soja": "Soja",
}


def slugify(text: str) -> str:
    """Reduz um texto a letras minúsculas sem acentos, separadas por hífen."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text.lower()


def topic_slug_from_filename(path: Path | str) -> str:
    """Extrai o ``slug`` do tema a partir do nome do arquivo."""
    stem = Path(path).stem
    slug = slugify(stem)
    suffix = slugify(FILENAME_SUFFIX)
    if slug.endswith(suffix):
        slug = slug[: -len(suffix)].strip("-")
    return slug or "geral"


def topic_label(slug: str) -> str:
    """Devolve o rótulo legível do tema, com acentuação quando conhecido."""
    if slug in TOPIC_LABELS:
        return TOPIC_LABELS[slug]
    return " ".join(word.capitalize() for word in slug.split("-"))


def topic_label_from_filename(path: Path | str) -> str:
    """Atalho para ``topic_label(topic_slug_from_filename(path))``."""
    return topic_label(topic_slug_from_filename(path))
