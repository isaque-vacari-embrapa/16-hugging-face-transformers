"""Pipeline de conversão PDF para Markdown com a biblioteca Docling.

Lê os PDFs da Coleção 500 Perguntas 500 Respostas da Embrapa em
``data/raw`` e grava os equivalentes em Markdown em ``data/interim``.

A biblioteca Docling é usada porque preserva a estrutura lógica do documento: os
enunciados das perguntas são reconhecidos como títulos de seção
(``## 147 O que é dormência de sementes?``) e os cabeçalhos/rodapés de
página (numeração, nome da coleção) são descartados na exportação.
Isso é o que torna viável extrair os pares pergunta/resposta de forma determinística
no passo seguinte (:mod:`app.extract`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.config import INTERIM_DIR, RAW_DIR, ensure_dirs
from app.topics import topic_label_from_filename, topic_slug_from_filename

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Resultado da conversão de um único PDF."""

    pdf_path: Path
    markdown_path: Path
    topic_slug: str
    topic_label: str
    characters: int
    elapsed_seconds: float
    skipped: bool = False

    def as_dict(self) -> dict[str, object]:
        """Serializa o resultado para relatórios em JSON."""
        return {
            "pdf": self.pdf_path.name,
            "markdown": self.markdown_path.name,
            "topic_slug": self.topic_slug,
            "topic_label": self.topic_label,
            "characters": self.characters,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "skipped": self.skipped,
        }


def list_pdfs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Lista os PDFs disponíveis em ordem alfabética."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Diretório de PDFs não encontrado: {raw_dir}")
    return sorted(raw_dir.glob("*.pdf"))


def build_converter(use_gpu: bool = True, num_threads: int = 8):
    """Cria um ``DocumentConverter`` afinado para os PDFs da coleção.

    O OCR é desligado porque os PDFs já contêm texto digital, e a
    detecção de estrutura de tabelas também, pois o conteúdo relevante
    (perguntas e respostas) é textual. Ambos são as etapas mais custosas
    do pipeline padrão da Docling.
    """
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device = AcceleratorDevice.AUTO if use_gpu else AcceleratorDevice.CPU
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )


def markdown_path_for(pdf_path: Path, interim_dir: Path = INTERIM_DIR) -> Path:
    """Define o caminho de saída em Markdown para um PDF de entrada."""
    return interim_dir / f"{topic_slug_from_filename(pdf_path)}.md"


def convert_pdf(
    pdf_path: Path,
    interim_dir: Path = INTERIM_DIR,
    converter=None,
    overwrite: bool = False,
) -> ConversionResult:
    """Converte um PDF para Markdown e grava o arquivo em ``interim_dir``."""
    ensure_dirs(interim_dir)
    output_path = markdown_path_for(pdf_path, interim_dir)
    slug = topic_slug_from_filename(pdf_path)
    label = topic_label_from_filename(pdf_path)

    if output_path.exists() and not overwrite:
        logger.info("Ignorando %s (Markdown já existe)", pdf_path.name)
        return ConversionResult(
            pdf_path=pdf_path,
            markdown_path=output_path,
            topic_slug=slug,
            topic_label=label,
            characters=len(output_path.read_text(encoding="utf-8")),
            elapsed_seconds=0.0,
            skipped=True,
        )

    if converter is None:
        converter = build_converter()

    logger.info("Convertendo %s ...", pdf_path.name)
    started = time.perf_counter()
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    elapsed = time.perf_counter() - started

    output_path.write_text(markdown, encoding="utf-8")
    logger.info(
        "%s -> %s (%d caracteres em %.1fs)",
        pdf_path.name,
        output_path.name,
        len(markdown),
        elapsed,
    )
    return ConversionResult(
        pdf_path=pdf_path,
        markdown_path=output_path,
        topic_slug=slug,
        topic_label=label,
        characters=len(markdown),
        elapsed_seconds=elapsed,
    )


def convert_all(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    overwrite: bool = False,
    use_gpu: bool = True,
    pdf_paths: Iterable[Path] | None = None,
) -> Iterator[ConversionResult]:
    """Converte todos os PDFs da coleção, um a um (gerador preguiçoso).

    A conversão é sequencial de propósito: os modelos de layout da Docling
    já paralelizam internamente e disputariam a mesma GPU/CPU.
    """
    pdfs = list(pdf_paths) if pdf_paths is not None else list_pdfs(raw_dir)
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {raw_dir}")

    converter = None
    for pdf_path in pdfs:
        output_path = markdown_path_for(pdf_path, interim_dir)
        if converter is None and not (output_path.exists() and not overwrite):
            converter = build_converter(use_gpu=use_gpu)
        yield convert_pdf(
            pdf_path,
            interim_dir=interim_dir,
            converter=converter,
            overwrite=overwrite,
        )
