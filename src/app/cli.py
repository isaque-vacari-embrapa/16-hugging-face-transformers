"""Interface de linha de comando do projeto (``qa-embrapa``).

Cada subcomando corresponde a uma etapa do pipeline::

    convert  ->  extract  ->  dataset  ->  train  ->  evaluate  ->  playground

O comando ``pipeline`` executa a sequência inteira.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import click

from app.config import (
    COMPARISON_MARKDOWN_FILE,
    COMPARISON_REPORT_FILE,
    DATASET_REPORT_FILE,
    EXTRACTION_REPORT_FILE,
    INTERIM_DIR,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    QA_DATASET_FILE,
    RAW_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    DatasetSplitConfig,
    ensure_dirs,
)
from app.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_KEY,
    generation_config_for,
    resolve_spec,
    training_config_for,
)
from app.topics import TOPIC_LABELS

#: Opção comum aos subcomandos que operam sobre um modelo específico.
MODEL_CHOICE = click.Choice(sorted(AVAILABLE_MODELS), case_sensitive=False)


def model_option(help_suffix: str = ""):
    """Decorador com a opção ``--model``, compartilhada por vários comandos."""
    linhas = " | ".join(
        f"{spec.key}: {spec.hf_name}" for spec in AVAILABLE_MODELS.values()
    )
    return click.option(
        "--model",
        "model_key",
        type=MODEL_CHOICE,
        default=DEFAULT_MODEL_KEY,
        show_default=True,
        help=f"Modelo a usar ({linhas}).{help_suffix}",
    )


# Precisa ser definido antes de qualquer alocação na GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

#: Cenário de demonstração: perguntas de exemplo por tema.
SAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("Soja", "Qual é a melhor época de semeadura da soja?"),
    ("Soja", "Como identificar a ferrugem asiática na lavoura?"),
    ("Milho", "Qual o espaçamento recomendado para o plantio de milho?"),
    ("Gado de Leite", "Como aumentar a produção de leite no período seco?"),
    ("Gado de Leite", "O que é mastite e como preveni-la no rebanho?"),
    ("Mandioca", "Como escolher as manivas para o plantio da mandioca?"),
    ("Hortas", "Como preparar o solo de uma horta doméstica?"),
    ("Café", "Quando devo irrigar a lavoura em período de estiagem?"),
    ("Banana", "Quais os sintomas do mal-do-panamá na bananeira?"),
    ("Sementes", "Como saber se uma semente tem dormência?"),
]


def configure_logging(verbose: bool) -> None:
    """Configura o log da aplicação e silencia bibliotecas verbosas."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "fsspec", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def echo_json(payload: object) -> None:
    """Imprime uma estrutura como JSON legível."""
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", is_flag=True, help="Ativa log em nível DEBUG.")
@click.option(
    "--no-mlflow",
    is_flag=True,
    help="Desliga o rastreamento no MLflow (útil em testes rápidos).",
)
@click.option(
    "--experiment",
    default=MLFLOW_EXPERIMENT_NAME,
    show_default=True,
    help="Nome do experimento no MLflow.",
)
@click.version_option(
    package_name="16-hugging-face-transformers", prog_name="qa-embrapa"
)
def cli(verbose: bool, no_mlflow: bool, experiment: str) -> None:
    """Sistema generativo de perguntas e respostas da coleção Embrapa."""
    configure_logging(verbose)

    from app import tracking

    if no_mlflow:
        tracking.disable()
    else:
        tracking.configure(experiment_name=experiment)


# ---------------------------------------------------------------------------
# 1. Conversão PDF -> Markdown (Docling)
# ---------------------------------------------------------------------------


@cli.command("convert")
@click.option(
    "--raw-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=RAW_DIR,
    show_default=True,
    help="Diretório com os PDFs de origem.",
)
@click.option(
    "--interim-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=INTERIM_DIR,
    show_default=True,
    help="Diretório de saída dos arquivos Markdown.",
)
@click.option("--overwrite", is_flag=True, help="Reconverte PDFs já convertidos.")
@click.option("--cpu", is_flag=True, help="Força a conversão na CPU.")
def convert_command(
    raw_dir: Path, interim_dir: Path, overwrite: bool, cpu: bool
) -> None:
    """Converte os PDFs de data/raw para Markdown em data/interim."""
    from app.convert import convert_all, list_pdfs

    pdfs = list_pdfs(raw_dir)
    results = []
    # ``length`` permite consumir o gerador de forma preguiçosa: a barra
    # avança a cada PDF, em vez de só aparecer no fim da conversão.
    with click.progressbar(
        convert_all(raw_dir, interim_dir, overwrite, use_gpu=not cpu, pdf_paths=pdfs),
        length=len(pdfs),
        label="Convertendo PDFs",
        item_show_func=lambda item: item.pdf_path.name if item else "",
    ) as items:
        for result in items:
            results.append(result.as_dict())

    converted = sum(1 for item in results if not item["skipped"])
    click.secho(
        f"{len(results)} arquivos processados "
        f"({converted} convertidos, {len(results) - converted} já existentes).",
        fg="green",
    )
    click.echo(f"Markdown em: {interim_dir}")


# ---------------------------------------------------------------------------
# 2. Extração dos pares de pergunta e resposta
# ---------------------------------------------------------------------------


@cli.command("extract")
@click.option(
    "--interim-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=INTERIM_DIR,
    show_default=True,
    help="Diretório com os arquivos Markdown.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=QA_DATASET_FILE,
    show_default=True,
    help="Arquivo JSON Lines de saída.",
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=EXTRACTION_REPORT_FILE,
    show_default=True,
    help="Arquivo JSON com o diagnóstico da extração.",
)
@click.option(
    "--no-recover",
    is_flag=True,
    help="Não tenta recuperar enunciados que a Docling deixou no corpo do texto.",
)
def extract_command(
    interim_dir: Path, output: Path, report: Path, no_recover: bool
) -> None:
    """Extrai os pares pergunta/resposta do Markdown para data/processed."""
    from app.extract import extract_all, summarize

    pairs, _ = extract_all(
        interim_dir=interim_dir,
        output_file=output,
        recover_inline_questions=not no_recover,
        report_file=report,
    )
    echo_json(summarize(pairs))
    click.secho(f"{len(pairs)} pares gravados em {output}", fg="green")
    click.echo(f"Relatório: {report}")


# ---------------------------------------------------------------------------
# 3. Partições do conjunto de dados
# ---------------------------------------------------------------------------


@cli.command("dataset")
@click.option(
    "--input",
    "input_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=QA_DATASET_FILE,
    show_default=True,
    help="Arquivo JSON Lines com os pares extraídos.",
)
@click.option(
    "--splits-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=SPLITS_DIR,
    show_default=True,
    help="Diretório de saída das partições.",
)
@click.option(
    "--validation-size",
    type=float,
    default=0.1,
    show_default=True,
    help="Fração destinada à validação.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.05,
    show_default=True,
    help="Fração destinada ao teste.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Semente.")
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DATASET_REPORT_FILE,
    show_default=True,
    help="Arquivo JSON com o resumo das partições.",
)
def dataset_command(
    input_file: Path,
    splits_dir: Path,
    validation_size: float,
    test_size: float,
    seed: int,
    report: Path,
) -> None:
    """Divide os pares em treino, validação e teste."""
    from app.dataset import build_splits

    statistics = build_splits(
        input_file=input_file,
        splits_dir=splits_dir,
        config=DatasetSplitConfig(
            validation_size=validation_size, test_size=test_size, seed=seed
        ),
        report_file=report,
    )
    echo_json({key: value for key, value in statistics.items() if key != "topics"})
    click.secho(f"Partições gravadas em {splits_dir}", fg="green")


# ---------------------------------------------------------------------------
# 4. Treinamento
# ---------------------------------------------------------------------------


@cli.command("train")
@model_option()
@click.option(
    "--splits-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=SPLITS_DIR,
    show_default=True,
    help="Diretório com as partições.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destino do modelo ajustado (padrão: definido pelo modelo escolhido).",
)
@click.option("--base-model", default=None, help="Checkpoint pré-treinado de origem.")
@click.option("--epochs", type=float, default=None, help="Número de épocas.")
@click.option("--batch-size", type=int, default=None, help="Lote por dispositivo.")
@click.option(
    "--grad-accum", type=int, default=None, help="Passos de acumulação de gradiente."
)
@click.option("--learning-rate", type=float, default=None, help="Taxa de aprendizado.")
@click.option(
    "--max-target-length",
    type=int,
    default=None,
    help="Tamanho máximo da resposta em tokens.",
)
@click.option(
    "--gradient-checkpointing",
    is_flag=True,
    help="Economiza VRAM ao custo de ~30% de tempo.",
)
@click.option(
    "--max-train-examples",
    type=int,
    default=None,
    help="Limita o treino a N exemplos (útil para testes rápidos).",
)
@click.option("--resume", is_flag=True, help="Retoma do último checkpoint.")
@click.option(
    "--no-register",
    is_flag=True,
    help="Treina sem empacotar nem registrar o modelo no Model Registry.",
)
@click.option(
    "--run-name",
    default=None,
    help="Nome da execução no MLflow (padrão: treino-<modelo>).",
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Arquivo JSON do relatório de treinamento.",
)
def train_command(
    model_key: str,
    splits_dir: Path,
    output_dir: Path | None,
    base_model: str | None,
    epochs: float | None,
    batch_size: int | None,
    grad_accum: int | None,
    learning_rate: float | None,
    max_target_length: int | None,
    gradient_checkpointing: bool,
    max_train_examples: int | None,
    resume: bool,
    no_register: bool,
    run_name: str | None,
    report: Path | None,
) -> None:
    """Ajusta o modelo pré-treinado escolhido nos dados da Embrapa."""
    from app.train import train

    spec = resolve_spec(model_key)
    overrides: dict[str, object] = {}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if base_model:
        overrides["base_model"] = base_model
    if epochs is not None:
        overrides["num_train_epochs"] = epochs
    if batch_size is not None:
        overrides["per_device_train_batch_size"] = batch_size
        overrides["per_device_eval_batch_size"] = batch_size
    if grad_accum is not None:
        overrides["gradient_accumulation_steps"] = grad_accum
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if max_target_length is not None:
        overrides["max_target_length"] = max_target_length
    if gradient_checkpointing:
        overrides["gradient_checkpointing"] = True
    config = training_config_for(spec, **overrides)

    click.secho(
        f"Modelo: {spec.label} ({spec.hf_name}, {spec.parameters} parâmetros) — "
        f"{'LoRA em 4 bits' if spec.use_lora else 'ajuste completo'}",
        fg="cyan",
        bold=True,
    )
    resultado = train(
        config=config,
        splits_dir=splits_dir,
        max_train_examples=max_train_examples,
        resume_from_checkpoint=resume,
        register_model=not no_register,
        run_name=run_name or f"treino-{spec.key}",
        report_path=report,
        spec=spec,
    )
    echo_json(
        {
            "modelo": spec.key,
            "elapsed_seconds": resultado["elapsed_seconds"],
            "train_metrics": resultado["train_metrics"],
            "validation_metrics": resultado["validation_metrics"],
            "mlflow_run_id": resultado.get("mlflow_run_id"),
            "mlflow_model_uri": resultado.get("mlflow_model_uri"),
        }
    )
    click.secho(f"Modelo ajustado salvo em {config.output_dir}", fg="green")
    if resultado.get("mlflow_model_uri"):
        click.secho(
            f"Registrado no MLflow como {resultado['mlflow_model_uri']} "
            f"(alias '{MLFLOW_CHAMPION_ALIAS}')",
            fg="green",
        )


# ---------------------------------------------------------------------------
# 5. Avaliação
# ---------------------------------------------------------------------------


@cli.command("evaluate")
@model_option()
@click.option(
    "--model-dir",
    default=None,
    help=(
        "Sobrescreve a origem dos pesos: diretório local, nome no Hub ou "
        "URI do MLflow (ex.: models:/qa-embrapa-ptt5@champion)."
    ),
)
@click.option(
    "--splits-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=SPLITS_DIR,
    show_default=True,
    help="Diretório com as partições.",
)
@click.option(
    "--limit",
    type=int,
    default=200,
    show_default=True,
    help="Número de exemplos de teste avaliados (0 = todos).",
)
@click.option("--batch-size", type=int, default=8, show_default=True, help="Lote.")
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Arquivo JSON do relatório (padrão: definido pelo modelo escolhido).",
)
def evaluate_command(
    model_key: str,
    model_dir: str | None,
    splits_dir: Path,
    limit: int,
    batch_size: int,
    report: Path | None,
) -> None:
    """Avalia qualidade e originalidade das respostas geradas."""
    from app.evaluation import evaluate_model

    spec = resolve_spec(model_key)
    result = evaluate_model(
        model_dir=model_dir or spec.output_dir,
        splits_dir=splits_dir,
        limit=limit or None,
        batch_size=batch_size,
        report_path=report or spec.evaluation_report_file,
        spec=spec,
        run_name=f"avaliacao-{spec.key}",
    )
    resumo = {
        name: {"quality": data["quality"], "originality": data["originality"]}
        for name, data in result["presets"].items()
    }
    echo_json(resumo)
    click.secho(
        f"Relatório completo em {report or spec.evaluation_report_file}", fg="green"
    )
    if result.get("mlflow_run_id"):
        click.echo(f"Execução MLflow: {result['mlflow_run_id']}")


# ---------------------------------------------------------------------------
# 6. Geração pontual e cenário de demonstração
# ---------------------------------------------------------------------------


@cli.command("ask")
@click.argument("question")
@model_option()
@click.option("--topic", default=None, help="Tema/cultura (ex.: Soja, Gado de Leite).")
@click.option(
    "--model-dir",
    default=None,
    help=(
        "Sobrescreve a origem dos pesos: diretório local, nome no Hub ou "
        "URI do MLflow (ex.: models:/qa-embrapa-ptt5@champion)."
    ),
)
@click.option(
    "--creative",
    is_flag=True,
    help="Usa amostragem (mais original) em vez de busca em feixe.",
)
@click.option("--temperature", type=float, default=0.9, show_default=True)
@click.option("--top-p", type=float, default=0.92, show_default=True)
@click.option(
    "--num-beams",
    type=int,
    default=None,
    help="Feixes na busca (padrão: definido pelo modelo escolhido).",
)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option(
    "--variants",
    type=int,
    default=1,
    show_default=True,
    help="Quantidade de respostas alternativas.",
)
def ask_command(
    question: str,
    model_key: str,
    topic: str | None,
    model_dir: str | None,
    creative: bool,
    temperature: float,
    top_p: float,
    num_beams: int | None,
    max_new_tokens: int,
    variants: int,
) -> None:
    """Gera a resposta para uma pergunta."""
    from app.generate import QAGenerator

    spec = resolve_spec(model_key)
    generator = QAGenerator(model_dir or spec.output_dir, spec=spec)
    config = generation_config_for(
        spec,
        do_sample=creative,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=variants,
    )
    if num_beams is not None:
        config = replace(config, num_beams=num_beams)
    if variants > 1 and not creative:
        config = replace(config, num_beams=max(config.num_beams, variants))

    answers = generator.answer_variants(question, topic, config=config)
    click.secho(f"\nTema: {topic or 'geral'}", fg="cyan")
    click.secho(f"Pergunta: {question}\n", fg="cyan", bold=True)
    for index, answer in enumerate(answers, start=1):
        prefix = f"[{index}] " if len(answers) > 1 else ""
        click.echo(f"{prefix}{answer}\n")


@cli.command("samples")
@model_option()
@click.option(
    "--model-dir",
    default=None,
    help=(
        "Sobrescreve a origem dos pesos: diretório local, nome no Hub ou "
        "URI do MLflow (ex.: models:/qa-embrapa-ptt5@champion)."
    ),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Arquivo Markdown (padrão: definido pelo modelo escolhido).",
)
def samples_command(model_key: str, model_dir: str | None, output: Path | None) -> None:
    """Gera o cenário de execução com perguntas e respostas de exemplo."""
    from app.generate import QAGenerator

    spec = resolve_spec(model_key)
    output = output or spec.samples_file
    generator = QAGenerator(model_dir or spec.output_dir, spec=spec)
    pairs = [(question, topic) for topic, question in SAMPLE_QUESTIONS]

    beam_config = generation_config_for(spec, do_sample=False)
    sample_config = generation_config_for(spec, do_sample=True)
    beam = generator.answer_many(pairs, config=beam_config)
    creative = generator.answer_many(
        pairs, config=replace(sample_config, num_return_sequences=1)
    )

    known_topics = set(TOPIC_LABELS.values())
    lines: list[str] = [
        "# Cenário de execução — perguntas e respostas de exemplo",
        "",
        f"Modelo: **{spec.label}** (`{spec.hf_name}`, {spec.parameters} "
        f"parâmetros) — pesos de `{generator.source}`",
        "",
        "Cada pergunta aparece com duas decodificações: **busca em feixe** "
        f"(`num_beams={beam_config.num_beams}`, "
        f"`length_penalty={beam_config.length_penalty}`, determinística e mais "
        "fiel ao material técnico) e **amostragem** "
        f"(`temperature={sample_config.temperature}`, "
        f"`top_p={sample_config.top_p}`, mais original).",
        "",
        "> **Aviso** — as respostas são geradas automaticamente por um modelo "
        f"de {spec.parameters} parâmetros ajustado sobre a coleção e **podem "
        "conter erros técnicos**. Um dos exemplos usa um tema (Café) que não "
        "existe na coleção, para mostrar o comportamento fora do domínio de "
        "treino.",
        "",
    ]
    for index, (topic, question) in enumerate(SAMPLE_QUESTIONS):
        fora = "" if topic in known_topics else " *(tema ausente da coleção)*"
        lines.extend(
            [
                f"## {index + 1}. {topic} — {question}{fora}",
                "",
                "**Busca em feixe**",
                "",
                f"> {beam[index][0] if beam[index] else '(vazio)'}",
                "",
                "**Amostragem criativa**",
                "",
                f"> {creative[index][0] if creative[index] else '(vazio)'}",
                "",
            ]
        )

    ensure_dirs(output.parent)
    output.write_text("\n".join(lines), encoding="utf-8")
    for index, (topic, question) in enumerate(SAMPLE_QUESTIONS):
        click.secho(f"[{topic}] {question}", fg="cyan", bold=True)
        click.echo(f"{beam[index][0] if beam[index] else '(vazio)'}\n")
    click.secho(f"Cenário gravado em {output}", fg="green")


# ---------------------------------------------------------------------------
# 7. Playground (Streamlit)
# ---------------------------------------------------------------------------


@cli.command(
    "playground",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--port", type=int, default=8501, show_default=True, help="Porta HTTP.")
@click.option(
    "--open-browser/--no-open-browser",
    default=False,
    show_default=True,
    help="Abre o navegador automaticamente (exige sessão gráfica local).",
)
@click.pass_context
def playground_command(ctx: click.Context, port: int, open_browser: bool) -> None:
    """Sobe o playground em Streamlit para testar o modelo."""
    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        # Sem o modo headless, a primeira execução do Streamlit na máquina
        # pede um e-mail no terminal e encerra caso a entrada padrão não
        # seja interativa.
        "--server.headless",
        "false" if open_browser else "true",
        *ctx.args,
    ]
    click.secho(f"Playground disponível em http://localhost:{port}", fg="green")
    click.echo("Interrompa com Ctrl+C.")
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 8. MLflow: interface e Model Registry
# ---------------------------------------------------------------------------


@cli.command("mlflow-ui")
@click.option("--port", type=int, default=5000, show_default=True, help="Porta HTTP.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Endereço.")
def mlflow_ui_command(port: int, host: str) -> None:
    """Sobe a interface do MLflow para comparar execuções e modelos."""
    from app import tracking

    tracking.configure()
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        MLFLOW_TRACKING_URI,
        "--host",
        host,
        "--port",
        str(port),
    ]
    click.secho(f"Interface do MLflow em http://{host}:{port}", fg="green")
    click.echo("Interrompa com Ctrl+C.")
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


@cli.command("models")
@click.option(
    "--name",
    default=None,
    help="Nome no Model Registry (padrão: todos os modelos do catálogo).",
)
def models_command(name: str | None) -> None:
    """Lista os modelos disponíveis e suas versões registradas."""
    from app.model_registry import list_registered_versions

    if name:
        alvos = {name: name}
    else:
        alvos = {
            spec.label: spec.registered_model for spec in AVAILABLE_MODELS.values()
        }

    click.secho("Modelos selecionáveis para treino:", bold=True)
    for spec in AVAILABLE_MODELS.values():
        click.echo(
            f"  {spec.key:6s} {spec.hf_name}  "
            f"({spec.parameters}, {spec.family}, "
            f"{'LoRA 4 bits' if spec.use_lora else 'ajuste completo'})"
        )

    total = 0
    for rotulo, registrado in alvos.items():
        versions = list_registered_versions(registrado)
        click.secho(f"\n{rotulo} — `{registrado}`", bold=True)
        if not versions:
            click.secho(
                "  Nenhuma versão registrada. Execute "
                f"`poetry run qa-embrapa train --model {rotulo}`.",
                fg="yellow",
            )
            continue
        total += len(versions)
        echo_json(versions)
    if total:
        click.secho(f"\n{total} versão(ões) em {MLFLOW_TRACKING_URI}", fg="green")


@cli.command("promote")
@click.argument("version")
@click.option(
    "--name",
    default=MLFLOW_REGISTERED_MODEL,
    show_default=True,
    help="Nome do modelo no Model Registry.",
)
@click.option(
    "--alias",
    default=MLFLOW_CHAMPION_ALIAS,
    show_default=True,
    help="Alias a apontar para a versão informada.",
)
def promote_command(version: str, name: str, alias: str) -> None:
    """Aponta um alias do Model Registry para uma versão específica."""
    from app import tracking

    if not tracking.configure():
        raise click.ClickException("Rastreamento no MLflow está desligado.")

    import mlflow

    mlflow.MlflowClient().set_registered_model_alias(name, alias, version)
    click.secho(f"models:/{name}@{alias} -> versão {version}", fg="green")


# ---------------------------------------------------------------------------
# 8. Pipeline completo e diagnóstico
# ---------------------------------------------------------------------------


@cli.command("pipeline")
@click.option(
    "--model",
    "model_keys",
    type=MODEL_CHOICE,
    multiple=True,
    help=(
        "Modelo a treinar e avaliar; repita a opção para vários. "
        f"Padrão: {DEFAULT_MODEL_KEY}."
    ),
)
@click.option("--overwrite", is_flag=True, help="Reconverte os PDFs já convertidos.")
@click.option("--epochs", type=float, default=None, help="Número de épocas do treino.")
@click.option(
    "--eval-limit",
    type=int,
    default=200,
    show_default=True,
    help="Exemplos usados na avaliação.",
)
@click.option("--skip-train", is_flag=True, help="Reaproveita o modelo já ajustado.")
@click.pass_context
def pipeline_command(
    ctx: click.Context,
    model_keys: tuple[str, ...],
    overwrite: bool,
    epochs: float | None,
    eval_limit: int,
    skip_train: bool,
) -> None:
    """Executa o pipeline completo, de PDF a relatório de avaliação.

    Com ``--model`` repetido, treina e avalia cada modelo em sequência e
    grava a comparação entre eles ao final.
    """
    escolhidos = list(model_keys) or [DEFAULT_MODEL_KEY]
    click.secho("[1/5] Convertendo PDFs com a Docling ...", fg="yellow", bold=True)
    ctx.invoke(
        convert_command,
        raw_dir=RAW_DIR,
        interim_dir=INTERIM_DIR,
        overwrite=overwrite,
        cpu=False,
    )
    click.secho("\n[2/5] Extraindo pares pergunta/resposta ...", fg="yellow", bold=True)
    ctx.invoke(
        extract_command,
        interim_dir=INTERIM_DIR,
        output=QA_DATASET_FILE,
        no_recover=False,
    )
    click.secho("\n[3/5] Criando as partições ...", fg="yellow", bold=True)
    ctx.invoke(
        dataset_command,
        input_file=QA_DATASET_FILE,
        splits_dir=SPLITS_DIR,
        validation_size=0.1,
        test_size=0.05,
        seed=42,
    )

    for posicao, chave in enumerate(escolhidos, start=1):
        spec = resolve_spec(chave)
        cabecalho = f"[4/5] Modelo {posicao}/{len(escolhidos)}: {spec.label}"
        if skip_train:
            click.secho(f"\n{cabecalho} — treino ignorado.", fg="yellow", bold=True)
        else:
            click.secho(f"\n{cabecalho} — ajustando ...", fg="yellow", bold=True)
            ctx.invoke(train_command, model_key=chave, epochs=epochs)
        click.secho(f"\n{cabecalho} — avaliando ...", fg="yellow", bold=True)
        ctx.invoke(evaluate_command, model_key=chave, limit=eval_limit)
        click.secho(f"\n{cabecalho} — cenário de exemplo ...", fg="yellow", bold=True)
        ctx.invoke(samples_command, model_key=chave)

    if len(escolhidos) > 1:
        click.secho("\n[5/5] Comparando os modelos ...", fg="yellow", bold=True)
        ctx.invoke(compare_command, model_keys=tuple(escolhidos))

    from app import tracking

    click.secho("\nPipeline concluído.", fg="green", bold=True)
    click.echo("  poetry run qa-embrapa playground   # testar o modelo")
    if tracking.is_enabled():
        click.echo("  poetry run qa-embrapa mlflow-ui    # execuções e Model Registry")


@cli.command("compare")
@click.option(
    "--model",
    "model_keys",
    type=MODEL_CHOICE,
    multiple=True,
    help="Modelos a comparar; repita a opção. Padrão: todos os avaliados.",
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=COMPARISON_REPORT_FILE,
    show_default=True,
    help="Arquivo JSON com os dados da comparação.",
)
@click.option(
    "--markdown",
    type=click.Path(dir_okay=False, path_type=Path),
    default=COMPARISON_MARKDOWN_FILE,
    show_default=True,
    help="Arquivo Markdown com o relatório comparativo.",
)
def compare_command(model_keys: tuple[str, ...], report: Path, markdown: Path) -> None:
    """Compara os modelos já avaliados e recomenda um deles."""
    from app.comparison import compare_models

    chaves = list(model_keys) or None
    resultado = compare_models(chaves, report_path=report, markdown_path=markdown)
    if not resultado["models"]:
        raise click.ClickException(
            "Nenhum relatório de avaliação encontrado. Execute "
            "`poetry run qa-embrapa evaluate --model <chave>` para cada modelo."
        )
    echo_json(resultado["summary"])
    recomendado = resultado["recommendation"]
    click.secho(
        f"\nRecomendado: {recomendado['label']} — {recomendado['reason']}",
        fg="green",
        bold=True,
    )
    click.echo(f"Relatório: {markdown}")


@cli.command("info")
def info_command() -> None:
    """Mostra o estado do ambiente e dos artefatos do projeto."""
    import torch

    from app.generate import is_model_directory

    def count(path: Path, pattern: str) -> int:
        return len(list(path.glob(pattern))) if path.is_dir() else 0

    payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_disponivel": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pdfs_em_raw": count(RAW_DIR, "*.pdf"),
        "markdown_em_interim": count(INTERIM_DIR, "*.md"),
        "dataset_jsonl": QA_DATASET_FILE.exists(),
        "particoes": count(SPLITS_DIR, "*.jsonl"),
        "modelos": {
            spec.key: {
                "checkpoint": spec.hf_name,
                "familia": spec.family,
                "parametros": spec.parameters,
                "ajustado": is_model_directory(spec.output_dir),
                "avaliado": spec.evaluation_report_file.exists(),
            }
            for spec in AVAILABLE_MODELS.values()
        },
        "relatorios": sorted(
            path.name for path in REPORTS_DIR.glob("*") if path.is_file()
        ),
    }
    if QA_DATASET_FILE.exists():
        with QA_DATASET_FILE.open(encoding="utf-8") as handle:
            payload["pares_extraidos"] = sum(1 for line in handle if line.strip())

    from app import tracking
    from app.model_registry import list_registered_versions

    payload["mlflow_ativo"] = tracking.is_enabled()
    payload["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI
    if tracking.is_enabled():
        versions = list_registered_versions()
        payload["mlflow_versoes_registradas"] = len(versions)
        payload["mlflow_alias_champion"] = next(
            (
                item["uri"]
                for item in versions
                if MLFLOW_CHAMPION_ALIAS in item["aliases"]
            ),
            None,
        )
    echo_json(payload)


if __name__ == "__main__":  # pragma: no cover
    cli()
