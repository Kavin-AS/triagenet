"""Command-line interface for TriageNet data operations."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd
import typer

from triagenet.config import DATA_PROCESSED
from triagenet.io import calce_loader, mit_loader, sandia_loader
from triagenet.io.unified_schema import validate_unified_cycle
from triagenet.pipeline import TriagePipeline

app = typer.Typer(help="TriageNet data and pipeline commands.")
INPUT_OPTION = typer.Option(..., "--input")
OUTPUT_OPTION = typer.Option(..., "--output")


@app.callback()
def main() -> None:
    """Run TriageNet CLI subcommands."""


@app.command()
def ingest() -> None:
    """Load all available raw datasets and write the canonical cycle Parquet table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    frames = []
    for name, loader in (
        ("sandia", sandia_loader.load),
        ("calce", calce_loader.load),
        ("mit", mit_loader.load),
    ):
        frame = loader()
        if frame.empty:
            typer.echo(f"{name}: no parseable raw files found")
            continue
        frames.append(frame)
        typer.echo(f"{name}: loaded {frame['cell_id'].nunique()} cells / {len(frame)} cycles")
    if not frames:
        raise typer.BadParameter(
            "No raw datasets were parsed. Run scripts/download_datasets.sh first."
        )

    cycles = pd.concat(frames, ignore_index=True).sort_values(
        ["dataset", "chemistry", "cell_id", "cycle_index"]
    )
    validate_unified_cycle(cycles)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "cycles.parquet"
    cycles.to_parquet(output_path, compression="snappy", index=False)

    total_cells = cycles["cell_id"].nunique()
    total_cycles = len(cycles)
    cells_per_chemistry = cycles.groupby("chemistry")["cell_id"].nunique().to_dict()
    cells_per_dataset = cycles.groupby("dataset")["cell_id"].nunique().to_dict()
    mean_cycles = cycles.groupby("cell_id")["cycle_index"].count().mean()
    typer.echo(f"wrote: {output_path}")
    typer.echo(f"total_cells: {total_cells}")
    typer.echo(f"cells_per_chemistry: {cells_per_chemistry}")
    typer.echo(f"cells_per_dataset: {cells_per_dataset}")
    typer.echo(f"total_cycles: {total_cycles}")
    typer.echo(f"mean_cycles_per_cell: {mean_cycles:.2f}")


@app.command()
def triage(input_path: Path = INPUT_OPTION, output: Path = OUTPUT_OPTION) -> None:
    """Run the full triage pipeline on CSV or Parquet cycles and write verdict JSON."""
    frame = _read_table(input_path)
    pipeline = TriagePipeline()
    verdicts = pipeline.predict_batch(frame.head(1), rule="risk_aware")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdicts.to_json(orient="records", indent=2))
    typer.echo(f"wrote: {output}")


@app.command("evaluate-triage")
def evaluate_triage(input_path: Path = INPUT_OPTION) -> None:
    """Compare triage decision rules on a cycle table."""
    frame = _read_table(input_path)
    pipeline = TriagePipeline()
    rows = []
    for rule in ("naive", "conservative", "risk_aware"):
        result = pipeline.predict_batch(frame, rule=rule)
        rows.append(
            {
                "rule": rule,
                "n": len(result),
                "mean_expected_value_usd": result["expected_value_usd"].mean(),
                "decisions": result["decision"].value_counts().to_dict(),
            }
        )
    for row in rows:
        typer.echo(row)


@app.command()
def serve(port: int = typer.Option(8000, "--port")) -> None:
    """Serve the FastAPI demo backend with uvicorn."""
    subprocess.run(
        ["uvicorn", "api.main:app", "--port", str(port)],
        check=True,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter("Input must be .parquet or .csv")


if __name__ == "__main__":
    app()
