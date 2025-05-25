#!/usr/bin/env python3
"""Classify vacancies as *interesting* or *not interesting*.

Three mutually-exclusive input modes
------------------------------------
1. ``--text``  – one or more vacancy strings.
2. ``--stdin`` – newline-separated vacancies from *stdin*.
3. ``--csv``   – CSV file with a **text** column (optional **label** column).

Output format
-------------
``<interesting|not interesting>    <vacancy text>``
If labels are provided (via CSV), overall accuracy and F1 are printed last.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, f1_score

try:  # PyTorch is optional at inference time
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

cli = typer.Typer(rich_markup_mode="rich")  # CLI entry-point


def _to_ints(raw_preds) -> List[int]:  # noqa: ANN001
    """Convert SetFit predictions → ``list[int]`` (Tensor / ndarray safe)."""
    if isinstance(raw_preds, list):
        return [int(x) for x in raw_preds]

    if isinstance(raw_preds, np.ndarray):
        return [int(x) for x in raw_preds.tolist()]

    if torch is not None and isinstance(raw_preds, torch.Tensor):
        return [int(x) for x in raw_preds.cpu().tolist()]

    return [int(raw_preds)]  # scalar fallback


def _print_results(texts: List[str], labels: List[int]) -> None:
    """Pretty-print predictions one per line."""
    for txt, lbl in zip(texts, labels, strict=True):
        tag = "interesting" if lbl == 1 else "not interesting"
        typer.echo(f"{tag}\t{txt}")


@cli.command()
def predict(
    model_dir: Path = typer.Option(
        Path("models/job_interest_classifier"),
        help="Directory with the trained SetFit model.",
        exists=True,
        readable=True,
    ),
    text: List[str] | None = typer.Option(
        None,
        "--text",
        "-t",
        help="One or more vacancy texts.",
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read vacancy texts from standard input.",
    ),
    csv: Path | None = typer.Option(
        None,
        "--csv",
        help="CSV with a `text` column (and optional `label`).",
    ),
) -> None:
    """
    Predict labels and (optionally) report metrics.

    Exactly **one** of *text*, *stdin* or *csv* must be supplied.
    """
    sources_selected = sum(bool(x) for x in (text, stdin, csv))
    if sources_selected != 1:
        typer.echo(
            "Error: choose exactly one of --text / --stdin / --csv.",
            err=True,
        )
        raise typer.Exit(1)

    classifier: SetFitModel = SetFitModel.from_pretrained(model_dir)

    # ───────────────────────── gather samples ──────────────────────────
    if csv is not None:
        df = pd.read_csv(csv)
        if "text" not in df.columns:
            typer.echo("CSV must contain a 'text' column.", err=True)
            raise typer.Exit(1)

        samples = df["text"].astype(str).tolist()
        gold = df["label"].astype(int).tolist() if "label" in df.columns else None
    elif stdin:
        samples = [line.strip() for line in sys.stdin if line.strip()]
        gold = None
    else:  # --text
        samples = list(text)  # type: ignore[arg-type]
        gold = None

    preds_raw = classifier.predict(samples)
    preds = _to_ints(preds_raw)

    _print_results(samples, preds)

    if gold is not None:
        acc = accuracy_score(gold, preds)
        f1 = f1_score(gold, preds)
        typer.secho(
            f"\naccuracy={acc:.3f}, f1={f1:.3f}",
            fg=typer.colors.GREEN,
        )


if __name__ == "__main__":
    cli()
