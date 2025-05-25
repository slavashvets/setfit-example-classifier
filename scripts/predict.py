#!/usr/bin/env python3
"""Predict whether supplied vacancies are interesting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import typer
from setfit import SetFitModel

app = typer.Typer()


def _to_int_list(preds: Any) -> List[int]:
    """Convert SetFit predictions to a plain list of `int`.

    The `.predict()` method may return:
    * Python `list`
    * NumPy `ndarray`
    * PyTorch `Tensor`
    * Scalar `int`

    Args:
        preds: Raw predictions from `SetFitModel.predict`.

    Returns:
        List of Python `int` values.
    """
    if isinstance(preds, list):
        return [int(x) for x in preds]
    if hasattr(preds, "tolist"):  # ndarray / Tensor
        return [int(x) for x in preds.tolist()]
    return [int(preds)]  # single scalar


@app.command()
def main(
    model_dir: Path = Path("models/job_interest_classifier"),
    text: List[str] | None = typer.Option(
        None,
        "--text",
        "-t",
        help="One or more vacancy texts to classify.",
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read newline-separated vacancies from stdin.",
    ),
) -> None:
    """
    Classify vacancies as *interesting* (1) or *not interesting* (0).

    Exactly one of *text* or *stdin* must be provided.

    Args:
        model_dir: Directory with a trained SetFit model.
        text:      Vacancy strings passed via CLI.
        stdin:     Read vacancy strings from standard input if *True*.
    """
    if not stdin and not text:
        typer.echo("Error: either --text or --stdin must be provided.", err=True)
        raise typer.Exit(code=1)

    classifier: SetFitModel = SetFitModel.from_pretrained(model_dir)

    if stdin:
        vacancy_texts: List[str] = [line.strip() for line in sys.stdin if line.strip()]
    else:
        vacancy_texts = list(text)  # type: ignore[arg-type]

    raw_preds = classifier.predict(vacancy_texts)
    predictions: List[int] = _to_int_list(raw_preds)

    for vacancy, label in zip(vacancy_texts, predictions, strict=True):
        tag = "interesting" if label == 1 else "not interesting"
        typer.echo(f"{tag}\t{vacancy}")


if __name__ == "__main__":
    app()
