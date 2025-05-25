#!/usr/bin/env python3
"""Train a SetFit classifier for “interesting vacancy” detection.

This script uses the **new** SetFit ``Trainer`` + ``TrainingArguments`` API.
It avoids the Metal (MPS) “pin_memory” warning by forcing the model onto CPU;
remove ``device="cpu"`` if you want to train on GPU/MPS.

The model is saved to *models/job_interest_classifier* by default.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import typer
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

cli = typer.Typer(rich_markup_mode="rich")


def _load(csv_path: Path) -> Dataset:
    """Load *csv_path* and return it as a Hugging Face Dataset."""
    return Dataset.from_pandas(pd.read_csv(csv_path))


def _metric(y_pred, y_true) -> Dict[str, float]:
    """Return accuracy and F1 score as *native* Python ``float`` values."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


@cli.command()
def train(
    train: Path = Path("data/train.csv"),
    val: Path = Path("data/val.csv"),
    out: Path = Path("models/job_interest_classifier"),
) -> None:
    """Train the classifier and save it to *out*."""
    warnings.filterwarnings("ignore", message="co_lnotab is deprecated")

    train_ds, val_ds = _load(train), _load(val)

    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # remove to use GPU / MPS
    )

    args = TrainingArguments(
        output_dir=str(out),
        num_epochs=4,
        batch_size=16,
        num_iterations=20,
        body_learning_rate=1e-5,
        head_learning_rate=2e-2,
        logging_strategy="epoch",
        save_strategy="no",
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        metric=_metric,
    )

    trainer.train()
    metrics = trainer.evaluate()
    typer.secho(
        f"[bold green]✔ training finished:[/] "
        f"accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}",
    )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    typer.echo(f"Model saved to {out.resolve()}")


if __name__ == "__main__":
    cli()
