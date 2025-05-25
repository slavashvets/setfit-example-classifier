#!/usr/bin/env python3
"""Train a SetFit classifier to flag interesting vacancies."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import typer
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, f1_score

app = typer.Typer()


def read_split(csv_path: Path) -> Dataset:
    """Load a CSV file and return a Hugging Face Dataset."""
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)


def compute_metrics(y_pred: Any, y_test: Any) -> Dict[str, float]:
    """Compute accuracy and F1 score, returning native Python floats."""
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    return {"accuracy": accuracy, "f1": f1}


def build_trainer(
    train_ds: Dataset,
    val_ds: Dataset,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SetFitTrainer:
    """Create and return a configured SetFitTrainer."""
    model = SetFitModel.from_pretrained(model_name, use_differentiable_head=False)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        num_iterations=20,
        num_epochs=4,
        learning_rate=2e-5,
        batch_size=16,
        metric=compute_metrics,
    )
    return trainer


@app.command()
def main(
    train: Path = Path("data/train.csv"),
    val: Path = Path("data/val.csv"),
    out: Path = Path("models/job_interest_classifier"),
) -> None:
    """
    Train and save the SetFit vacancy interest classifier.

    Args:
        train: Path to the training CSV file.
        val:   Path to the validation CSV file.
        out:   Directory where the trained model will be saved.
    """
    train_ds = read_split(train)
    val_ds = read_split(val)

    trainer = build_trainer(train_ds, val_ds)
    trainer.train()

    metrics = trainer.evaluate()
    typer.echo(
        f"Validation — accuracy: {metrics['accuracy']:.3f}, f1: {metrics['f1']:.3f}"
    )

    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    typer.echo(f"Model saved to {out.resolve()}")


if __name__ == "__main__":
    app()
