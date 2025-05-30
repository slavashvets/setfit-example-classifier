import logging
import warnings
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Get rid of some warnings that are not useful for the user.
logging.getLogger("setfit").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"co_lnotab is deprecated, use co_lines instead.",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
)

TRAIN_DF: pd.DataFrame = pd.read_csv("data/train.csv")
EVAL_DF: pd.DataFrame = pd.read_csv("data/eval.csv")
TEST_DF: pd.DataFrame = pd.read_csv("data/test.csv")
BASE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR: Path = Path("models/job_interest_classifier")

console = Console()


def compute_metrics(y_pred, y_true) -> dict[str, float]:
    """Return accuracy and F1 for SetFit trainer."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def classify_texts(
    texts: pd.Series, model_dir: Path, probabilities: bool = False
) -> list:
    """Predict binary labels for a sequence of vacancy texts."""
    model = SetFitModel.from_pretrained(model_dir)

    if probabilities:
        probas = model.predict_proba(texts.to_list())
        return [proba[1] * 100 for proba in probas]
    else:
        raw = model.predict(texts.to_list())
        return np.atleast_1d(raw).astype(int).tolist()


def train() -> dict[str, float]:
    """Exact body of scripts/train.py:main(), encapsulated."""
    model = SetFitModel.from_pretrained(BASE_MODEL, device="cpu")

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_pandas(TRAIN_DF),
        eval_dataset=Dataset.from_pandas(EVAL_DF),
        metric=compute_metrics,
        args=TrainingArguments(
            output_dir=str(MODEL_DIR),
            num_epochs=4,
            batch_size=16,
            num_iterations=20,
            body_learning_rate=1e-5,
            head_learning_rate=2e-2,
            eval_strategy="epoch",
            save_strategy="best",
            logging_strategy="epoch",
        ),
    )

    trainer.train()
    return trainer.evaluate()


def test() -> None:
    """Compute and print accuracy and F1 (if labels present) for test data."""
    df = TEST_DF
    preds = classify_texts(df["text"], MODEL_DIR, probabilities=False)
    probas = classify_texts(df["text"], MODEL_DIR, probabilities=True)

    results_table = Table(title="Classification Results")
    results_table.add_column("Index", justify="right", style="cyan")
    results_table.add_column("Text", style="magenta")
    results_table.add_column("Probability (%)", justify="center", style="yellow")
    results_table.add_column("Prediction", justify="center", style="yellow")

    for (idx, text), proba, pred in zip(df["text"].items(), probas, preds):
        results_table.add_row(str(idx), text, f"{proba:.2f}%", str(pred))

    console.print(results_table)

    metrics = compute_metrics(preds, df["label"])
    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="center", style="yellow")

    for metric_name, metric_value in metrics.items():
        metrics_table.add_row(metric_name, f"{metric_value:.4f}")

    console.print(metrics_table)


if __name__ == "__main__":
    fire.Fire()
