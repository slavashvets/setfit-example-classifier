# setfit-example-classifier

Example SetFit classification of job-vacancy postings using a small **synthetic** dataset.

<p align="center">
  <img src="./docs/demo.svg">
</p>

## Project goal

Demonstrate how to fine-tune a Sentence-Transformers encoder with the [SetFit](https://github.com/huggingface/setfit) framework to distinguish “interesting” tech jobs (senior, remote, well-paid) from everything else.

## Quick start

[Taskfile](https://taskfile.dev/installation/) handles task management, and [uv](https://docs.astral.sh/uv/getting-started/installation/) manages the virtual environment and dependencies automatically (including python executable).

```bash
# Train the model and save the best checkpoint under models/
task train

# Evaluate on the held-out test set
task test

# List all available tasks
task --list
```

## Dataset

Sixty short, hand-crafted job-ad snippets labelled
**1 = interesting** (e.g. “Senior Rust Developer — Remote — \$150k”)
**0 = uninteresting** (e.g. “Call-centre agent — rotating shifts”).
See the three CSVs under `data/`.

## Model

- Base encoder: **all-MiniLM-L6-v2**
- Contrastive fine-tuning + linear classification head via **SetFit**
- CPU-only by default — change `device` in `main.py` if you have a GPU.

## License

Released under the **MIT License** – see `LICENSE` for details.
