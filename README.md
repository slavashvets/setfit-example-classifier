# setfit-example-classifier

Example SetFit classification of job-vacancy postings using a small **synthetic** dataset.

<p align="center">
  <img src="./docs/demo.svg">
</p>

## Project goal

Demonstrate how to fine-tune a Sentence-Transformers encoder with the [SetFit](https://github.com/huggingface/setfit) framework to distinguish “interesting” tech jobs (senior, remote, well-paid) from everything else.

## Quick start

`uv` handles the virtual environment and dependencies automatically:

```bash
# Train the model and save the best checkpoint under models/
uv run main.py train

# Evaluate on the held-out test set
uv run main.py test
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
