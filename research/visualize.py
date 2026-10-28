"""Offline curves and attention heatmaps (matplotlib); live scalars use TensorBoard.

python -m research.visualize work/ablations/logs/* --output work/plots
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_records(directory: Path) -> list[dict]:
    path = directory / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plot_runs(run_dirs: list[Path], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    runs = {p.name: load_records(p) for p in run_dirs
            if (p / "metrics.jsonl").exists()}
    produced = []
    for split, key in (("train", "loss"), ("train", "lr"),
                       ("train", "tokens_per_sec"), ("validation", "nll"),
                       ("validation", "ppl"), ("test", "bleu")):
        fig, ax = plt.subplots(figsize=(9, 5))
        present = False
        for name, records in runs.items():
            values = [(row["step"], row["metrics"][key]) for row in records
                      if row.get("split") == split and key in row.get("metrics", {})]
            if values:
                x, y = zip(*values)
                ax.plot(x, y, label=name)
                present = True
        if present:
            ax.set(xlabel="Optimizer step", ylabel=key, title=f"{split} / {key}")
            ax.legend(fontsize="small")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path = output / f"{split}_{key}.png"
            fig.savefig(path, dpi=160)
            produced.append(path)
        plt.close(fig)
    return produced


def capture_attention(model, enabled: bool = True) -> None:
    from models.transformer import MultiHeadAttention
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.capture_attention = enabled


def attention_heatmaps(model, output: Path, limit: int = 3) -> list[Path]:
    """After eval forward pass, draw first head for up to `limit` attention modules."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models.transformer import MultiHeadAttention
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, module in model.named_modules():
        if not isinstance(module, MultiHeadAttention) or module.last_attention is None:
            continue
        if len(files) >= limit:
            break
        scores = module.last_attention[0, 0].float().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        picture = ax.imshow(scores, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set(xlabel="Key index", ylabel="Query index", title=f"{name} / head 0")
        fig.colorbar(picture, ax=ax)
        fig.tight_layout()
        path = output / f"attention_{len(files):02d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(path)
    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("work/plots"))
    args = p.parse_args()
    for path in plot_runs(args.runs, args.output):
        print(path)


if __name__ == "__main__":
    main()
