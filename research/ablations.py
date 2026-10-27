"""Generate reproducible configs for the paper's Table 3 and new controls.

python -m research.ablations --base configs/base_en_de.yaml --out work/ablations --suite all
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Experiment:
    name: str
    source: str
    changes: dict
    category: str = "paper"


def paper_matrix() -> list[Experiment]:
    items = [Experiment("base", "Table 3 baseline", {})]
    def add(name, section, changes):
        items.append(Experiment(name, "Table 3 " + section, changes))
    for n in (1, 4, 16, 32):
        add(f"A_heads_{n}", "(A)", {"model.num_heads": n})
    for n in (16, 32):
        add(f"B_dk_{n}", "(B)", {"model.d_k": n})
    for n in (2, 4, 8):
        add(f"C_layers_{n}", "(C)", {"model.N": n})
    for n in (256, 1024):
        add(f"C_dmodel_{n}", "(C)", {"model.d_model": n})
    for n in (1024, 4096):
        add(f"C_dff_{n}", "(C)", {"model.d_ff": n})
    for v in (0.0, 0.2):
        add(f"D_dropout_{int(10*v)}", "(D)", {"model.dropout": v})
        add(f"D_smoothing_{int(10*v)}", "(D)", {"model.label_smoothing": v})
    add("E_learned_position", "(E)", {"model.pos_encoding": "learned"})
    add("big", "big", {"model.N": 6, "model.d_model": 1024,
        "model.d_ff": 4096, "model.num_heads": 16, "model.dropout": 0.3,
        "optim.max_steps": 300000})
    return items


def mechanism_matrix() -> list[Experiment]:
    settings = [
        ("unscaled_attention", "Eq. 1 without sqrt(d_k)", {"model.attention_kind": "unscaled_dot"}),
        ("no_position", "Sec. 3.5 zero positional signal", {"model.pos_encoding": "none"}),
        ("pre_norm", "Sec. 3.1 modern Pre-LN comparison", {"model.norm_style": "pre"}),
        ("no_final_norm", "Sec. 3.1 exact original Post-LN output", {"model.final_norm": False}),
        ("no_weight_tying", "Sec. 3.4 untied output", {"model.tie_softmax_weight": False}),
        ("unshared_embeddings", "Sec. 3.4 untied input embedding", {"model.share_embeddings": False}),
        ("gelu", "Sec. 3.3 activation comparison", {"model.activation": "gelu"}),
        ("attention_dropout_02", "Sec. 5.4 attention dropout", {"model.attn_dropout": 0.2}),
    ]
    return [Experiment(a, b, c, "mechanism") for a, b, c in settings]


def materialize(base: dict, experiment: Experiment, seed: int, output_dir: Path) -> dict:
    cfg = deepcopy(base)
    for dotted, value in experiment.changes.items():
        group, key = dotted.split(".", 1)
        if group not in ("model", "optim") or group not in cfg:
            raise ValueError(f"Unsupported override: {dotted}")
        cfg[group][key] = value
    # Original Post-LN encoder/decoder has no additional output LayerNorm.
    # Existing legacy repo defaults to an extra final LayerNorm for compatibility.
    if experiment.category == "paper":
        cfg.setdefault("model", {})["final_norm"] = False
    runtime = cfg.setdefault("runtime", {})
    runtime["seed"] = seed
    label = f"{experiment.name}-s{seed}"
    for field, subdir in [("log_dir", "logs"), ("tb_dir", "tensorboard"),
                          ("ckpt_dir", "checkpoints")]:
        runtime[field] = str(output_dir / subdir / label)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/base_en_de.yaml")
    p.add_argument("--out", default="work/ablations")
    p.add_argument("--suite", choices=("paper", "mechanism", "all"), default="paper")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    matrix = ([] if args.suite == "mechanism" else paper_matrix())
    if args.suite in ("all", "mechanism"):
        matrix.extend(mechanism_matrix())
    output = Path(args.out)
    folder = output / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for item in matrix:
        for seed in args.seeds:
            cfg = materialize(base, item, seed, output)
            filename = folder / f"{item.name}-s{seed}.yaml"
            filename.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")
            index.append({"experiment": item.name, "seed": seed, "source": item.source,
                          "category": item.category, "changes": item.changes,
                          "config": str(filename), "status": "not_run"})
    (output / "plan.json").write_text(json.dumps(index, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"{len(index)} configurations generated. No training was executed.")


if __name__ == "__main__":
    main()
