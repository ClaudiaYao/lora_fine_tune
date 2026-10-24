"""Compare the Transformers+PEFT run with the Unsloth run: training results and inference speed.

Reads
  results/<run>.json            written by train-*.py   (metrics, config, loss history)
  logs/inference-<run>.jsonl    written by inference-*.py (one timed generation per line)
and writes results/comparison.md (plus results/loss_comparison.png if matplotlib is installed).

Usage:
    python compare-results.py
    python compare-results.py --runs transformers unsloth
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# (key, label, lower_is_better or None if there is no "better")
TRAIN_METRICS = [
    ("load_time_s", "Model + LoRA load time (s)", True),
    ("train_time_s", "Train time (s)", True),
    ("tokens_per_s", "Train throughput (tokens/s)", False),
    ("peak_gpu_gb", "Peak GPU memory (GB)", True),
    ("trainable_params", "Trainable parameters", None),
    ("train_sequences", "Train sequences", None),
    ("final_train_loss", "Final train loss", True),
    ("eval_loss_before", "Eval loss before", None),
    ("eval_loss_after", "Eval loss after", True),
    ("perplexity_after", "Perplexity after", True),
]
# Settings that must match for a speed/quality comparison to be fair.
CONFIG_KEYS = ["model_id", "data_file", "max_samples", "eval_samples", "max_seq_len", "lora_r", "lora_alpha",
               "lora_dropout", "target_modules", "epochs", "learning_rate", "batch_size", "grad_accum",
               "lr_scheduler", "warmup_ratio", "gradient_checkpointing", "seed"]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:,.4g}" if isinstance(v, float) else f"{v:,}"
    return str(v)


def ratio(v, base):
    if isinstance(v, (int, float)) and isinstance(base, (int, float)) and base:
        return f"{v / base:.2f}x"
    return ""


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def load_results(runs):
    results = {}
    for run in runs:
        path = RESULTS_DIR / f"{run}.json"
        if path.exists():
            results[run] = json.loads(path.read_text())
        else:
            print(f"warning: {path} not found (run train-{run}.py first)")
    return results


def training_section(results):
    runs = list(results)
    base = runs[0]
    headers = ["metric"] + runs + [f"{r} / {base}" for r in runs[1:]]
    rows = [[label] + [fmt(results[r].get(key)) for r in runs]
            + [ratio(results[r].get(key), results[base].get(key)) for r in runs[1:]]
            for key, label, _ in TRAIN_METRICS]
    info = [[k] + [fmt(results[r].get(k)) for r in runs] + [""] * (len(runs) - 1)
            for k in ("model", "dataset", "gpu")]
    out = ["## Training", md_table(headers, info + rows), ""]

    if len(runs) > 1:
        out.append("**Winners**")
        for key, label, lower_better in TRAIN_METRICS:
            if lower_better is None:
                continue
            vals = {r: results[r].get(key) for r in runs if isinstance(results[r].get(key), (int, float))}
            if len(vals) < 2:
                continue
            best = (min if lower_better else max)(vals, key=vals.get)
            worst = (max if lower_better else min)(vals, key=vals.get)
            if vals[best] == vals[worst]:
                continue  # a tie is not a winner
            gap = vals[worst] / vals[best] if lower_better else vals[best] / vals[worst]
            out.append(f"- {label}: **{best}** ({fmt(vals[best])} vs {fmt(vals[worst])}, {gap:.2f}x better)")
        out.append("")

    diffs = []
    for key in CONFIG_KEYS:
        values = {r: results[r].get("config", {}).get(key) for r in runs}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            diffs.append(f"- `{key}`: " + ", ".join(f"{r}={fmt(v)}" for r, v in values.items()))
    if diffs:
        out += ["**Warning: the runs used different settings, so the comparison is not apples to apples:**", *diffs, ""]
    return "\n".join(out)


def load_inference(run):
    """Latest record per (mode, prompt) from logs/inference-<run>.jsonl."""
    path = LOGS_DIR / f"inference-{run}.jsonl"
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[(rec["mode"], rec["prompt"])] = rec
    return latest


def inference_section(runs):
    data = {r: load_inference(r) for r in runs}
    data = {r: d for r, d in data.items() if d}
    if not data:
        return "## Inference\nNo inference logs found. Run inference-*.py with the same `--prompts-file` for each run.\n"
    names = list(data)
    out = ["## Inference"]

    rows = []
    for mode, label in (("finetuned", "fine-tuned"), ("base", "base (adapter off)")):
        common_prompts = set.intersection(*[{p for (m, p) in d if m == mode} for d in data.values()])
        if not common_prompts:
            continue
        stats = {}
        for r, d in data.items():
            recs = [d[(mode, p)] for p in common_prompts]
            stats[r] = {
                "device": recs[0].get("device"),
                "latency": statistics.mean(x["seconds"] for x in recs),
                "tps": statistics.mean(x["tokens_per_s"] for x in recs if x["tokens_per_s"]),
                "tokens": statistics.mean(x["new_tokens"] for x in recs),
            }
        base = names[0]
        for key, name in (("latency", "Mean latency per answer (s)"), ("tps", "Mean generation speed (tokens/s)"),
                          ("tokens", "Mean new tokens")):
            rows.append([f"{label}: {name}"] + [fmt(round(stats[r][key], 3)) for r in names]
                        + [ratio(stats[r][key], stats[base][key]) for r in names[1:]])
        rows.append([f"{label}: prompts compared"] + [str(len(common_prompts))] * len(names) + [""] * (len(names) - 1))
        devices = {stats[r]["device"] for r in names}
        if len(devices) > 1:
            out.append(f"**Warning: different devices ({', '.join(map(str, devices))}); speeds are not comparable.**\n")

    headers = ["metric"] + names + [f"{r} / {names[0]}" for r in names[1:]]
    out += [md_table(headers, rows), ""]

    # Side-by-side answers to eyeball style quality.
    prompts = sorted(set.intersection(*[{p for (m, p) in d if m == "finetuned"} for d in data.values()]))[:3]
    if prompts:
        out.append("### Sample answers (fine-tuned)")
        for p in prompts:
            out.append(f"**Prompt:** {p}\n")
            for r in names:
                out.append(f"- *{r}:* {data[r][('finetuned', p)]['response']}")
            out.append("")
    return "\n".join(out)


def plot_losses(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    plt.figure(figsize=(7, 3.5))
    plotted = False
    for run, res in results.items():
        history = res.get("loss_history")
        if history:
            plt.plot(*zip(*history), label=run)
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel("step"); plt.ylabel("train loss"); plt.title("Training loss"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["transformers", "unsloth"], help="first run is the baseline")
    args = p.parse_args()

    results = load_results(args.runs)
    if not results:
        raise SystemExit("No results found. Run train-transformer.py / train-unsloth.py first.")

    sections = ["# Transformers+PEFT vs Unsloth", training_section(results), inference_section(list(results))]
    if plot_losses(results, RESULTS_DIR / "loss_comparison.png"):
        sections.append("Loss curves: `results/loss_comparison.png`\n")
    report = "\n".join(sections)

    (RESULTS_DIR / "comparison.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {RESULTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
