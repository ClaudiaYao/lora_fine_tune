import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = [("transformers", "#2a6fdb", "-"), ("unsloth", "#e8743b", "--")]

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
for name, color, style in RUNS:
    history = json.loads((HERE / f"{name}.json").read_text())["loss_history"]
    ax.plot([s for s, _ in history], [l for _, l in history],
            label=name, color=color, ls=style, lw=1.8, alpha=0.9)

ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Training loss (log)")
ax.set_title("LoRA fine-tune training loss: transformers vs unsloth\n"
             "Qwen2.5-1.5B-Instruct, 20 epochs, r=16")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
out = HERE / "loss_chart.png"
fig.savefig(out)
print(f"saved {out}")
