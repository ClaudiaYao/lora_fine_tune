"""Helpers shared by train-*.py and inference-*.py.

Everything that is identical between the Transformers+PEFT and Unsloth runs lives here
(arguments, data loading, tokenising, the Trainer loop, saving, generation). The scripts
themselves only contain what differs: how the model + LoRA is loaded.
"""

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA_FILE = ROOT / "data" / "train.jsonl"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"


def hf_token():
    return os.environ.get("HF_TOKEN")  # only needed for gated models such as Llama


# --------------------------------------------------------------------------- logging
class _Tee:
    """Write everything printed to stdout to a log file as well."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):  # isatty(), encoding, ... come from the real stream
        return getattr(self.stream, name)


def start_log(name, args=None):
    """Mirror stdout into logs/<name>.log (appended, with a timestamp header)."""
    LOGS_DIR.mkdir(exist_ok=True)
    path = LOGS_DIR / f"{name}.log"
    file = open(path, "a", encoding="utf-8", buffering=1)
    file.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} | {name} =====\n")
    sys.stdout = _Tee(sys.stdout, file)
    print(f"logging to {path}")
    if args is not None:
        print("args:", json.dumps(vars(args), default=str))
    return path


# --------------------------------------------------------------------------- arguments
def build_train_parser(description, run_name):
    p = argparse.ArgumentParser(description=description)
    # model & data
    p.add_argument("--run-name", default=run_name, help="label for outputs/ and results/")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-file", default=str(DEFAULT_DATA_FILE), help="chat-format JSONL from generate_data.py")
    p.add_argument("--max-samples", type=int, default=1000, help="training examples")
    p.add_argument("--eval-samples", type=int, default=100, help="held-out examples")
    p.add_argument("--max-seq-len", type=int, default=1024)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument(
        "--target-modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    # training
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size * grad-accum")
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def build_inference_parser(description, run_name):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run-name", default=run_name, help="label for logs/inference-<run-name>.*")
    p.add_argument("--adapter", default=str(ROOT / "outputs" / run_name / "adapter"), help="saved adapter directory")
    p.add_argument("--prompt", action="append", help="prompt to answer (repeatable)")
    p.add_argument("--prompts-file", help="text file with one prompt per line; use the same file for every run you compare")
    p.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up generation")
    p.add_argument("--compare-base", action="store_true", help="also answer with the adapter switched off")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy / deterministic")
    p.add_argument("--max-seq-len", type=int, default=1024)
    return p


# --------------------------------------------------------------------------- environment
def setup_environment(seed):
    """Check for a CUDA GPU, seed everything, reset peak-memory tracking."""
    assert torch.cuda.is_available(), "No CUDA GPU found. Run this on a GPU machine (e.g. RunPod)."
    use_bf16 = torch.cuda.is_bf16_supported()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__} | {gpu_name} | bf16={use_bf16}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    return use_bf16, gpu_name


# --------------------------------------------------------------------------- data
def load_examples(path, max_samples, eval_samples, seed):
    """Read {"messages": [user, assistant]} lines; return (train_examples, eval_examples)."""
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line)["messages"] for line in f if line.strip()]
    random.Random(seed).shuffle(examples)

    # A small file cannot fill both splits: hold out at most ~10% for evaluation.
    if len(examples) < max_samples + eval_samples:
        eval_samples = min(eval_samples, max(1, len(examples) // 10))
    train_examples = examples[: len(examples) - eval_samples][:max_samples]
    eval_examples = examples[len(examples) - eval_samples:]

    print(f"{Path(path).name}: {len(train_examples)} train / {len(eval_examples)} eval")
    print("\nExample:\n", train_examples[0][0]["content"][:300], "\n->", train_examples[0][1]["content"][:300])
    return train_examples, eval_examples

'''
max_seq_len in common.py (--max-seq-len, default 1024)
This covers both the user prompt and the assistant reply, plus the special tokens and the system prompt that the chat template adds.
tokenize_example cuts full_ids to this length, and full_ids holds the whole conversation. If prompt + reply is 600 tokens, only the first
512 are kept, and the end of the reply is lost. Training sequences are never longer than this.

--max-tokens in generate_data.py (default 1024)
This limits only the output, which is the assistant's answer from Bedrock. The prompt does not count toward it. It controls how long the
LLM may write when generating your training data, and it is what caused the cut-off sentences you saw.
'''

def tokenize_example(tokenizer, messages, max_seq_len):
    """Chat-template one example. Only the assistant reply contributes to the loss."""
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_seq_len]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:max_seq_len]
    if all(l == -100 for l in labels):
        return None  # prompt alone filled the window
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_dataset(tokenizer, examples, max_seq_len):
    rows = (tokenize_example(tokenizer, m, max_seq_len) for m in examples)
    return Dataset.from_list([r for r in rows if r])


# --------------------------------------------------------------------------- training
def train_model(model, tokenizer, train_ds, eval_ds, args, use_bf16):
    """Run the HF Trainer, measuring eval loss before/after, time and peak GPU memory."""
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    # transformers 5 removed warmup_ratio; a float warmup_steps below 1 means the same thing there.
    has_warmup_ratio = "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)}
    warmup = {"warmup_ratio": args.warmup_ratio} if has_warmup_ratio else {"warmup_steps": args.warmup_ratio}

    training_args = TrainingArguments(
        output_dir=str(ROOT / "outputs" / args.run_name / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        **warmup,
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    eval_loss_before = trainer.evaluate()["eval_loss"]
    print(f"eval loss before training: {eval_loss_before:.4f}")

    start = time.time()
    train_output = trainer.train()
    train_time = time.time() - start

    eval_loss_after = trainer.evaluate()["eval_loss"]
    peak_gpu_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"\neval loss after training:  {eval_loss_after:.4f}")
    print(f"train time {train_time:.0f}s | peak GPU memory {peak_gpu_gb:.1f} GB")

    train_tokens = sum(len(x) for x in train_ds["input_ids"])
    return {
        "train_sequences": len(train_ds),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_time_s": round(train_time, 1),
        "tokens_per_s": round(train_tokens * args.epochs / train_time),
        "peak_gpu_gb": round(peak_gpu_gb, 2),
        "final_train_loss": round(train_output.training_loss, 4),
        "eval_loss_before": round(eval_loss_before, 4),
        "eval_loss_after": round(eval_loss_after, 4),
        "perplexity_after": round(math.exp(eval_loss_after), 3),
        "_loss_history": [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h],
    }


# --------------------------------------------------------------------------- saving
def save_outputs(model, tokenizer, args, metrics, gpu_name):
    """Save the adapter, a results JSON, and a loss curve; print the comparison table."""
    output_dir = ROOT / "outputs" / args.run_name
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    history = metrics.pop("_loss_history")
    _save_loss_plot(history, args.run_name, output_dir / "loss.png")

    results = {
        "run": args.run_name, "model": args.model_id, "dataset": Path(args.data_file).name,
        "gpu": gpu_name, "epochs": args.epochs, "lora_r": args.lora_r,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        **metrics,
        "config": vars(args),            # lets compare-results.py check the runs are comparable
        "loss_history": history,         # [[step, train_loss], ...]
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.run_name}.json").write_text(json.dumps(results, indent=2))
    print(f"Adapter saved to {adapter_dir}\n")
    print_comparison()


def _save_loss_plot(history, run_name, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    plt.figure(figsize=(6, 3)); plt.plot(*zip(*history)); plt.xlabel("step"); plt.ylabel("train loss")
    plt.title(f"{run_name}: training loss"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(path); plt.close()


def print_comparison():
    """Side-by-side table of every run found in results/."""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    rows = ["run", "model", "dataset", "gpu", "load_time_s", "train_time_s", "tokens_per_s", "peak_gpu_gb",
            "final_train_loss", "eval_loss_before", "eval_loss_after", "perplexity_after", "trainable_params"]
    print(f"{'':20}" + "".join(f"{r['run']:>28}" for r in runs))
    for key in rows:
        print(f"{key:20}" + "".join(f"{str(r.get(key)):>28}" for r in runs))


# --------------------------------------------------------------------------- inference
def generate(model, tokenizer, prompt, max_new_tokens, temperature, use_adapter=True):
    """use_adapter=False answers with the plain base model (adapter switched off)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature)
    else:
        kwargs.update(do_sample=False)
    ctx = nullcontext() if use_adapter else model.disable_adapter()

    _sync(device)
    start = time.perf_counter()
    with ctx, torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    _sync(device)
    seconds = time.perf_counter() - start

    new_ids = output[0][inputs["input_ids"].shape[1]:]
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        "new_tokens": len(new_ids),
        "seconds": seconds,
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU work is asynchronous; wait so the timer is honest


def _device_name(model):
    device = next(model.parameters()).device
    return torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)


def log_generation(args, mode, prompt, result, device_name):
    """Append one timed generation to logs/inference-<run-name>.jsonl (read by compare-results.py)."""
    LOGS_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": args.run_name, "mode": mode, "device": device_name,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "prompt": prompt, "response": result["text"],
        "new_tokens": result["new_tokens"], "seconds": round(result["seconds"], 3),
        "tokens_per_s": round(result["new_tokens"] / result["seconds"], 2) if result["seconds"] else None,
    }
    with open(LOGS_DIR / f"inference-{args.run_name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_prompts(args):
    prompts = list(args.prompt or [])
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        prompts += [l.strip() for l in lines if l.strip()]
    return prompts


def answer_prompts(model, tokenizer, args):
    """Answer --prompt / --prompts-file prompts (timed and logged), or loop on stdin."""
    device_name = _device_name(model)

    if not args.no_warmup:
        # First CUDA call pays one-off costs (kernel setup, Unsloth patching); keep them out of the timings.
        generate(model, tokenizer, "Hello", 8, 0.0)

    def show(prompt):
        modes = [("base", False)] if args.compare_base else []
        for mode, use_adapter in modes + [("finetuned", True)]:
            result = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature, use_adapter)
            log_generation(args, mode, prompt, result, device_name)
            label = "FINE-TUNED:" if use_adapter else "BASE:      "
            print(f"\n{label} {result['text']}")
            print(f"            [{result['new_tokens']} tokens in {result['seconds']:.2f}s "
                  f"= {result['new_tokens'] / result['seconds']:.1f} tok/s on {device_name}]")
        print()

    prompts = collect_prompts(args)
    if prompts:
        for prompt in prompts:
            print("=" * 80, f"\nPROMPT: {prompt}")
            show(prompt)
        return

    print("Interactive mode. Empty line or 'exit' to quit.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            break
        if not prompt or prompt.lower() == "exit":
            break
        show(prompt)
