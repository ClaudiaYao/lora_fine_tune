"""Generate persona-styled training data for SLM LoRA fine-tuning via AWS Bedrock.

Pipeline: take plain instruction prompts -> ask a Bedrock LLM to answer each one in
the target persona's voice -> save (prompt, styled answer) pairs as chat-format JSONL.

Usage:
    python generate_data.py --persona shakespeare --num-samples 200
    python generate_data.py --persona yoda --prompts-file my_prompts.txt
    python generate_data.py --list-models
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PERSONAS = {
    "shakespeare": (
        "You are William Shakespeare. Answer in Early Modern English with "
        "Elizabethan vocabulary (thou, thee, hath, doth), rich metaphor, and an "
        "occasional iambic rhythm. Keep the answer factually correct and helpful."
    ),
    # "yoda": (
    #     "You are Yoda from Star Wars. Speak in his inverted syntax "
    #     "(object-subject-verb), with wise, cryptic, calm phrasing. Keep the "
    #     "answer factually correct and helpful."
    # ),
    # "sherlock": (
    #     "You are Sherlock Holmes. Answer with sharp deductive reasoning, "
    #     "Victorian diction, and a hint of arrogance. Keep the answer factually "
    #     "correct and helpful."
    # ),
}

DEFAULT_DATASET = "databricks/databricks-dolly-15k"


def make_client():
    load_dotenv()
    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": os.getenv("AWS_REGION", ""),
        "config": Config(retries={"max_attempts": 2, "mode": "adaptive"}),
    }
    # Fall back to the default AWS credential chain if .env values are blank.
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        # if os.getenv("AWS_SESSION_TOKEN"):
        #     kwargs["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return boto3.client(**kwargs)


# def list_models():
#     load_dotenv()
#     bedrock = boto3.client(
#         "bedrock",
#         region_name=os.getenv("AWS_REGION", "us-east-1"),
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
#         # aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
#     )
#     for m in bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]:
#         print(f"{m['modelId']:70s} {m['providerName']}")


def load_used_prompts(path):
    """Prompts already present in the output file (the user turn of each record)."""
    used = set()
    if not path.exists():
        return used
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                used.add(json.loads(line)["messages"][0]["content"].strip())
            except (json.JSONDecodeError, KeyError, IndexError):
                continue  # ignore a malformed line, e.g. one cut off by an interrupted run
    return used


def load_prompts(args, used):
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text().splitlines()
        prompts = [l.strip() for l in lines if l.strip()]
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split="train")
        # Prefer rows without extra context so the prompt stands alone.
        col = "instruction" if "instruction" in ds.column_names else ds.column_names[0]
        prompts = [
            r[col]
            for r in ds
            if not r.get("context") and not r.get("input")
        ]
    random.seed(args.seed)
    random.shuffle(prompts)

    # Skip prompts already in the output file (and duplicates within this batch), then
    # take the next num_samples, so --num-samples always means "new examples".
    fresh, seen = [], set(used)
    for pr in prompts:
        pr = pr.strip()
        if pr in seen:
            continue
        seen.add(pr)
        fresh.append(pr)
        if len(fresh) >= args.num_samples:
            break
    return fresh


def generate_one(client, model_id, system_prompt, prompt, max_tokens, temperature):
    for attempt in range(5):
        try:
            print("prompt from dataset:", prompt)
            resp = client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            if resp.get("stopReason") == "max_tokens":
                # Truncated mid-sentence: drop it rather than train on a cut-off ending.
                print("dropped (hit max_tokens):", prompt[:60])
                return None
            return resp["output"]["message"]["content"][0]["text"].strip()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "ModelTimeoutException"):
                time.sleep(2**attempt)
                continue
            raise
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persona", default="shakespeare", help=f"one of {list(PERSONAS)} or custom text")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--prompts-file", help="text file with one prompt per line")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset to draw prompts from")
    p.add_argument("--output", default="data/train.jsonl")
    p.add_argument("--model-id", help="overrides BEDROCK_MODEL_ID from .env")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # p.add_argument("--list-models", action="store_true", help="list Bedrock text models and exit")
    args = p.parse_args()

    # if args.list_models:
    #     list_models()
    #     return

    client = make_client()
    model_id = args.model_id or os.getenv("BEDROCK_MODEL_ID")
    if not model_id:
        raise SystemExit("Set BEDROCK_MODEL_ID in .env or pass --model-id")

    system_prompt = PERSONAS.get(args.persona, args.persona) + (
        " Keep the answer concise (under 200 words) and always finish with a "
        "complete sentence."
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    used = load_used_prompts(out_path)
    prompts = load_prompts(args, used)
    print(f"{len(used)} prompts already in {out_path}; "
          f"generating {len(prompts)} new samples with {model_id} (persona: {args.persona})")

    # Append to the existing file; make sure the last line is terminated first.
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
        if needs_newline:
            with out_path.open("a", encoding="utf-8") as f:
                f.write("\n")

    done = 0
    with ThreadPoolExecutor(args.workers) as pool, out_path.open("a", encoding="utf-8") as f:
        futures = {
            pool.submit(
                generate_one, client, model_id, system_prompt, pr,
                args.max_tokens, args.temperature,
            ): pr
            for pr in prompts
        }
        for fut in as_completed(futures):
            prompt = futures[fut]
            try:
                answer = fut.result()
            except Exception as e:
                print(f"skip ({type(e).__name__}): {prompt[:60]}")
                continue
            if not answer:
                continue
            record = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # keep finished examples even if the run is interrupted
            done += 1
            if done % 10 == 0:
                print(f"{done}/{len(prompts)}")

    print(f"Appended {done} examples to {out_path} ({len(used) + done} total)")


if __name__ == "__main__":
    main()
