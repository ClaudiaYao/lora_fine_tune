"""LoRA fine-tuning with Unsloth.

Flow: environment -> data -> model + LoRA -> tokenise -> train -> save.
Everything except load_model_with_lora() is shared with train-transformer.py via common.py.

Usage:
    python train-unsloth.py
    python train-unsloth.py --data-file data/train.jsonl --epochs 3 --max-samples 500
"""

import unsloth  # must be imported before transformers / peft
from unsloth import FastLanguageModel

import time

import torch

import common


def load_model_with_lora(args, use_bf16):
    """Load the base model with FastLanguageModel and attach LoRA with Unsloth's kernels."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        load_in_4bit=False,  # bf16 weights, same as train-transformer.py
        token=common.hf_token(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules, bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
    )
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = common.build_train_parser(__doc__, run_name="unsloth").parse_args()
    common.start_log(f"train-{args.run_name}", args)

    use_bf16, gpu_name = common.setup_environment(args.seed)
    train_examples, eval_examples = common.load_examples(
        args.data_file, args.max_samples, args.eval_samples, args.seed
    )
    load_start = time.time()
    model, tokenizer = load_model_with_lora(args, use_bf16)
    load_time = time.time() - load_start

    train_ds = common.build_dataset(tokenizer, train_examples, args.max_seq_len)
    eval_ds = common.build_dataset(tokenizer, eval_examples, args.max_seq_len)
    print(f"{len(train_ds)} train / {len(eval_ds)} eval sequences")

    metrics = common.train_model(model, tokenizer, train_ds, eval_ds, args, use_bf16)
    metrics["load_time_s"] = round(load_time, 1)
    common.save_outputs(model, tokenizer, args, metrics, gpu_name)


if __name__ == "__main__":
    main()
