"""LoRA fine-tuning with Transformers + PEFT.

Flow: environment -> data -> model + LoRA -> tokenise -> train -> save.
Everything except load_model_with_lora() is shared with train-unsloth.py via common.py.

Usage:
    python train-transformer.py
    python train-transformer.py --data-file data/train.jsonl --epochs 3 --max-samples 500
"""

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import common


def load_model_with_lora(args, use_bf16):
    """Load the base model with AutoModelForCausalLM and attach LoRA with peft."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=common.hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16 if use_bf16 else torch.float32, token=common.hf_token()
    ).to("cuda")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = common.build_train_parser(__doc__, run_name="transformers").parse_args()
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
