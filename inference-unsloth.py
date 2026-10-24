"""Chat with a LoRA adapter trained by train-unsloth.py (Unsloth). Needs a CUDA GPU.

The base model is read from the adapter's config.

Usage:
    python inference-unsloth.py --prompt "How do I boil an egg?"
    python inference-unsloth.py --prompt "Why is the sky blue?" --compare-base
    python inference-unsloth.py            # interactive
"""

import unsloth  # must be imported before transformers / peft
from unsloth import FastLanguageModel

import torch

import common


def load_model(adapter_dir, max_seq_len):
    """Load the base model plus the saved adapter in Unsloth's fast inference mode."""
    assert torch.cuda.is_available(), "Unsloth needs a CUDA GPU. Use inference-transformer.py on a Mac."
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=False,
        token=common.hf_token(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    print(f"Loaded {adapter_dir}")
    return model, tokenizer


def main():
    args = common.build_inference_parser(__doc__, run_name="unsloth").parse_args()
    common.start_log(f"inference-{args.run_name}", args)
    model, tokenizer = load_model(args.adapter, args.max_seq_len)
    common.answer_prompts(model, tokenizer, args)


if __name__ == "__main__":
    main()
