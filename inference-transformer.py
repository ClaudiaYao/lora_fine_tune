"""Chat with a LoRA adapter trained by train-transformer.py (Transformers + PEFT).

Runs on CUDA, Apple Silicon (MPS) or CPU. The base model is read from the adapter's config.

Usage:
    python inference-transformer.py --prompt "How do I boil an egg?"
    python inference-transformer.py --prompt "Why is the sky blue?" --compare-base
    python inference-transformer.py            # interactive
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

import common


def load_model(adapter_dir):
    """Load the base model plus the saved adapter on the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=common.hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, dtype=dtype, token=common.hf_token())
    model = model.to(device).eval()
    print(f"Loaded {adapter_dir} on {device}")
    return model, tokenizer


def main():
    args = common.build_inference_parser(__doc__, run_name="transformers").parse_args()
    common.start_log(f"inference-{args.run_name}", args)
    model, tokenizer = load_model(args.adapter)
    common.answer_prompts(model, tokenizer, args)


if __name__ == "__main__":
    main()

#  python inference-transformer.py --prompts-file data/eval_prompts.txt --compare-base