"""
Backbone model loading with optional LoRA for baselines.

Supports:
1. Frozen base (no adaptation)
2. Dense multi-task LoRA on FFN projections (W^u, W^g, W^d)
3. Task-specific LoRA (one adapter per task)
4. Task-family LoRA (one adapter per family)

Paper Section 4.5: "The primary LoRA baseline applies updates only to
W^u, W^g, and W^d so that it has the same target location as TaskMap."
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def load_backbone(model_name: str = "Qwen/Qwen2.5-1.5B", dtype: str = "bfloat16"):
    """Load a frozen pretrained causal LM."""
    torch_dtype = getattr(torch, dtype, torch.bfloat16)

    attn_impl = "eager"
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def add_lora(model, rank: int = 16, alpha: int = 32,
             target_modules: list = None):
    """Add LoRA adapters to FFN projections."""
    if target_modules is None:
        target_modules = ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    print("Loading Qwen2.5-1.5B...")
    model, tokenizer = load_backbone("Qwen/Qwen2.5-1.5B")
    trainable, total = count_parameters(model)
    print(f"Frozen base: {trainable:,} trainable / {total:,} total")

    print("\nAdding LoRA (r=16)...")
    model = add_lora(model, rank=16, alpha=32)
    trainable, total = count_parameters(model)
    print(f"With LoRA: {trainable:,} trainable / {total:,} total")
