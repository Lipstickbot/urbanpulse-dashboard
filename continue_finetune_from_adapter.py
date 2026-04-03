"""
Continue fine-tuning an existing UrbanPulse LoRA adapter.

Use this when you already have:
- adapter_model.safetensors
- adapter_config.json
- tokenizer files

Example:
    python continue_finetune_from_adapter.py

Recommended:
- Google Colab with GPU
- or a local CUDA machine

Environment variables you can set:
    ADAPTER_DIR=C:/Users/vovot/Downloads/urbanpulse-qwen-lora/content/urbanpulse-qwen-lora
    DATASET_PATH=C:/path/to/urbanpulse_multilang_qlora.jsonl
    OUTPUT_DIR=C:/path/to/urbanpulse-qwen-lora-continued
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = Path(r"C:\Users\vovot\Downloads\urbanpulse-qwen-lora\content\urbanpulse-qwen-lora")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR)))
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(BASE_DIR / "training_data" / "urbanpulse_multilang_qlora.jsonl")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs" / "urbanpulse-qwen-lora-continued")))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
EPOCHS = float(os.getenv("TRAIN_EPOCHS", "1.0"))
LEARNING_RATE = float(os.getenv("TRAIN_LR", "1e-4"))


def format_sample(example: dict) -> dict:
    messages = example["messages"]
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]
    assistant_text = messages[2]["content"]
    text = (
        f"<|system|>\n{system_text}\n"
        f"<|user|>\n{user_text}\n"
        f"<|assistant|>\n{assistant_text}"
    )
    return {"text": text}


def main() -> None:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ADAPTER_DIR),
        is_trainable=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)

    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            max_seq_length=MAX_SEQ_LENGTH,
            fp16=torch.cuda.is_available(),
            bf16=False,
        ),
        dataset_text_field="text",
    )

    trainer.train()
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Continued adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
