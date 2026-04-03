"""
QLoRA fine-tuning script for UrbanPulse Smart City multilingual assistant.

Recommended environment:
    Google Colab T4 / local CUDA GPU

Example:
    python train_qlora_unsloth.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = str(BASE_DIR / "training_data" / "urbanpulse_multilang_qlora.jsonl")
OUTPUT_DIR = str(BASE_DIR / "outputs" / "urbanpulse-qwen2.5-1.5b-lora")
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024


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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",
            max_seq_length=MAX_SEQ_LENGTH,
            save_strategy="epoch",
        ),
        dataset_text_field="text",
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
