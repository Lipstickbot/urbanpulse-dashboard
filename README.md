# urbanpulse-dashboard

## Continue training your own model

If you already have these files:
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`

then your LoRA adapter is already trained once.

To continue training it on your UrbanPulse dataset:

```bash
pip install -r requirements.txt
python continue_finetune_from_adapter.py
```

Optional environment variables:

```bash
set ADAPTER_DIR=C:\Users\vovot\Downloads\urbanpulse-qwen-lora\content\urbanpulse-qwen-lora
set DATASET_PATH=C:\Users\vovot\OneDrive\Документы\Playground\training_data\urbanpulse_multilang_qlora.jsonl
set OUTPUT_DIR=C:\Users\vovot\OneDrive\Документы\Playground\outputs\urbanpulse-qwen-lora-continued
python continue_finetune_from_adapter.py
```

Recommended:
- run this on Google Colab GPU
- or a local CUDA machine
