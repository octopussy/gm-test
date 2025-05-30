from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

import torch
print("DEVICE:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# === 1. Настройки ===
model_name = "gpt2"
max_length = 512
dataset_path = "dataset_1000.jsonl"
output_dir = "./gpt2-gm"

# === 2. Загрузка модели и токенизатора ===
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 не имеет pad_token по умолчанию
# if tokenizer.pad_token is None:
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # если pad_token добавился

# === 3. Загрузка и токенизация датасета ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

def tokenize(batch):
    prompts = [f"{inp} {out}" for inp, out in zip(batch["input"], batch["output"])]
    return tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# === 4. Настройка обучения ===
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    prediction_loss_only=True,
    fp16=False  # на Mac отключаем, на NVIDIA можно включить
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 — это не masked language model
)

# === 5. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# === 6. Обучение ===
trainer.train()

# === 7. Сохранение ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
