from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Загрузка дообученной модели
# model_path = "gpt2"  # Папка, в которую ты сохранял
model_path = "ru-gm"  # Папка, в которую ты сохранял
# model_path = "sberbank-ai/rugpt3medium_based_on_gpt2"  # Папка, в которую ты сохранял

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Создаём пайплайн генерации текста
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Диалог: подаём реплику игрока
prompt = "[Игрок]: Я открываю дверь в подземелье.\n[Мастер]:"
# prompt = "Hello, Sailor!\n"

output = generator(prompt, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.9)[0]["generated_text"]

print(output)
