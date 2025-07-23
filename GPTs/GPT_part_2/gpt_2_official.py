from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("Hello, I'm a language model,")[0]["generated_text"])
