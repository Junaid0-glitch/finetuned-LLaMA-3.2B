# 🧹🚨 Fine-Tuning LLaMA 3.2B with Unsloth on ShareGPT Data

This project demonstrates how to **fine-tune** the [Unsloth LLaMA 3.2B Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-instruct) model using the [`trl.SFTTrainer`](https://github.com/huggingface/trl) on ShareGPT-style conversational data from the `mlabonne/FineTome-100k` dataset.

## 🚀 Features

- ⚡ Lightweight & memory-efficient training with **4-bit quantization**
- 🧠 Trained using **instruction-tuned** ShareGPT-style dialogues
- 🧰 Built with `transformers`, `trl`, `unsloth`, and `datasets`
- 🧪 Supports **interactive inference** after training

---

## 🎞️ Requirements

```bash
pip install unsloth datasets trl accelerate bitsandbytes
```

---

## 💬 Inference Loop

```python
inference_model, inference_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)

while True:
    user_input = input("> ")
    if user_input.lower() in ["bye", "exit", "stop"]:
        break

    formated_prompt = inference_tokenizer.apply_chat_template([{
        "role": "user",
        "content": user_input
    }], tokenize=False)

    model_inputs = inference_tokenizer(formated_prompt, return_tensors="pt").to("cuda")

    generated_ids = inference_model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=inference_tokenizer.pad_token_id
    )

    response = inference_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
```

---

## 🔧 Load & Prepare Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

## 📚 Load & Preprocess Dataset

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt


dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)

dataset = dataset.map(
    lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples['conversations']
        ]
    },
    batched=True
)
```

---
![Screenshot 2025-06-01 213744](https://github.com/user-attachments/assets/5a5089f7-f4a6-460d-b177-da77c3532bfa)
