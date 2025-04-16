from transformers import pipeline, AutoTokenizer


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)


def generate_alert(input_data):

    prompt = f"""<|im_start|>user
{input_data['instruction']}
<|im_end|>
<|im_start|>assistant
</think>\n"""

    response = generator(
        prompt, max_new_tokens=150, temperature=0.6, top_p=0.9, do_sample=True
    )

    generated_text = response[0]["generated_text"].split("</think>\n")[-1].strip()
    return generated_text
