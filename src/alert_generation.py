from transformers import pipeline, AutoTokenizer

def generate_alert(input_data):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

    prompt = f"""<|im_start|>user
        {input_data['instruction']}
        <|im_end|>
        <|im_start|>assistant
        </think>\n
        """

    response = generator(
        prompt,
        max_new_tokens=150,
        temperature=0.6,
        top_p=0.9,
        do_sample=True
    )
    
    # Extraer y formatear alerta
    generated_text = response[0]['generated_text'].split("</think>\n")[-1].strip()
    return generated_text


if __name__ == "__main__":
    input_data = {
        "instruction": '''Generate a reputation alert in English using this format:
        "REPUTATION ALERT: [MAIN_ENTITY] - [SENTIMENT]. Summary: [CONCISE_TEXT]"

        Input data:
        - Original text: "CEO criticized for controversial remarks during press conference"
        - Image description: "Angry crowd protesting outside company headquarters"
        - Detected entities: [{'entity': 'CEO', 'type': 'PERSON'}, {'entity': 'Company', 'type': 'ORG'}]
        - Overall sentiment: negative'''
    }

    alert = generate_alert(input_data)
    print(alert)
