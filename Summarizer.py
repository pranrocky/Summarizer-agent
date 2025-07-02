from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=quantization_config,
)

def summarize_user_input(user_text):
    system_prompt = (
        "<|im_start|>system\n"
        "You are an expert assistant. Your task is to extract ONLY the role(s) or resource(s) the user is seeking, and give a brief explanation why"
        "from the following list: ['mentor', 'founder', 'investor'].\n"
        "- Output strictly a JSON object with a 'needs' key, containing an array of zero or more of ONLY those keywords.\n"
        "- Do NOT include the user's own role, only what they are asking for or need.\n"
        "- After the JSON, provide a brief explanation (in plain text) of your reasoning for the selection. "
        "If none of the keywords are relevant, return an empty array and explain why.\n"
        "Example 1:\n"
        "User: I am a founder who just raised Series A and need guidance for scaling my startup.\n"
        "Output:\n"
        "{\"needs\": [\"mentor\"]}\n"
        "Explanation: The user is seeking guidance, which matches the role 'mentor'.\n"
        "Example 2:\n"
        "User: I am looking for someone to invest in my business idea.\n"
        "Output:\n"
        "{\"needs\": [\"investor\"]}\n"
        "Explanation: The user is seeking an investor.\n"
        "Example 3:\n"
        "User: I am a founder and want to meet other founders.\n"
        "Output:\n"
        "{\"needs\": [\"founder\"]}\n"
        "Explanation: The user wants to connect with other founders.\n"
        "Example 4:\n"
        "User: I am just browsing.\n"
        "Output:\n"
        "{\"needs\": []}\n"
        "Explanation: The user is not seeking any of the listed roles.\n"
        "<|im_end|>\n"
    )
    user_prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant"
    prompt = system_prompt + user_prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

# Example usage:
user_text = "I enjoy attending startup events and meeting new people in the ecosystem. Just here to expand my network and learn from others."
summary = summarize_user_input(user_text)
print(summary)
