import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NousResearch/Hermes-3-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # Efficient quantized loading
)

prompt = """<|im_start|>system
You are a helpful assistant who always explains questions with a real world example.<|im_end|>
<|im_start|>user
What is a CNN?<|im_end|>
<|im_start|>assistant
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=4096,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
