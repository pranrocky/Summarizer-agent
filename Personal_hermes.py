from fastapi import FastAPI, Request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()

# Load model ONCE at startup
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # Optional: improves accuracy
    bnb_4bit_quant_type="nf4",        # "nf4" is standard, "fp4" is another option
    bnb_4bit_compute_dtype="float16", # or "bfloat16" if your GPU supports it
)
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=64)
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {"response": response}
