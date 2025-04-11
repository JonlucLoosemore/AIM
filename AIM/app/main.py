from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

app = FastAPI()

# Model configuration
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_dir = "/model"  # Directory to store the model in the container
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load or download model and tokenizer
try:
    print(f"Loading model from {model_dir} or downloading if not present...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=model_dir,
        local_files_only=False  # Download if not cached
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=model_dir,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        local_files_only=False  # Download if not cached
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model or tokenizer: {str(e)}")

# Request model
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/generate")
def generate(request: PromptRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))