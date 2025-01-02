from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import json

# Initialize FastAPI
app = FastAPI()

# Model and tokenizer loading
MODEL_PATH = "llama3"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Check for GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Use DataParallel for multi-GPU utilization
device_count = torch.cuda.device_count()
if device_count > 1:
    print(f"Using {device_count} GPUs with DataParallel")
    model = torch.nn.DataParallel(model, device_ids=list(range(device_count)))  # Use all available GPUs

# Set up logging
logging.basicConfig(
    filename="api_benchmark.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Thread pool for concurrent requests
executor = ThreadPoolExecutor(max_workers=4)

class RequestPayload(BaseModel):
    requests: list
    num_threads: int

@app.post("/generate/")
async def generate_batch(request: RequestPayload):
    start_time = time.time()
    logging.info("Request Initialization")

    input_requests = request.requests
    num_threads = request.num_threads

    # Prepare a list to hold all futures
    futures = []

    # Loop through requests
    for i, req in enumerate(input_requests):
        question = req.get("question")
        context = req.get("context")

        # Tokenize the input texts
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to device (GPU/CPU)

        # Submit to thread pool for concurrent processing
        futures.append(executor.submit(generate_answer, inputs, i))

    # Wait for all tasks to finish and collect responses
    responses = [future.result() for future in futures]

    # Log processing time
    processing_time = time.time() - start_time
    logging.info(f"API Processing Time: {processing_time:.2f} seconds")

    # Structure responses with metadata
    detailed_responses = [{"index": i, "response": res} for i, res in enumerate(responses)]

    # Save to JSON log
    with open("api_responses.json", "a", encoding="utf-8") as json_log:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time": processing_time,
            "responses": detailed_responses
        }, json_log, ensure_ascii=False, indent=4)

    return {"responses": responses}

def generate_answer(inputs, index):
    try:
        # Generate batch inference using DataParallel
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=50, num_return_sequences=1, 
                top_k=50, top_p=0.95, temperature=0.7
            )

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logging.error(f"Error generating answer for index {index}: {str(e)}")
        return f"Error processing request {index}"
