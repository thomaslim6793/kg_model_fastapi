from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from post_process import post_process_REBEL
import torch  # Import torch to check for GPU availability
import os

from dotenv import load_dotenv
import os


# Define the class model
class TextRequest(BaseModel):
    text: str

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS from the frontend

load_dotenv()
origins = [os.getenv("FRONTEND_ORIGIN", "http://localhost:5175")]
print("Origins:", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Babelscape REBEL model at server startup
model = None
tokenizer = None
device = None  # Variable to store the device (CPU or GPU)

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device

    # Check if a GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer and move model to the appropriate device
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")


# Define the API endpoint to generate triplets
@app.post("/generate")
async def generate_triplets(request: TextRequest):
    global device
    text = request.text
    
    # Tokenize the input text and move inputs to the appropriate device (CPU or GPU)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate multiple sequences using beam search
    gen_kwargs = {
        "num_beams": 20,  # Try a higher number of beams to improve output quality
        "max_length": 256,  # Ensure the length is sufficient
        "length_penalty": 1,  # neither penalize nor reward length
        "num_return_sequences": 20  # Generate multiple sequences and pick the best
    }
    
    # Generate sequences on the device (CPU or GPU)
    generated_tokens = model.generate(**inputs, **gen_kwargs)

    # Decode all generated sequences
    decoded_results = [
        tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        for generated_token_ids in generated_tokens
    ]
    
    # Post-process each decoded result to extract triplets
    all_triplets = []
    for decoded_result in decoded_results:
        post_processed_result = post_process_REBEL(decoded_result)
        all_triplets.extend(post_processed_result)
    
    return {"triplets": all_triplets}