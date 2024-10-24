from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
import json
from dotenv import load_dotenv

# Define the request model to accept both text and gen_kwargs
class TextRequest(BaseModel):
    text: str
    gen_kwargs: dict

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

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-2')

# Replace with your actual endpoint name
SAGEMAKER_ENDPOINT = "huggingface-pytorch-inference-2024-10-24-08-57-29-896"

@app.post("/generate")
async def generate_triplets(request: TextRequest):
    # Prepare the input for SageMaker
    payload = {
        "inputs": request.text,
        "gen_kwargs": request.gen_kwargs
    }

    # Invoke the SageMaker endpoint
    response = sagemaker_client.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT,
        Body=json.dumps(payload),
        ContentType='application/json'
    )

    # Read and decode the response
    response_body = response['Body'].read().decode()

    # The response body might be a JSON string within a list
    # Let's parse it appropriately

    # First, try to load the response body as JSON
    try:
        # Attempt to parse the response body directly
        result = json.loads(response_body)
    except json.JSONDecodeError:
        # If it's a string within a list, parse accordingly
        result_list = json.loads(response_body)
        if isinstance(result_list, list) and len(result_list) > 0:
            inner_json_str = result_list[0]
            result = json.loads(inner_json_str)
        else:
            # Handle error if the response format is unexpected
            return {"error": "Unexpected response format from SageMaker endpoint"}

    # Now, 'result' should be a dictionary with a 'triplets' key
    triplets = result.get('triplets', [])

    # Return the triplets to the frontend
    return {"triplets": triplets}