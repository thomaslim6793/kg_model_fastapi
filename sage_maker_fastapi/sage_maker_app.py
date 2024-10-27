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
origins = [os.getenv("FRONTEND_ORIGIN")]
print("Origins:", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-2')

# Replace with your actual endpoint name
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT")

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
    result = json.loads(response['Body'].read().decode())[0]  # Extract the first element, which is the actual result
    result_dict = json.loads(result)
    # Extract triplets
    triplets = result_dict.get('triplets', [])

    # Return the triplets to the frontend
    return {"triplets": triplets}