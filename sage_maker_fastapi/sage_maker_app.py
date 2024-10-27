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
    allow_origins=origins,
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
    # Check if request.text is in byte format or string format
    if isinstance(request.text, bytes):
        print("Input is in byte format, converting to string.")
        input_text = request.text.decode('utf-8')  # Decode bytes to string
    elif isinstance(request.text, str):
        print("Input is already in string format.")
        input_text = request.text  # Keep it as a string
    else:
        print(f"Input is in an unexpected format: {type(request.text)}")
        return {"error": "Input text must be a string or bytes"}

    # Print the text format and content for debugging
    print(f"Input text after conversion: {input_text}")

    # Prepare the input for SageMaker
    payload = {
        "inputs": input_text,  # Ensure 'inputs' is always a string
        "gen_kwargs": request.gen_kwargs
    }

    try:
        # Invoke the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            Body=json.dumps(payload),  # Ensure the payload is JSON
            ContentType='application/json'
        )
    except Exception as e:
        print(f"SageMaker invocation failed: {str(e)}")
        return {"error": f"SageMaker invocation failed: {str(e)}"}

    # Read and decode the response
    try:
        response_body = response['Body'].read().decode()
        result = json.loads(response_body)
        if isinstance(result, list) and result:
            result_dict = json.loads(result[0])
        else:
            result_dict = result
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to decode SageMaker response: {str(e)}")
        return {"error": f"Failed to decode SageMaker response: {str(e)}"}

    # Extract triplets
    triplets = result_dict.get('triplets', [])
    return {"triplets": triplets}