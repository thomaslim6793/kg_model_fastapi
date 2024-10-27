import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
import json
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

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

# Function to partition text into sentences and further split long sentences
def partition_text(text, max_words_per_sentence=20):
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    partitioned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)  # Tokenize the sentence into words

        # If the sentence is too long, break it into smaller parts
        if len(words) > max_words_per_sentence:
            for i in range(0, len(words), max_words_per_sentence):
                small_sentence = ' '.join(words[i:i + max_words_per_sentence])
                partitioned_sentences.append(small_sentence)
        else:
            partitioned_sentences.append(sentence)

    return partitioned_sentences

# Function to invoke the SageMaker endpoint asynchronously
async def invoke_sagemaker(payload):
    try:
        # Invoke the SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            Body=json.dumps(payload),
            ContentType='application/json'
        )
        # Read and decode the response
        response_body = response['Body'].read().decode()
        result = json.loads(response_body)
        return json.loads(result[0]) if isinstance(result, list) else result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SageMaker invocation failed: {str(e)}")

@app.post("/generate")
async def generate_triplets(request: TextRequest):
    # Decode the input text if necessary
    if isinstance(request.text, bytes):
        input_text = request.text.decode('utf-8')
    elif isinstance(request.text, str):
        input_text = request.text
    else:
        return {"error": "Input text must be a string or bytes"}

    # Partition the input text into sentences, capping sentences at 20 words
    sentences = partition_text(input_text, max_words_per_sentence=20)

    all_triplets = []
    batch_size = 2  # Process 2 sentences at a time

    # Process sentences in batches of 2
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        
        # Prepare the payloads for the batch
        tasks = [invoke_sagemaker({"inputs": sentence, "gen_kwargs": request.gen_kwargs}) for sentence in batch]

        try:
            # Run both requests concurrently
            results = await asyncio.gather(*tasks)

            # Extract and combine triplets from both responses
            for result_dict in results:
                triplets = result_dict.get('triplets', [])
                all_triplets.extend(triplets)

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return {"error": f"Error during processing: {str(e)}"}

    return {"triplets": all_triplets}