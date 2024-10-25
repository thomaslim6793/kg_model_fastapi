import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

# Set up your S3 bucket location and model tarball
model_data = 's3://my-huggingface-models/rebel_model/model.tar.gz'

# Define the IAM role
role = 'arn:aws:iam::245804628218:role/service-role/AmazonSageMaker-ExecutionRole-20241023T182449'

# Initialize a HuggingFaceModel
huggingface_model = HuggingFaceModel(
    role=role,
    model_data=model_data,
    transformers_version='4.37.0',
    pytorch_version='2.1.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='code'
)

# Define serverless inference configuration
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072,  # Choose between 1024 and 6144 MB
    max_concurrency=2        # Maximum number of concurrent invocations
)

# Deploy the model using serverless inference
predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config
)

# You can now invoke the endpoint
response = predictor.predict({
    "inputs": "Barack Obama was born in Hawaii.",
    "gen_kwargs": {
        "num_beams": 20,
        "max_length": 256,
        "length_penalty": 1,
        "num_return_sequences": 20
    }
})

print(response)