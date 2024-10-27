import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from post_process import post_process_REBEL  # Adjust the import if necessary

def model_fn(model_dir, context=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer, device

def predict_fn(input_data, model_tokenizer_device):
    model, tokenizer, device = model_tokenizer_device
    inputs = input_data['inputs']

    gen_kwargs = input_data.get('gen_kwargs', {
        "num_beams": 20,
        "max_length": 256,
        "length_penalty": 1,
        "num_return_sequences": 20
    })

    tokenized_inputs = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    generated_tokens = model.generate(**tokenized_inputs, **gen_kwargs)

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

def input_fn(request_body, request_content_type='application/json'):
    if request_content_type == 'application/json':
        if isinstance(request_body, bytes):
            decoded_body = request_body.decode('utf-8')
        elif isinstance(request_body, str):
            decoded_body = request_body
        else:
            raise ValueError(f"Unsupported request body type: {type(request_body)}")
        
        return json.loads(decoded_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction_output, accept='application/json'):
    return json.dumps(prediction_output), accept