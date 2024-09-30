import os
import json
import torch
from flask import Flask, request, jsonify
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# Load configuration from config.json
config_path = "config.json"

# Check if config.json exists and read the Hugging Face API token
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file '{config_path}' not found. Please create it and add your Hugging Face API token.")

with open(config_path, 'r') as f:
    config = json.load(f)

hf_api_token = config.get("hf_api_token")

if not hf_api_token:
    raise ValueError("Hugging Face API token not found in config.json. Please add it under 'hf_api_token' key.")

# Authenticate with Hugging Face
login(hf_api_token)

# Initialize Flask application
app = Flask(__name__)

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Check if CUDA (GPU) is available
is_cuda_available = torch.cuda.is_available()

# Configure model to use 8-bit quantization with GPU if available
quantization_config = BitsAndBytesConfig(
    load_in_8bit=is_cuda_available,  # Enable 8-bit loading only if CUDA is available
    llm_int8_threshold=6.0,
) if is_cuda_available else None

# Load the model with quantization and place it on the GPU
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,  # Use half precision to optimize memory usage
    low_cpu_mem_usage=True,
    device_map="auto" if is_cuda_available else {"": "cpu"}  # Map to GPU if available, otherwise CPU
)

# Set the model to evaluation mode to reduce overhead
model.eval()

# Define a Flask route to handle chatbot requests
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get the input prompt from the request
        data = request.json
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({"error": "Prompt is empty or missing"}), 400

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)

        # Generate a response using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the generated tokens into a response string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
