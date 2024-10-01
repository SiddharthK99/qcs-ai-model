import os
import json
import torch
import logging
from flask import Flask, request, jsonify
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from config.json
config_path = "config.json"

# Check if config.json exists and read the Hugging Face API token
if not os.path.exists(config_path):
    logger.error(f"Configuration file '{config_path}' not found.")
    raise FileNotFoundError(f"Configuration file '{config_path}' not found. Please create it and add your Hugging Face API token.")

with open(config_path, 'r') as f:
    config = json.load(f)

hf_api_token = config.get("hf_api_token")

if not hf_api_token:
    logger.error("Hugging Face API token not found in config.json.")
    raise ValueError("Hugging Face API token not found in config.json. Please add it under 'hf_api_token' key.")

# Authenticate with Hugging Face
login(token=hf_api_token)
logger.info("Authenticated with Hugging Face.")

# Initialize Flask application
app = Flask(__name__)

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.1-70B"
logger.info(f"Loading tokenizer for model '{model_name}'.")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Check if CUDA (GPU) is available
is_cuda_available = torch.cuda.is_available()
device = "cuda" if is_cuda_available else "cpu"
logger.info(f"CUDA available: {is_cuda_available}. Using device: {device}.")

# Configure model to use 8-bit quantization with GPU if available
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Always use 8-bit for memory efficiency
    llm_int8_threshold=6.0,
) if is_cuda_available else None

# Load the model with quantization and place it on the GPU or CPU
logger.info(f"Loading model '{model_name}' with device map: {device}.")
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.float16 if is_cuda_available else torch.float32,  # Use appropriate dtype
    low_cpu_mem_usage=True,
    device_map="auto" if is_cuda_available else {"": "cpu"}  # Map to GPU if available, otherwise CPU
)

# Set the model to evaluation mode to reduce overhead
model.eval()
logger.info("Model loaded and set to evaluation mode.")

# Define a prompt template to guide the model
PROMPT_TEMPLATE = """You are an AI assistant that provides accurate and concise answers to user queries.

User: {user_input}
Assistant:"""

# Optional: Implement conversation history (simple version using in-memory storage)
# For production, consider using a database or other persistent storage
conversation_history = {}

# Define a Flask route to handle chatbot requests
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get the input prompt from the request
        data = request.json
        user_id = data.get('user_id')  # To manage conversation history per user
        prompt = data.get('prompt', '').strip()

        if not prompt:
            logger.warning("Received empty prompt.")
            return jsonify({"error": "Prompt is empty or missing"}), 400

        if not user_id:
            logger.warning("User ID is missing in the request.")
            return jsonify({"error": "User ID is missing"}), 400

        # Retrieve conversation history if exists
        history = conversation_history.get(user_id, "")

        # Create the complete prompt with history
        complete_prompt = PROMPT_TEMPLATE.format(user_input=prompt)
        if history:
            complete_prompt = history + "\n" + complete_prompt

        logger.info(f"Generating response for user_id={user_id} with prompt: {prompt}")

        # Tokenize the input prompt
        inputs = tokenizer(
            complete_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Adjust max_length as per model's context window
        ).to(device)

        # Generate a response using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=250,  # Limit tokens to maintain concise answers
            temperature=0.7,      # Adjust temperature for creativity
            top_p=0.9,            # Increase top_p for diversity
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Penalize repetition
            no_repeat_ngram_size=3,  # Prevent repeating n-grams
            early_stopping=True
        )

        # Decode the generated tokens into a response string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's reply by removing the prompt
        response = response[len(complete_prompt):].strip()

        # Update conversation history
        conversation_history[user_id] = complete_prompt + "\n" + response

        logger.info(f"Response for user_id={user_id}: {response}")

        # Post-process the response to remove any irrelevant info
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while generating the response."}), 500

# Main function to run the Flask app
if __name__ == '__main__':
    # Load environment variables for Flask configuration if needed
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in ['true', '1', 't']

    logger.info(f"Starting Flask app on {host}:{port} with debug={debug}.")
    # Start the Flask app
    app.run(host=host, port=port, debug=debug)
