import os
import json
import torch
import logging
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    raise FileNotFoundError(
        f"Configuration file '{config_path}' not found. Please create it and add your Hugging Face API token."
    )

with open(config_path, 'r') as f:
    config = json.load(f)

hf_api_token = config.get("hf_api_token")

if not hf_api_token:
    logger.error("Hugging Face API token not found in config.json.")
    raise ValueError(
        "Hugging Face API token not found in config.json. Please add it under 'hf_api_token' key."
    )

# Authenticate with Hugging Face
try:
    login(token=hf_api_token, add_to_git_credential=True)  # Save token to git credentials
    logger.info("Authenticated with Hugging Face.")
except Exception as e:
    logger.error(f"Failed to authenticate with Hugging Face: {str(e)}")
    raise

# Initialize Flask application
app = Flask(__name__)

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.1-8B"  # Ensure this is the correct model name
logger.info(f"Loading tokenizer and model for '{model_name}'.")

try:
    # Use AutoTokenizer and AutoModelForCausalLM for compatibility
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {str(e)}")
    raise

# Check if CUDA (GPU) is available
is_cuda_available = torch.cuda.is_available()
device = "cuda" if is_cuda_available else "cpu"
logger.info(f"CUDA available: {is_cuda_available}. Using device: {device}.")

# Configure model to use 8-bit quantization with GPU if available
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                        # Use 8-bit for memory efficiency
    bnb_8bit_fp32_cpu_offload=True,           # Correct parameter name for CPU offloading
    llm_int8_threshold=6.0,                    # Threshold for switching between FP32 and INT8
) if is_cuda_available else None

try:
    # Load the model with quantization and offloading
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if is_cuda_available else torch.float32,  # Use appropriate dtype
        low_cpu_mem_usage=True,
        device_map="auto",  # Let transformers handle device placement
    )
    logger.info("Model loaded successfully with quantization and CPU offloading.")
except ValueError as ve:
    logger.error(f"ValueError during model loading: {str(ve)}")
    raise
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Set the model to evaluation mode to reduce overhead
model.eval()
logger.info("Model set to evaluation mode.")

# Define a prompt template to guide the model
PROMPT_TEMPLATE = "{user_input}"

# Optional: Implement conversation history (simple version using in-memory storage)
# For production, consider using a database or other persistent storage
conversation_history = {}

# Define a Flask route to handle chatbot requests
@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get the input prompt from the request
        data = request.json
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
        )

        # Move inputs to appropriate device
        if is_cuda_available:
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to("cpu")

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

    logger.info(f"Starting Flask app on {host}:{port}.")
    # Start the Flask app
    app.run(host=host, port=port)
