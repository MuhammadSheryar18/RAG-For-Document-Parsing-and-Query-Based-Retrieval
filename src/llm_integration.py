import os
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve the Hugging Face API key from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is not set. Please add it to your .env file.")

# The Hugging Face Inference endpoint for the model (default model: DistilGPT2)
API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def generate_response(query: str, context: str, max_new_tokens: int = 150, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """
    Generate a response using the model on Hugging Face.
    
    This function constructs a prompt by combining the context and query,
    sends the prompt to the Hugging Face Inference API, and returns the generated text.
    
    Args:
        query (str): The user's question.
        context (str): The context extracted from the relevant document(s).
        max_new_tokens (int): Max number of tokens to generate.
        temperature (float): Sampling temperature for creativity (higher = more random).
        top_p (float): Nucleus sampling probability for diversity.
    
    Returns:
        str: The generated response.
    """
    try:
        # Build a prompt combining context and the user's question.
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        logger.info(f"Constructed prompt: {prompt[:100]}...")  # log first 100 characters for brevity

        # Prepare headers and payload for the HF Inference API
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,  # Allow adjustment of tokens
                "temperature": temperature,        # Allow temperature adjustment
                "top_p": top_p                     # Allow top_p adjustment
            }
        }
        
        # Make the request to the inference API
        logger.info("Sending request to Hugging Face Inference API.")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_message = f"Hugging Face API returned status code {response.status_code}: {response.text}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        # Parse the JSON response
        result = response.json()
        
        # Log the raw result if it's unexpected
        if not (isinstance(result, list) and len(result) > 0):
            logger.warning(f"Unexpected response format: {result}")
            generated_text = str(result)
        else:
            generated_text = result[0].get("generated_text", "")
            if not generated_text:
                logger.warning(f"No 'generated_text' found in result: {result}")
                generated_text = str(result)
        
        logger.info("Received response from Hugging Face model.")
        return generated_text

    except Exception as e:
        logger.error(f"Error generating response from Hugging Face: {str(e)}")
        raise

if __name__ == "__main__":
    # Example local test
    test_query = "Who is Carolina Rapach?"
    test_context = "Carolina Rapach is a well-known data scientist who has worked on..."

    try:
        answer = generate_response(test_query, test_context)
        logger.info(f"Generated answer: {answer}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
