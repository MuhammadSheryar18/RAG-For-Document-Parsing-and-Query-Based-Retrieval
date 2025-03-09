from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from functools import lru_cache
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the model to avoid reloading it every time
@lru_cache(maxsize=1)
def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and caches the embedding model.

    Args:
        model_name (str): Name of the Hugging Face model.

    Returns:
        SentenceTransformer: Loaded model instance.
    """
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise


def get_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a single piece of text.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: The embedding vector.
    """
    model = load_model()
    try:
        logger.info(f"Generating embedding for a single text sample.")
        embedding = model.encode(text, convert_to_numpy=True)
        logger.info(f"Generated embedding shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text: {e}")
        raise


def get_embeddings(text_list: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks in batches.

    Args:
        text_list (List[str]): List of text strings.
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Array of embedding vectors.
    """
    model = load_model()
    try:
        logger.info(f"Generating embeddings for {len(text_list)} samples in batches of {batch_size}.")
        embeddings = model.encode(
            text_list,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings for list of texts: {e}")
        raise


def get_embeddings_from_file(file_path: str, batch_size: int = 64, target_sentences: Optional[int] = 5) -> np.ndarray:
    """
    Process a document file and generate embeddings for all the sentences in it.
    
    Args:
        file_path (str): Path to the document file (PDF, DOCX, etc.)
        batch_size (int): Batch size for embedding generation.
        target_sentences (Optional[int]): Approximate number of sentences per chunk.
    
    Returns:
        np.ndarray: Embeddings for the document's sentences.
    """
    try:
        # Load and preprocess the document (Implement this based on your ingestion pipeline)
        # Assuming the function process_document returns a list of text chunks
        processor = DocumentProcessor()
        chunks = processor.process_document(file_path)
        
        # Generate embeddings for the document chunks
        embeddings = get_embeddings(chunks, batch_size)
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings from file {file_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage for testing the embedding module.
    sample_text = "This is a test sentence for generating an embedding."
    embedding = get_embedding(sample_text)
    logger.info(f"Generated embedding shape for single text: {embedding.shape}")
    
    sample_texts = [
        "This is the first sentence.",
        "Here is another sentence.",
        "Sentence transformers provide great embeddings."
    ]
    embeddings = get_embeddings(sample_texts)
    logger.info(f"Generated embeddings shape for list of texts: {embeddings.shape}")
    
    # Example for processing a document file
    try:
        file_path = "example.pdf"
        embeddings_from_file = get_embeddings_from_file(file_path)
        logger.info(f"Generated embeddings from file with shape: {embeddings_from_file.shape}")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
