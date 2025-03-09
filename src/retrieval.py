import faiss
import numpy as np
import logging
import time
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Build a FAISS index from the provided embeddings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings with shape (num_embeddings, embedding_dim).
        index_type (str): Type of FAISS index. Options:
            - "flat": IndexFlatL2 (exact search)
            - "ivf": IndexIVFFlat (inverted file, requires training)
            - "hnsw": IndexHNSWFlat (approximate nearest neighbors)
        
    Returns:
        faiss.Index: A FAISS index with the embeddings added.
    """
    start_time = time.time()
    num_embeddings, dim = embeddings.shape
    logger.info(f"Building FAISS index for {num_embeddings} embeddings of dimension {dim} using index type '{index_type}'")
    
    # Create index based on the specified type
    if index_type == "flat":
        index = faiss.IndexFlatL2(dim)  # Exact search, uses L2 distance
    elif index_type == "ivf":
        nlist = min(100, num_embeddings)  # Number of clusters for IVF index
        quantizer = faiss.IndexFlatL2(dim)  # Quantizer for training
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)  # Inverted File index
        if not index.is_trained:
            logger.info("Training IVF index...")
            index.train(embeddings)  # Train the index with the embeddings
    elif index_type == "hnsw":
        M = 32  # Number of neighbors in HNSW (approximate search)
        index = faiss.IndexHNSWFlat(dim, M)  # HNSW index
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add embeddings to the index
    index.add(embeddings)
    elapsed = time.time() - start_time
    logger.info(f"FAISS index built with {index.ntotal} embeddings in {elapsed:.2f} seconds.")
    return index

def search_index(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the top k nearest neighbors of the query embedding.
    
    Args:
        index (faiss.Index): The FAISS index to search.
        query_embedding (np.ndarray): Query embedding (1D array of shape (embedding_dim,) or 2D array (1, embedding_dim)).
        top_k (int): Number of nearest neighbors to retrieve.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - distances (np.ndarray): Distances to the nearest neighbors.
            - indices (np.ndarray): Indices of the nearest neighbors in the index.
    """
    start_time = time.time()
    
    # Ensure the query is 2D
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    
    # Validate top_k does not exceed total number of items in the index
    num_items = index.ntotal
    if top_k > num_items:
        logger.warning(f"top_k ({top_k}) exceeds number of embeddings ({num_items}). Adjusting top_k to {num_items}.")
        top_k = num_items
    
    # Perform the search and get distances and indices
    distances, indices = index.search(query_embedding, top_k)
    elapsed = time.time() - start_time
    logger.info(f"Retrieved top {top_k} neighbors in {elapsed:.4f} seconds.")
    return distances, indices

if __name__ == "__main__":
    try:
        # Create dummy embeddings: 10 vectors of dimension 384.
        dummy_embeddings = np.random.rand(10, 384).astype('float32')
        # Choose an index type: "flat", "ivf", or "hnsw".
        index = build_faiss_index(dummy_embeddings, index_type="hnsw")
        
        # Create a random query embedding of matching dimension.
        query = np.random.rand(384).astype('float32')
        distances, indices = search_index(index, query, top_k=3)
        
        logger.info(f"Query distances: {distances}")
        logger.info(f"Query indices: {indices}")
    except Exception as e:
        logger.error(f"Error in retrieval test: {str(e)}")
