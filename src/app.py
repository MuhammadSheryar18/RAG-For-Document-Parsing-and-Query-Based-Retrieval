import os
import logging
import gradio as gr

# Import modules from our project
from ingestion import DocumentProcessor
from embedding import get_embedding, get_embeddings
from retrieval import build_faiss_index, search_index
from llm_integration import generate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_generate_response(document_paths: list, query: str, doc_type: str):
    """
    Full pipeline that:
      1. Processes multiple documents to extract and chunk text.
      2. Generates embeddings for each chunk.
      3. Builds a FAISS index and retrieves the most relevant chunks based on the query.
      4. Uses the LLM to generate a response based on the retrieved context.
    
    Args:
        document_paths (list): List of paths to the uploaded documents.
        query (str): The user's query.
        doc_type (str): Type of document ('pdf', 'txt', 'html', 'xml').

    Returns:
        Tuple[str, str]: A tuple containing:
            - The generated answer from the LLM.
            - The context from the retrieved document chunks.
    """
    try:
        all_chunks = []
        
        # 1. Ingest and process each document.
        processor = DocumentProcessor()
        for document_path in document_paths:
            logger.info(f"Processing document: {document_path}")
            chunks = processor.process_document(document_path, doc_type)
            if chunks:
                all_chunks.extend(chunks)
            else:
                logger.warning(f"No text extracted from document: {document_path}")
        
        if not all_chunks:
            return "No text could be extracted from any document.", ""
        
        # 2. Generate embeddings for the text chunks.
        logger.info("Generating embeddings for document chunks...")
        embeddings = get_embeddings(all_chunks)
        
        # 3. Build FAISS index from the embeddings.
        logger.info("Building FAISS index...")
        index = build_faiss_index(embeddings)
        
        # 4. Convert query to an embedding and retrieve relevant chunks.
        logger.info("Generating embedding for query and performing search...")
        query_embedding = get_embedding(query)
        distances, indices = search_index(index, query_embedding, top_k=3)
        
        # Retrieve the top chunks based on the indices.
        top_chunks = [all_chunks[i] for i in indices[0]]
        context = " ".join(top_chunks)
        logger.info("Context for LLM generated.")
        
        # 5. Generate response using the Hugging Face model.
        answer = generate_response(query, context)
        return answer, context

    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        return f"Error in processing: {str(e)}", ""


def ui_function(uploaded_files, query: str, doc_type: str):
    """
    UI wrapper function that accepts uploaded files, a query,
    and runs the full RAG pipeline.
    
    Args:
        uploaded_files: List of file objects uploaded via the UI.
        query (str): The user's question.
        doc_type (str): The type of document ('pdf', 'txt', 'html', 'xml').
    
    Returns:
        Tuple[str, str]: The generated answer and the retrieved context.
    """
    if not uploaded_files:
        return "Please upload documents.", ""
    
    # Get the file paths from the uploaded files
    file_paths = [file.name for file in uploaded_files]
    return process_and_generate_response(file_paths, query, doc_type)


# Create a new Gradio Blocks interface.
with gr.Blocks(title="RAG Prototype: Document-based Q&A") as demo:
    gr.Markdown("# Document-based Question Answering")
    gr.Markdown(
        "Upload multiple documents (PDF, TXT, HTML, or XML) and ask a question. "
        "The system will extract and process the text, retrieve relevant content, "
        "and generate a response using an LLM."
    )
    
    with gr.Row():
        file_input = gr.File(
            label="Upload Documents", 
            file_count="multiple",  # Allow multiple files to be uploaded
            file_types=[".pdf", ".txt", ".html", ".xml"]
        )
        doc_type_input = gr.Dropdown(
            choices=["pdf", "txt", "html", "xml"],
            value="pdf",
            label="Document Type"
        )
    
    query_input = gr.Textbox(label="Enter Your Query", placeholder="Type your question here...")
    submit_btn = gr.Button("Submit")
    
    with gr.Row():
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)
        context_output = gr.Textbox(label="Retrieved Context", interactive=False, lines=10)
    
    submit_btn.click(
        fn=ui_function,
        inputs=[file_input, query_input, doc_type_input],
        outputs=[answer_output, context_output]
    )

# Launch the interface and open it automatically in your browser
demo.launch(share=False, inbrowser=True)
