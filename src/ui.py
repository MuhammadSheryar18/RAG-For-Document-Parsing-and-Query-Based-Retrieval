import gradio as gr
from ingestion import DocumentProcessor
from llm_integration import generate_response

def answer_query(uploaded_files, doc_type: str, query: str):
    """
    Process the uploaded documents and generate an answer for the query.
    
    Args:
        uploaded_files: List of file objects uploaded via the UI.
        doc_type (str): Type of document ('pdf', 'txt', 'html', 'xml', 'docx').
        query (str): User's question.
    
    Returns:
        Tuple[str, str]: A tuple containing:
            - The generated response.
            - The retrieved context from the document.
    """
    if not uploaded_files:
        return "Please upload at least one document.", ""

    processor = DocumentProcessor()
    all_chunks = []

    try:
        # Process each uploaded document based on type
        for uploaded_file in uploaded_files:
            file_path = uploaded_file.name  # Get the file path
            chunks = processor.process_document(file_path, doc_type)
            all_chunks.extend(chunks)  # Combine all chunks from multiple documents
    except Exception as e:
        return f"Error processing documents: {str(e)}", ""

    # Combine the chunks into a single context
    context = " ".join(all_chunks)

    try:
        # Generate response from the LLM based on the query and the combined document context
        answer = generate_response(query, context)
    except Exception as e:
        return f"Error generating response: {str(e)}", ""

    return answer, context

# Create a Gradio interface for handling multiple file uploads
with gr.Blocks(title="Document-based Question Answering") as demo:
    gr.Markdown("# Document-based Question Answering")
    gr.Markdown("Upload multiple documents (PDF, TXT, HTML, XML, DOCX) and ask a question. The system will process the documents and generate a response using an LLM.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Documents", file_types=[".pdf", ".txt", ".html", ".xml", ".docx"], multiple=True)
        doc_type_input = gr.Dropdown(choices=["pdf", "txt", "html", "xml", "docx"], value="pdf", label="Document Type")
    
    query_input = gr.Textbox(label="Enter Your Query", placeholder="Type your question here...")
    submit_btn = gr.Button("Submit")
    
    with gr.Row():
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)
        context_output = gr.Textbox(label="Retrieved Context", interactive=False, lines=10)
    
    submit_btn.click(
        fn=answer_query,
        inputs=[file_input, doc_type_input, query_input],
        outputs=[answer_output, context_output]
    )

demo.launch(share=False, inbrowser=True)
