import fitz  # PyMuPDF for PDF parsing
from bs4 import BeautifulSoup  # For HTML parsing
from newspaper import Article  # For extracting content from web articles
import xml.etree.ElementTree as ET  # For XML parsing
from docx import Document  # For DOCX parsing
from typing import Dict, List, Union, Generator
import logging
import os
import nltk
import json
import ijson  # For streaming JSON parsing

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles parsing of different document formats."""

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """
        Extract text from PDF files.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text content.
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise

    @staticmethod
    def parse_html(file_path: str) -> Dict[str, str]:
        """
        Extract content from local HTML files using BeautifulSoup.
        
        Args:
            file_path (str): Path to the HTML file.
        
        Returns:
            Dict[str, str]: Dictionary containing the title and text content.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                
                # Extract title
                title = ''
                if soup.title:
                    title = soup.title.string
                
                # Extract text from body, removing script and style elements
                for script in soup(['script', 'style']):
                    script.decompose()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
                
                return {"title": title, "text": text}
        except Exception as e:
            logger.error(f"Error parsing HTML from {file_path}: {str(e)}")
            raise

    @staticmethod
    def parse_xml(file_path: str) -> Dict:
        """
        Parse XML files (e.g., ArXiv dataset).
        
        Args:
            file_path (str): Path to the XML file.
        
        Returns:
            Dict: Parsed XML content as a dictionary.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Create a dictionary with tag-text pairs for each element.
            return {elem.tag: elem.text for elem in root.iter()}
        except Exception as e:
            logger.error(f"Error parsing XML {file_path}: {str(e)}")
            raise

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """
        Extract text from DOCX files using python-docx.
        
        Args:
            file_path (str): Path to the DOCX file.
        
        Returns:
            str: Extracted text content.
        """
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            raise

    @staticmethod
    def parse_json(file_path: str) -> Generator[str, None, None]:
        """
        Stream and parse large JSON files, yielding chunks of content.
        Handles both single objects and arrays of objects.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Yields:
            str: Text content from each parsed object.
        """
        try:
            total_text = []
            with open(file_path, 'rb') as file:
                # Process the JSON file as a stream of objects
                for record in ijson.items(file, 'item'):
                    # Extract meaningful text from the JSON object
                    text_parts = []
                    
                    # For arxiv-metadata format, extract specific fields
                    if isinstance(record, dict):
                        important_fields = ['title', 'abstract', 'authors', 'categories']
                        for field in important_fields:
                            if field in record:
                                value = record[field]
                                if isinstance(value, list):
                                    value = ', '.join(str(x) for x in value)
                                text_parts.append(f"{field}: {value}")
                    
                    if text_parts:
                        text = ' '.join(text_parts)
                        total_text.append(text)
                        
                        # Yield chunks of reasonable size to avoid memory issues
                        if len(total_text) >= 10:  # Process 10 records at a time
                            yield '\n\n'.join(total_text)
                            total_text = []
                            
            # Yield any remaining text
            if total_text:
                yield '\n\n'.join(total_text)
                
        except Exception as e:
            logger.error(f"Error parsing JSON from {file_path}: {str(e)}")
            raise

class TextPreprocessor:
    """Handles text preprocessing and cleaning."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and preprocess text content.
        
        Args:
            text (str): Raw text content.
        
        Returns:
            str: Cleaned text.
        """
        # Remove extra whitespace and basic cleaning
        return " ".join(text.split()).strip()

    @staticmethod
    def chunk_text(text: str, target_sentences: int = 5) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.
        Group sentences until approximately `target_sentences` are reached per chunk.
        
        Args:
            text (str): Input text.
            target_sentences (int): Approximate number of sentences per chunk.
        
        Returns:
            List[str]: List of text chunks.
        """
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= target_sentences:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

class DocumentProcessor:
    """Main class for the document processing pipeline."""

    def __init__(self):
        self.parser = DocumentParser()
        self.preprocessor = TextPreprocessor()

    def process_documents(self, file_paths: List[str], doc_type: str) -> List[str]:
        """
        Process multiple documents end-to-end: parsing, cleaning, and chunking.

        Args:
            file_paths (List[str]): List of file paths.
            doc_type (str): Type of document ('pdf', 'txt', 'html', 'xml', 'docx', 'json').

        Returns:
            List[str]: Combined chunks from all the documents.
        """
        all_chunks = []
        for file_path in file_paths:
            try:
                # Process each document using the selected document type
                chunks = self.process_document(file_path, doc_type)
                all_chunks.extend(chunks)  # Combine the chunks from all documents
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {str(e)}")
        return all_chunks

    def process_document(self, file_path: str, doc_type: str) -> List[str]:
        """
        Process a single document: parsing, cleaning, and chunking.

        Args:
            file_path (str): Path to the file.
            doc_type (str): Type of document ('pdf', 'txt', 'html', 'xml', 'docx', 'json').

        Returns:
            List[str]: List of text chunks.
        """
        try:
            if doc_type == 'pdf':
                content = self.parser.parse_pdf(file_path)
            elif doc_type == 'html':
                content = self.parser.parse_html(file_path)['text']
            elif doc_type == 'xml':
                content = self.parser.parse_xml(file_path)
            elif doc_type == 'docx':
                content = self.parser.parse_docx(file_path)
            elif doc_type == 'json':
                # Handle streaming JSON content differently
                chunks = []
                for text_chunk in self.parser.parse_json(file_path):
                    cleaned_text = self.preprocessor.clean_text(text_chunk)
                    chunks.extend(self.preprocessor.chunk_text(cleaned_text))
                return chunks
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            if isinstance(content, str):
                cleaned_text = self.preprocessor.clean_text(content)
                chunks = self.preprocessor.chunk_text(cleaned_text)
                return chunks
            
            return content

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def auto_detect_doc_type(self, file_path: str) -> str:
        """
        Auto-detect the document type based on the file extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Detected document type ('pdf', 'txt', 'html', 'xml', 'docx', or 'json').
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".pdf":
            return "pdf"
        elif ext in [".txt"]:
            return "txt"
        elif ext in [".html", ".htm"]:
            return "html"
        elif ext == ".xml":
            return "xml"
        elif ext == ".docx":
            return "docx"
        elif ext == ".json":
            return "json"
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
