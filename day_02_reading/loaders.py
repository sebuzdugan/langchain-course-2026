from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

# --- loaders ---

def load_pdf(path: str) -> List[Document]:
    """
    Loads a PDF file using PyPDFLoader.
    
    Args:
        path (str): The file path to the PDF.
        
    Returns:
        List[Document]: A list of Documents, one for each page.
    """
    # loading pdf file
    print(f"--- Loading PDF: {path} ---")
    loader = PyPDFLoader(path)
    return loader.load()

def load_text(path: str) -> List[Document]:
    """
    Loads a text file using TextLoader.
    
    Args:
        path (str): The file path to the text file.
        
    Returns:
        List[Document]: A list of Documents (usually just one).
    """
    # loading text file
    print(f"--- Loading Text: {path} ---")
    loader = TextLoader(path)
    return loader.load()

def load_web(url: str) -> List[Document]:
    """
    Loads a web page using WebBaseLoader.
    
    Args:
        url (str): The URL of the web page.
        
    Returns:
        List[Document]: A list of Documents (usually just one, unless split).
    """
    # loading web page
    print(f"--- Loading Web: {url} ---")
    loader = WebBaseLoader(url)
    return loader.load()

# --- unified interface ---

class DocumentLoader:
    """
    A unified interface for loading documents from various sources.
    """
    
    @staticmethod
    def load(source: str) -> List[Document]:
        """
        Smartly loads a document based on the source string.
        
        - URLs (http/https) -> WebBaseLoader
        - .pdf files -> PyPDFLoader
        - Other files -> TextLoader
        
        Args:
            source (str): The file path or URL.
            
        Returns:
            List[Document]: The loaded documents.
        """
        # check if source is a url
        if source.startswith("http://") or source.startswith("https://"):
            return load_web(source)
        # check if source is a pdf
        elif source.lower().endswith(".pdf"):
            return load_pdf(source)
        else:
            # default to text loader for local files
            return load_text(source)

__all__ = ["load_pdf", "load_text", "load_web", "DocumentLoader"]
