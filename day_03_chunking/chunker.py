from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks documents using Semantic Chunking.
    
    This method uses embeddings to identify "semantic breakpoints" in the text,
    ensuring that chunks represent coherent ideas rather than arbitrary splits.
    
    Args:
        docs (List[Document]): The list of documents to chunk.
        
    Returns:
        List[Document]: The list of chunked documents.
    """
    print("--- Chunking Documents (Semantic) ---")
    
    # initialize embeddings (requires openai api key in env)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # initialize semantic chunker
    # breakpoint_threshold_type="percentile" is a good default
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    # split documents
    chunks = text_splitter.split_documents(docs)
    
    print(f"--- Generated {len(chunks)} chunks from {len(docs)} documents ---")
    return chunks
