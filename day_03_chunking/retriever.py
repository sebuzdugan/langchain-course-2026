from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_hybrid_retriever(docs: List[Document]) -> BaseRetriever:
    """
    Creates a Hybrid Retriever (BM25 + Vector Search).
    
    Args:
        docs (List[Document]): The list of chunked documents to index.
        
    Returns:
        BaseRetriever: The ensemble retriever.
    """
    print("--- Creating Hybrid Retriever ---")
    
    # 1. bm25 retriever (sparse / keyword)
    # good for exact matches and specific terms
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    # 2. vector retriever (dense / semantic)
    # good for conceptual matching
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="day_03_hybrid"
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. ensemble retriever
    # combines both with equal weight (0.5 / 0.5)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever
