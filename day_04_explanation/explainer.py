from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_explanation(question: str, retriever: BaseRetriever) -> str:
    """
    Generates a clear, simple explanation for a given question using RAG.
    
    Args:
        question (str): The user's question.
        retriever (BaseRetriever): The retriever to use for finding relevant context.
        
    Returns:
        str: The generated explanation.
    """
    print(f"\n❓ Question: {question}")
    
    # 1. retrieve relevant chunks
    print("--- Retrieving relevant context ---")
    docs = retriever.invoke(question)
    print(f"✅ Retrieved {len(docs)} chunks.")
    
    # 2. combine context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. create prompt
    # this prompt is optimized for simple, clear explanations
    template = """You are a helpful teacher explaining complex topics in simple terms.

Context:
{context}

Question: {question}

Instructions:
- Use the context above to answer the question
- Explain it as if you're talking to someone learning this for the first time
- Use simple language and avoid jargon when possible
- If you use technical terms, explain them briefly
- Be concise but complete

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. initialize llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 5. create chain
    chain = prompt | llm
    
    # 6. generate explanation
    print("--- Generating explanation ---")
    response = chain.invoke({"context": context, "question": question})
    
    return response.content
