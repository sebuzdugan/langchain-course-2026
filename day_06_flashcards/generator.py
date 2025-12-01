from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from day_06_flashcards.models import FlashcardSet

def generate_flashcards(topic: str, retriever: BaseRetriever = None) -> FlashcardSet:
    """
    Generates a set of flashcards for a given topic using RAG (optional) and structured output.
    """
    print(f"\n🗂️  Generating flashcards for topic: {topic}")
    
    context = ""
    if retriever:
        # 1. retrieve relevant chunks
        print("--- Retrieving relevant context ---")
        docs = retriever.invoke(topic)
        print(f"✅ Retrieved {len(docs)} chunks.")
        # 2. combine context
        context = "\n\n".join([doc.page_content for doc in docs])
    else:
        print("--- No retriever provided. Using LLM knowledge. ---")
    
    # 3. create prompt
    template = """You are an expert teacher creating study flashcards.

Context:
{context}

Topic: {topic}

Instructions:
- Create 5 flashcards based on the context.
- The "front" should be a specific term, concept, or simple question.
- The "back" should be a clear, concise definition or answer.
- Focus on key concepts that are important for understanding.
- The output must be a valid JSON object matching the FlashcardSet schema.
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. initialize llm with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(FlashcardSet)
    
    # 5. create chain
    chain = prompt | structured_llm
    
    # 6. generate flashcards
    print("--- Generating flashcards (Structured Output) ---")
    flashcards = chain.invoke({"context": context, "topic": topic})
    
    return flashcards
