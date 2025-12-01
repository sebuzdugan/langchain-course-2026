from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from day_05_quizzes.models import Quiz

def generate_quiz(topic: str, retriever: BaseRetriever = None) -> Quiz:
    """
    Generates a quiz for a given topic using RAG (optional) and structured output.
    """
    print(f"\n📝 Generating quiz for topic: {topic}")
    
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
    template = """You are an expert teacher creating a quiz to test student understanding.

Context:
{context}

Topic: {topic}

Instructions:
- Create a quiz with 3 multiple-choice questions based on the context.
- Each question should have 4 options.
- Ensure the questions test understanding, not just memorization.
- Provide a clear explanation for the correct answer.
- The output must be a valid JSON object matching the Quiz schema.
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. initialize llm with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Quiz)
    
    # 5. create chain
    chain = prompt | structured_llm
    
    # 6. generate quiz
    print("--- Generating quiz (Structured Output) ---")
    quiz = chain.invoke({"context": context, "topic": topic})
    
    return quiz
