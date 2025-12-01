from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from day_07_planning.models import StudyPlan

def generate_study_plan(topic: str, duration: str, retriever: BaseRetriever = None) -> StudyPlan:
    """
    Generates a study plan for a given topic and duration using RAG (optional) and structured output.
    """
    print(f"\n📅 Generating study plan for topic: {topic} ({duration})")
    
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
    template = """You are an expert study planner helping a student master a topic.

Context:
{context}

Topic: {topic}
Available Time: {duration}

Instructions:
- Create a structured study plan based on the context and available time.
- Break the time down into logical sessions (if the duration allows).
- For each session, define specific activities (reading, reviewing, practicing).
- Ensure the plan is realistic and actionable.
- The output must be a valid JSON object matching the StudyPlan schema.
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. initialize llm with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(StudyPlan)
    
    # 5. create chain
    chain = prompt | structured_llm
    
    # 6. generate plan
    print("--- Generating study plan (Structured Output) ---")
    plan = chain.invoke({"context": context, "topic": topic, "duration": duration})
    
    return plan
