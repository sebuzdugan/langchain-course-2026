from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from .state import AgentState, InputState, OutputState
from .semantic import SemanticMemory

# Initialize Semantic Memory
semantic_memory = SemanticMemory()

# --- nodes ---

def planner_node(state: AgentState):
    """
    Determines intent and retrieves relevant memories.
    """
    question = state["question"].lower()
    
    # 1. Retrieve relevant facts from semantic memory
    # In a real app, we'd use the question to search.
    # Here we just get all facts to inject into context.
    memories = semantic_memory.get_relevant_facts(question)
    
    # 2. Determine intent (simple keyword heuristic)
    intent = "chat"
    if "save" in question or "remember" in question or "my name is" in question:
        intent = "save_memory"
    
    print(f"--- Planner: Intent -> {intent} | Memories: {len(memories)} ---")
    return {"intent": intent, "memories": memories, "steps": ["Planned"]}

def memory_node(state: AgentState):
    """
    Saves a new fact to semantic memory.
    """
    question = state["question"]
    
    # Simple extraction: just save the whole message for this demo
    # In reality, use an LLM to extract the core fact.
    # Improved simple heuristic:
    if "remember that" in question:
        fact = question.split("remember that")[1].strip()
    elif "save" in question:
        fact = question.replace("save", "").strip()
    else:
        fact = question
    
    semantic_memory.save_fact(fact)
    
    return {
        "answer": f"I've remembered that: {fact}",
        "steps": ["Saved Memory"]
    }

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def chat_node(state: AgentState):
    """
    Generates a response using context and memories via a real LLM.
    """
    memories = state.get("memories", [])
    memory_context = "\n".join([f"- {m}" for m in memories])
    
    print(f"--- Chat Node (Context: {len(memories)} facts) ---")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    context = state.get("context", "")
    
    system_prompt = """You are a helpful AI Learning Assistant.
    
    Your goal is to help the user learn based on their questions.
    
    Here is what you know about the user (Semantic Memory):
    {memory_context}
    
    Here is some relevant context from the uploaded documents (if any):
    {context}
    
    If the user asks a question, answer it helpfully.
    If the memory context is relevant, refer to it.
    If the document context is relevant, use it to answer the question.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "memory_context": memory_context if memory_context else "No specific facts known yet.",
        "context": context if context else "No document context available.",
        "question": state["question"]
    })
        
    return {
        "answer": response.content,
        "steps": ["Chatted"]
    }

# --- routing ---

def route_by_intent(state: AgentState) -> Literal["save_memory", "chat"]:
    return state["intent"]

# --- graph construction ---

workflow = StateGraph(AgentState, input=InputState, output=OutputState)

workflow.add_node("planner", planner_node)
workflow.add_node("save_memory", memory_node)
workflow.add_node("chat", chat_node)

workflow.add_edge(START, "planner")

workflow.add_conditional_edges(
    "planner",
    route_by_intent,
    {
        "save_memory": "save_memory",
        "chat": "chat"
    }
)

workflow.add_edge("save_memory", END)
workflow.add_edge("chat", END)

# Create the checkpointer for Episodic Memory
# We use a connection to a local sqlite file
conn = sqlite3.connect("day_08_memory/checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Compile with checkpointer
app = workflow.compile(checkpointer=checkpointer)
