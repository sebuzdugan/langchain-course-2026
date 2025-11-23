from typing import Literal
from langgraph.graph import StateGraph, START, END
from .state import AgentState, InputState, OutputState

# --- nodes ---

def planner_node(state: AgentState):
    # determines the user's intent.
    # for day 1, we'll use a simple keyword heuristic.
    # later, this will be an llm call.
    question = state["question"].lower()
    intent = "explain"
    
    if "quiz" in question:
        intent = "quiz"
    elif "flashcard" in question:
        intent = "flashcards"
    elif "plan" in question:
        intent = "study_plan"
        
    print(f"--- Planner: Intent detected -> {intent} ---")
    return {"intent": intent, "steps": ["Planned"]}

def retrieval_node(state: AgentState):
    # fetches relevant context.
    print("--- Retrieval Node ---")
    return {"steps": ["Retrieved"]}

def explanation_node(state: AgentState):
    # generates a simple explanation.
    print("--- Explanation Node ---")
    return {
        "answer": f"Here is a simple explanation for: {state['question']}",
        "steps": ["Explained"]
    }

def quiz_node(state: AgentState):
    # generates a quiz.
    print("--- Quiz Node ---")
    return {
        "answer": "Here is your generated quiz...",
        "quiz": "Question 1: ...",
        "steps": ["Generated Quiz"]
    }

def flashcard_node(state: AgentState):
    # generates flashcards.
    print("--- Flashcard Node ---")
    return {
        "answer": "Here are your flashcards...",
        "flashcards": "Front: ... Back: ...",
        "steps": ["Generated Flashcards"]
    }

def study_plan_node(state: AgentState):
    # generates a study plan.
    print("--- Study Plan Node ---")
    return {
        "answer": "Here is your study plan...",
        "study_plan": "Week 1: ...",
        "steps": ["Generated Plan"]
    }

def validation_node(state: AgentState):
    # checks if the output is valid.
    # if not, we could loop back (not implemented for day 1 simple flow).
    print("--- Validation Node ---")
    return {"steps": ["Validated"]}

# --- routing ---

def route_by_intent(state: AgentState) -> Literal["explain", "quiz", "flashcards", "study_plan"]:
    return state["intent"]

# --- graph construction ---

workflow = StateGraph(AgentState, input=InputState, output=OutputState)

# add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("explain", explanation_node)
workflow.add_node("quiz", quiz_node)
workflow.add_node("flashcards", flashcard_node)
workflow.add_node("study_plan", study_plan_node)
workflow.add_node("validate", validation_node)

# add edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")

# conditional routing from retriever to specific skill
workflow.add_conditional_edges(
    "retriever",
    route_by_intent,
    {
        "explain": "explain",
        "quiz": "quiz",
        "flashcards": "flashcards",
        "study_plan": "study_plan"
    }
)

# all skills go to validation
workflow.add_edge("explain", "validate")
workflow.add_edge("quiz", "validate")
workflow.add_edge("flashcards", "validate")
workflow.add_edge("study_plan", "validate")

# validation goes to end
workflow.add_edge("validate", END)

app = workflow.compile()
