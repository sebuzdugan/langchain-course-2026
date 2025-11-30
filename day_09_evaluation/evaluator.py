import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1 to 5")
    reasoning: str = Field(description="Reasoning for the score")

def evaluate_response(question: str, answer: str, ground_truth: str):
    """
    Evaluates the answer against the ground truth using an LLM.
    Returns a dict with score and reasoning.
    """
    
    # Initialize the judge LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(EvaluationScore)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator. Compare the generated answer to the ground truth."),
        ("human", """
        Question: {question}
        
        Ground Truth: {ground_truth}
        
        Generated Answer: {answer}
        
        Rate the Generated Answer on a scale of 1 to 5 based on accuracy and clarity.
        1 = Completely wrong
        5 = Perfect match in meaning
        
        Provide your reasoning.
        """)
    ])
    
    chain = prompt_template | structured_llm
    
    result = chain.invoke({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer
    })
    
    return result.dict()
