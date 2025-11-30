import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from day_09_evaluation.evaluator import evaluate_response

load_dotenv()

# 1. Define the System Under Test (The Agent)
# For this demo, we'll use a simple LLM call to represent our "Explanation Agent" from Day 4.
def generate_agent_response(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return llm.invoke(question).content

def main():
    print("\n⚖️  Starting Day 9: Evaluation (LLM-as-a-Judge)...\n")
    
    # 2. Load Dataset
    dataset_path = "day_09_evaluation/dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} test cases.\n")
    
    results = []
    
    # 3. Run Evaluation Loop
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"--- Test Case {i}: {question} ---")
        
        # A. Generate Answer
        answer = generate_agent_response(question)
        print(f"🤖 Agent Answer: {answer[:100]}...") # Truncate for display
        
        # B. Evaluate
        eval_result = evaluate_response(question, answer, ground_truth)
        score = eval_result["score"]
        reasoning = eval_result["reasoning"]
        
        print(f"👨‍⚖️  Judge Score: {score}/5")
        print(f"📝 Reasoning: {reasoning}\n")
        
        results.append(score)
        
    # 4. Aggregate Results
    avg_score = sum(results) / len(results)
    print("-" * 50)
    print(f"📊 Final Report: Average Score = {avg_score:.2f}/5")
    print("-" * 50)
    print("\n✅ Day 9 Verification Complete!")

if __name__ == "__main__":
    main()
