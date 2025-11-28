from typing import List
from pydantic import BaseModel, Field

class QuizQuestion(BaseModel):
    """Represents a single multiple-choice question."""
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of 4 possible answers (A, B, C, D)", min_items=4, max_items=4)
    correct_answer: str = Field(description="The correct answer (must be one of the options)")
    explanation: str = Field(description="Explanation of why the answer is correct")

class Quiz(BaseModel):
    """Represents a generated quiz."""
    topic: str = Field(description="The topic of the quiz")
    questions: List[QuizQuestion] = Field(description="List of questions in the quiz")
