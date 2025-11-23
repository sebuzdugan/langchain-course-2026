import operator
from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict

class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
    steps: List[str]

class AgentState(TypedDict):
    question: str
    intent: str # 'explain', 'quiz', 'flashcards', 'study_plan'
    
    # content fields
    answer: Optional[str]
    quiz: Optional[str]
    flashcards: Optional[str]
    study_plan: Optional[str]
    
    # append-only log of steps taken
    steps: Annotated[List[str], operator.add]
