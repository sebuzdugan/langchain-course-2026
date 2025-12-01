import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

class InputState(TypedDict):
    question: str
    context: Optional[str]

class OutputState(TypedDict):
    answer: str
    steps: List[str]

class AgentState(TypedDict):
    question: str
    context: Optional[str]
    intent: str 
    
    # content fields
    answer: Optional[str]
    
    # memory fields
    memories: List[str] # Facts retrieved from semantic memory
    
    # append-only log of steps taken
    steps: Annotated[List[str], operator.add]
