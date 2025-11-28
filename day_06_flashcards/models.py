from typing import List
from pydantic import BaseModel, Field

class Flashcard(BaseModel):
    """Represents a single flashcard."""
    front: str = Field(description="The term, concept, or question on the front of the card")
    back: str = Field(description="The definition, explanation, or answer on the back of the card")

class FlashcardSet(BaseModel):
    """Represents a set of flashcards."""
    topic: str = Field(description="The topic of the flashcards")
    cards: List[Flashcard] = Field(description="List of flashcards")
