from typing import List
from pydantic import BaseModel, Field

class StudyItem(BaseModel):
    """Represents a specific topic or activity within a study session."""
    activity: str = Field(description="The specific activity (e.g., 'Read section on X', 'Review Flashcards')")
    duration_minutes: int = Field(description="Estimated duration in minutes")
    resources: List[str] = Field(description="List of relevant resources or sections to focus on")

class StudySession(BaseModel):
    """Represents a block of study time."""
    goal: str = Field(description="The main goal of this session")
    items: List[StudyItem] = Field(description="List of specific activities for this session")
    total_duration_minutes: int = Field(description="Total duration of the session in minutes")

class StudyPlan(BaseModel):
    """Represents the overall study plan."""
    topic: str = Field(description="The main topic of the study plan")
    goal: str = Field(description="The overall learning objective")
    sessions: List[StudySession] = Field(description="List of study sessions")
