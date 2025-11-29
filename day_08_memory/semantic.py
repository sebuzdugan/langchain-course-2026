import json
import os
from typing import List

class SemanticMemory:
    """
    A simple semantic memory store that saves facts to a JSON file.
    In a real app, this would likely be a vector database.
    """
    
    def __init__(self, file_path: str = "day_08_memory/user_profile.json"):
        self.file_path = file_path
        self._load_memory()

    def _load_memory(self):
        """Loads facts from the JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.facts = json.load(f)
            except json.JSONDecodeError:
                self.facts = []
        else:
            self.facts = []

    def _save_memory(self):
        """Saves facts to the JSON file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.facts, f, indent=2)

    def save_fact(self, fact: str):
        """Saves a new fact if it doesn't already exist."""
        if fact not in self.facts:
            self.facts.append(fact)
            self._save_memory()
            print(f"💾 Saved to Semantic Memory: {fact}")
        else:
            print(f"ℹ️ Fact already known: {fact}")

    def get_all_facts(self) -> List[str]:
        """Returns all stored facts."""
        return self.facts

    def get_relevant_facts(self, query: str) -> List[str]:
        """
        Returns relevant facts. 
        For this simple implementation, we'll just return all facts 
        since we don't have a vector store set up for this specific part yet.
        """
        # In a real implementation, you'd embed the query and search the vector store.
        return self.facts
