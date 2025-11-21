# 10 Days of LangChain 2026: Build Your Own AI Learning Assistant

Welcome to the **10 Days of LangChain 2026** series! In this course, we are building a fully agentic AI Learning Assistant that learns with you.

## 🎯 The Goal
Build an AI that:
1.  **Reads** your PDFs, notes, and articles.
2.  **Understands** them using advanced semantic chunking.
3.  **Teaches** you with simple explanations.
4.  **Tests** you with quizzes and flashcards.
5.  **Plans** your study schedule.
6.  **Remembers** your progress.

## 🗺️ The Roadmap

| Day | Topic | Description |
| :--- | :--- | :--- |
| **Day 1** | 🧠 **Build the Brain** | Setting up LangGraph, the core agent, and the state schema. |
| **Day 2** | 📖 **Reading Sources** | Ingesting PDFs, text files, and web pages. |
| **Day 3** | 🔪 **Smart Chunking** | Semantic chunking & Hybrid Retrieval (Dense + Sparse + Rerank). |
| **Day 4** | 🗣️ **Explanations** | Generating clear, simple explanations from complex text. |
| **Day 5** | 📝 **Quizzes** | Generating dynamic quizzes to test understanding. |
| **Day 6** | 🗂️ **Flashcards** | Creating spaced-repetition flashcards. |
| **Day 7** | 📅 **Study Plans** | Agentic planning for study sessions. |
| **Day 8** | 💾 **Memory** | Adding episodic and semantic memory to the agent. |
| **Day 9** | ⚖️ **Evaluation** | Validating the agent's outputs and performance. |
| **Day 10** | 🚀 **Full App** | Putting it all together into a polished application. |

## 🏗️ Architecture
See [architecture/README.md](architecture/README.md) for a deep dive into the 7-layer architecture.

## 🛠️ Tech Stack
- **LangChain Core**: The backbone.
- **LangGraph**: For agentic orchestration.
- **Pydantic v2**: For strict data validation.
- **OpenAI / Anthropic**: LLM providers.
- **Vector Store**: (ChromaDB/Pinecone - TBD).

## 🚀 Getting Started

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up your `.env` file with API keys.
3.  Follow along with each Day's folder!
# langchain-course-2026
