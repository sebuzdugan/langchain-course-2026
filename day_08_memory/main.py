import uuid
from day_08_memory.agent import app

def run_chat(thread_id: str, user_input: str):
    print(f"\n💬 User ({thread_id}): {user_input}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the agent
    # stream_mode="values" returns the full state at each step
    for event in app.stream({"question": user_input}, config=config, stream_mode="values"):
        if "answer" in event and event["answer"]:
            print(f"🤖 Agent: {event['answer']}")

def main():
    print("\n🧠 Starting Day 8: Memory (Verification)...\n")
    
    # 1. Create a thread ID (simulating a user session)
    # We use a fixed ID here to demonstrate persistence across script runs if we wanted,
    # but for this single run, we'll show semantic memory working.
    thread_id = "user_session_1"
    
    # 2. First interaction: Save a fact
    run_chat(thread_id, "Please remember that my name is Sebastian and I love coding.")
    
    # 3. Second interaction: Recall the fact (Semantic Memory)
    run_chat(thread_id, "What do you know about me?")
    
    # 4. Third interaction: New thread, same user (Semantic Memory should persist)
    thread_id_2 = "user_session_2"
    print(f"\n--- Switching to new thread {thread_id_2} ---")
    run_chat(thread_id_2, "Who am I?")

    print("\n✅ Day 8 Verification Complete!")

if __name__ == "__main__":
    main()
