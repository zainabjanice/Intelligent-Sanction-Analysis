from reasoning.query_router import ask_kg

print("=" * 60)
print("Interpol Knowledge Graph — Reasoning Chat")
print("=" * 60)
print("Type 'exit' to quit\n")

while True:
    q = input("Question: ")

    if q.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    try:
        answer = ask_kg(q)
        print("\n" + answer + "\n")
    except Exception as e:
        print("\nI couldn't answer that yet.")
        print("Reason:", str(e), "\n")
