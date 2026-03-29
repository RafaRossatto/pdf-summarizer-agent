from features.StepFunAssistant import StepFunAssistant

def main():
    try:
        # Create an instance of the assistant
        assistant = StepFunAssistant()
        
        # Make a request
        query = "What are the benefits of using Object-Oriented Programming?"
        answer = assistant.ask(query)
        
        print(f"\n Assistant: {answer}")
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    main()