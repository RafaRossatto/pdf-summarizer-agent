from features.StepFunAssistant import StepFunAssistant
from features.pdf import PDF

def main():
    
    # load file
    pdf = PDF("/home/rafarossatto/Desktop/quali/1.pdf")
    pdf.load_pdf()
    
        # Apply cleaning
    pdf.clean_text()
    # Pegar o texto limpo como string
    clean_text = pdf.get_cleaned_text()

    print(f"Tamanho do texto: {len(clean_text)} caracteres")
    print(f"Primeiras 200 caracteres: {clean_text[:200]}")
    input()

    try:
        # Create an instance of the assistant
        assistant = StepFunAssistant()
        
        # Make a request
        query = "What are the capital of Brazil?"
        answer = assistant.ask(query)
        
        print(f"\n Assistant: {answer}")
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    main()