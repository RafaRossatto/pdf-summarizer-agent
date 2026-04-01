from features.StepFunAssistant import StepFunAssistant
from features.pdf import PDF

def main():
    
    # load file
    path = "/home/rafarossatto/Desktop/quali/2.pdf"
    pdf = PDF(path)
    pdf.load_pdf()
    
        # Apply cleaning
    pdf.clean_text()
    # Pegar o texto limpo como string
    clean_text = pdf.get_cleaned_text()

    try:
    #     # Create an instance of the assistant
        assistant = StepFunAssistant()
        #result = assistant.ask_json(clean_text, save_to_json=True)
        # Processar com informações do arquivo original
        result = assistant.ask_json(
            paper_text=clean_text,
            save_to_json=True,
            original_file_path=path  # Passa o caminho do arquivo original
)
        # # Método 1: Salva JSON automaticamente com nome baseado no título
        # result = assistant.ask_json(paper_text, save_to_json=True)

        # # Método 2: Com nome personalizado
        # result = assistant.ask_json(paper_text, save_to_json=True, custom_filename="meu_artigo_2024.json")

        # # Método 3: Sem salvar (apenas retorna)
        # result = assistant.ask_json(paper_text, save_to_json=False)

        # # Método 4: Com retry e salvamento automático
        # result = assistant.ask_json_with_retry(paper_text, max_retries=3, save_to_json=True)

        # Acessa os campos
        print(f"Title: {result['title']}")
        print(f"DOI: {result['doi']}")
        print(f"Objective: {result['summary']['objective']}")
        print(f"Methods: {result['summary']['methods']}")
        print(f"Results: {result['summary']['results']}")
        print(f"Conclusion: {result['summary']['conclusion']}")
    except Exception as e:
         print(f"System Error: {e}")

if __name__ == "__main__":
    main()