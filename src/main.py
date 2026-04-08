from features.StepFunAssistant import StepFunAssistant
from features.pdf import PDF
import argparse

def main():
    """
    Main function to handle command line arguments and PDF processing.
    
    This function orchestrates the entire PDF processing workflow:
    1. Parses command line arguments
    2. Creates a PDF object with the provided file path
    3. Loads and validates the PDF file
    4. Applies text cleaning filters
    5. Processes the cleaned text with StepFunAssistant
    6. Exports the extracted information as JSON
    
    The function includes error handling for common issues and provides
    user feedback at each step of the process.
    
    Command line arguments:
        file_path (str): Path to the PDF file to process (required)
        --output, -o (str): Path where the JSON output will be saved (optional)
    
    Returns:
        None. Exits with status code:
            - 0: Successful execution
            - 1: Error occurred (handled by PDF class or StepFunAssistant)
    
    Notes:
        - The PDF file must exist and be accessible
        - The output directory must have write permissions
        - The script automatically overwrites existing JSON files
        - All processing messages are printed to console for user feedback
        - The JSON output contains: title, DOI, summary (objective, methods, 
          results, conclusion), and metadata
    
    Dependencies:
        - PDF class from features.pdf: Handles PDF loading and text cleaning
        - StepFunAssistant class from features.StepFunAssistant: Handles AI analysis
        
    Example usage:
        python main.py /path/to/document.pdf
        python main.py /path/to/document.pdf -o /path/to/output.json
    
    See also:
        PDF class in features.pdf - Contains all PDF processing logic
        StepFunAssistant class in features.StepFunAssistant - Contains AI analysis logic
        clean_text() method - Applies text cleaning filters
        ask_json() method - Sends text to AI and returns structured JSON
    """
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Load and process a PDF file, extracting structured information using AI'
    )
    
    # Add file path argument (required)
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the PDF file to load and process'
    )
    
    # Add output option for JSON (optional)
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to the .json output file (if not provided, uses default outputs/ directory)'
    )

    # Parse command line arguments
    args = parser.parse_args()
    
    # Initialize PDF processor with the provided file path
    pdf = PDF(args.file_path)
    pdf.load_pdf()
    
    # Apply text cleaning filters to remove noise and improve quality
    pdf.clean_text()
    print("\nText cleaned successfully!")
    doi = pdf.get_doi()
    if doi:
        print(f"DOI encontrado: {doi}")
        # Exemplo: DOI encontrado: 10.1038/s41586-020-1234-5
    else:
        print("Nenhum DOI encontrado")
    input()
    
    # Extract the cleaned text as a string for AI processing
    clean_text = pdf.get_cleaned_text()

    try:
        # Initialize the AI assistant for paper analysis
        assistant = StepFunAssistant()
        
        # Send the cleaned text to the AI and get structured JSON response
        result = assistant.ask_json(
            paper_text=clean_text,
            save_to_json=True,
            output_file_path=args.file_path,
            custom_filename=args.output  # Uses the --output path for JSON export
        )

        # Display the extracted information to the user
        print("\n=== Extracted Paper Information ===")
        print(f"Title: {result['title']}")
        print(f"DOI: {result['doi']}")
        print("\n--- Summary ---")
        print(f"Objective: {result['summary']['objective']}")
        print(f"Methods: {result['summary']['methods']}")
        print(f"Results: {result['summary']['results']}")
        print(f"Conclusion: {result['summary']['conclusion']}")
        print("===================================\n")
        
    except KeyError as e:
        print(f"Error: Missing expected field in response - {e}")
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    main()