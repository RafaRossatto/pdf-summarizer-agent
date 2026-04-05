# PDF Summarizer Agent

A powerful Python tool for extracting and analyzing scientific paper content using AI. This tool processes PDF files, cleans the extracted text, and uses the StepFun AI model to extract structured information including titles, DOIs, and comprehensive summaries.

## Features

- 📄 **PDF Processing**: Extracts and cleans text from PDF files
- 🤖 **AI-Powered Analysis**: Uses StepFun AI model to analyze scientific papers
- 📊 **Structured Output**: Returns JSON with title, DOI, and summary sections
- 💾 **Automatic Saving**: Saves analysis results as JSON files
- 🛡️ **Robust Error Handling**: Graceful degradation with detailed error responses
- 🔧 **Command Line Interface**: Easy to use with configurable options

## Workflow

1. **Get PDF** 
   - Load the PDF file from the provided path
   - Extract raw text content

2. **Clean Text**
   - Remove unwanted characters and formatting
   - Normalize whitespace and line breaks
   - Prepare clean text for LLM processing

3. **LLM Analysis**
   - Send cleaned text to OpenRouter API
   - LLM extracts: Title, DOI, Objective, Methods, Results, Conclusion
   - Returns structured JSON response

4. **Export JSON**
   - Save extracted information as JSON file
   - Add metadata (timestamp, model used, text length)
   - Display results in terminal

## Project Structure

**pdf-summarizer-agent/**
- **src/**
  - **features/**
    - `StepFunAssistant.py` - AI analysis client
    - `pdf.py` - PDF processing utilities
  - `main.py` - Main entry point
- **outputs/** - Default JSON output directory
- `requirements.txt` - Python dependencies
- `.env` - API configuration (create this)
- `README.md` - Documentation


## Installation

All requirements are available in a `requirements.txt` file.
To install use the following line:
pip install -r requirements.txt

## API Configuration

This project requires an API key from a Large Language Model (LLM). For this project, we used [OpenRouter](https://openrouter.ai/), where you can get a free API key.

### Getting Your Free API Key

1. Go to [https://openrouter.ai](https://openrouter.ai)
2. Create a free account
3. Navigate to **API Keys** section
4. Generate your API key

### Setting Up the .env File

For security and to keep your key private, you need to create a `.env` file in the project root:

```bash
# Create .env file
touch .env

MY_KEY=your_openrouter_api_key_here
BASE_URL=https://openrouter.ai/api/v1
```

## How to Use the Code

### Command Line Usage

After setting up your `.env` file, run the script from the terminal:

```bash
# Basic usage - processes PDF and saves JSON automatically
python src/main.py /path/to/your/paper.pdf

# Specify custom output location for the JSON file
python src/main.py /path/to/paper.pdf --output /path/to/result.jsona

# Short form using -o (saves to current directory)
python src/main.py /path/to/paper.pdf -o result.json
```