
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os
import json
import re

# ============================================
# DEFINIÇÃO DOS SCHEMAS (Pydantic)
# ============================================

class Author(BaseModel):
    """
    Represents an author of a scientific paper with their institutional affiliation.
    
    This schema defines the structure for author information extracted from
    academic papers, ensuring consistent formatting across all parsed documents.
    """
    name: str = Field(
        description="Full name of the author as it appears in the paper, including first and last names"
    )
    affiliation: Optional[str] = Field(
        description="Institution, university, or research organization the author belongs to",
        default=None
    )


class PaperAnalysis(BaseModel):
    """
    Complete schema for extracting and structuring scientific paper metadata and content.
    
    This model defines the comprehensive structure for parsing academic papers,
    capturing all essential bibliographic information and a structured summary
    of the paper's core scientific contributions.
    """
    title: str = Field(
        description="Exact title of the scientific paper as it appears in the publication, preserving original capitalization and formatting"
    )
    journal: Optional[str] = Field(
        description="Name of the journal, conference proceeding, or publication venue where the paper was published",
        default=None
    )
    publication_date: Optional[str] = Field(
        description="Publication date in any of these formats: year only (e.g., '2024'), year-month (e.g., '2024-03'), or full date (e.g., '2024-03-15')",
        default=None
    )
    authors: List[Author] = Field(
        description="Complete list of all authors who contributed to the paper, maintaining the order as they appear in the publication",
        default=[]
    )

    summary: dict = Field(
        description="Structured summary containing four required keys: 'objective' (research aim), 'methods' (methodology), 'results' (key findings), and 'conclusion' (main takeaways and implications)"
    )


# ============================================
# CLASSE PRINCIPAL
# ============================================

class StepFunAssistant:
    """
    A client class for interacting with the StepFun AI model to analyze scientific papers.
    
    This class provides a complete pipeline for extracting structured information from
    academic papers using the StepFun language model. It leverages LangChain for
    prompt templating and output parsing to ensure consistent, high-quality extraction
    of bibliographic metadata and content summaries.
    
    Key Features:
        - Automatic extraction of paper title, journal, publication date, and authors
        - Structured summarization (objective, methods, results, conclusion)
        - JSON output with automatic file saving
        - Configurable model parameters and output destinations
    
    Dependencies:
        - Requires environment variables: MY_KEY (API key) and BASE_URL (API endpoint)
        - Uses LangChain for prompt management and response parsing
        - OpenAI-compatible client interface for StepFun API
    """
    
    def __init__(self, model="stepfun/step-3.5-flash", output_dir="outputs"):
        """
        Initializes the StepFunAssistant with configuration and creates the API client.
        
        This constructor sets up the complete analysis pipeline by loading environment
        variables, validating configuration, creating the API client, and initializing
        LangChain components for structured output parsing.
        
        Args:
            model (str, optional): The AI model identifier to use for analysis.
                Defaults to "stepfun/step-3.5-flash". Other available models depend
                on your StepFun API subscription.
            output_dir (str, optional): Directory path where JSON outputs will be saved.
                Defaults to "outputs". The directory is automatically created if it
                doesn't exist.
        
        Raises:
            ValueError: If required environment variables (MY_KEY, BASE_URL) are missing
                from the .env file or system environment.
            RuntimeError: If the API client initialization fails due to connection
                or authentication issues.
        
        """
        load_dotenv()
        
        self.api_key = os.getenv("MY_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._validate_config()
        self.client = self._initialize_client()
        
        # Configurates the LangChain components
        self._setup_langchain()
    
    # ============================================
    # Cnfiguration Methods.
    # ============================================
    
    def _validate_config(self):
        """
        Validates that required environment variables are present before API initialization.
        
        This method checks for the existence of essential configuration variables
        that are required to authenticate and connect to the StepFun API. It should
        be called immediately after loading environment variables and before attempting
        to create the API client.
        
        The method verifies two critical variables:
            - MY_KEY: The API authentication key for accessing StepFun services
            - BASE_URL: The endpoint URL for the StepFun API
        
        Raises:
            ValueError: With descriptive error messages when either:
                - MY_KEY is missing, empty, or None
                - BASE_URL is missing, empty, or None
        
        Returns:
            None: This method doesn't return a value; it either succeeds silently
            or raises an exception.
        
        Note:
            This is an internal helper method and is not intended to be called directly
            by users of the class. It is automatically invoked during initialization.
        """
        if not self.api_key:
            raise ValueError("Error: MY_KEY not found in .env file")
        if not self.base_url:
            raise ValueError("Error: BASE_URL not found in .env file")

    def _initialize_client(self):
        """
        Creates and configures the OpenAI-compatible client instance for StepFun API.
        
        This method initializes an HTTP client that communicates with the StepFun API
        using the OpenAI client library. Since StepFun provides an OpenAI-compatible
        interface, we can use the standard OpenAI client with custom base URL and
        authentication.
        
        The client handles:
            - HTTP connection pooling and request management
            - Authentication header injection (API key)
            - Request/response serialization and deserialization
            - Error handling and retry logic (as configured by the OpenAI library)
        
        Returns:
            OpenAI: A configured OpenAI client instance ready to make API calls.
        
        Raises:
            RuntimeError: If client initialization fails due to:
                - Invalid API key format or expired credentials
                - Network connectivity issues to the BASE_URL endpoint
                - Invalid BASE_URL format (missing protocol, malformed URL)
                - Library configuration errors
        
        Note:
            This is an internal helper method and is not intended to be called directly
            by users of the class. It is automatically invoked during initialization.
        
        Example of successful initialization output:
            Client initialized successfully using model: stepfun/step-3.5-flash
        
        See Also:
            - OpenAI Python library documentation for client configuration options
            - StepFun API documentation for authentication requirements
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"Client initialized successfully using model: {self.model}")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize client: {e}")
    
    def _setup_langchain(self):
        """Configures the LangChain components (parser and prompt template)."""
        
        self.parser = JsonOutputParser(pydantic_object=PaperAnalysis)
        format_instructions = self.parser.get_format_instructions()
        
        # Use "user" em vez de "human" para compatibilidade com StepFun
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a scientific article analysis assistant specialized in extracting key information from academic papers.

    Your task is to analyze the provided paper text and extract the following information:

    {format_instructions}

    Important rules:
    - Return ONLY valid JSON
    - Do not add any text before or after the JSON
    - Do not use markdown code blocks (like ```json)
    - If a field is not found in the paper, use null for strings or [] for lists
    - Each summary field (objective, methods, results, conclusion) should be 1-2 sentences
    """),
            
            ("user", "Extract information from this scientific paper:\n\n{paper_text}")  # ← "user" em vez de "human"
        ])
        
        self._format_instructions = format_instructions
    
    # ============================================
    # MÉTODOS DE CRIAÇÃO DE MENSAGENS
    # ============================================
    
    def _create_messages(self, paper_text):
        """
        Creates message objects from LangChain templates and converts them to API-compatible format.
        
        This method bridges LangChain's message format and the StepFun API's expected format.
        LangChain returns message objects (SystemMessage, HumanMessage, etc.) while the
        StepFun API expects simple dictionaries with 'role' and 'content' keys.
        
        The conversion process:
            1. Uses LangChain's prompt template to format messages with the paper text
            2. Maps LangChain message types to API-compatible role names
            3. Converts each message object to a dictionary with role and content
        
        Role Mapping:
            - "human" (LangChain) → "user" (StepFun API)
            - "ai" (LangChain) → "assistant" (StepFun API)
            - "system" (LangChain) → "system" (StepFun API)
        
        Args:
            paper_text (str): The extracted text content from the scientific paper
                to be analyzed. This will be inserted into the user message.
        
        Returns:
            List[Dict[str, str]]: A list of message dictionaries formatted for the StepFun API.
            Each dictionary has the structure: {"role": "system/user/assistant", "content": "message text"}
        
        Example:
            >>> messages = self._create_messages("Paper content here...")
            >>> print(messages)
            [
                {"role": "system", "content": "You are a scientific article analysis assistant..."},
                {"role": "user", "content": "Extract information from this scientific paper: Paper content here..."}
            ]
        
        Note:
            This method includes debug print statements that show the role conversion
            process. These can be removed in production or disabled via logging configuration.
        """
        messages = self.prompt_template.format_messages(
            paper_text=paper_text,
            format_instructions=self._format_instructions
        )
        
        # Mapeamento de roles: LangChain -> StepFun API
        role_map = {
            "human": "user",
            "ai": "assistant", 
            "system": "system"
        }
        
        result = []
        for msg in messages:
            role = role_map.get(msg.type, msg.type)
            print(role)
            result.append({"role": role, "content": msg.content})
            print(f"🔍 DEBUG: msg.type={msg.type} -> role={role}")  # Debug
        return result
    
    # ============================================
    # MÉTODO PRINCIPAL
    # ============================================
    
    def ask_json(self, paper_text, doi=None, temperature=0.7, max_tokens=8000, 
                 save_to_json=True, custom_filename=None, output_file_path=None):
        """
        Sends scientific paper text to the AI model and returns a structured JSON response.
        
        Args:
            paper_text (str): The extracted and cleaned text from the PDF file.
            doi (str, optional): Digital Object Identifier of the paper.
            temperature (float): Controls randomness of responses (0.0-1.0).
            max_tokens (int): Maximum number of tokens in the response.
            save_to_json (bool): Whether to save the result to a JSON file.
            custom_filename (str, optional): Custom filename for JSON output.
            output_file_path (str, optional): Original PDF file path.
        
        Returns:
            dict: A structured dictionary containing the analysis results.
        """
        try:
            print("🔵 Creating messages with LangChain template...")
            messages = self._create_messages(paper_text)
            
            print("🔵 Calling API...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            print(f"🔵 Response received (length: {len(response_content)} chars)")
            
            # Parse the response using LangChain's JsonOutputParser
            print("🔵 Parsing with JsonOutputParser...")
            try:
                parsed_result = self.parser.parse(response_content)
                print("✅ LangChain parser succeeded")
            except Exception as parse_error:
                print(f"⚠️ LangChain parser failed: {parse_error}")
                print("🔧 Trying manual fallback...")
                parsed_result = self._manual_parse(response_content)
            
            # Organize the final result
            organized_result = {
                "title": parsed_result.get("title"),
                "journal": parsed_result.get("journal"),
                "publication_date": parsed_result.get("publication_date"),
                "authors": parsed_result.get("authors", []),
                "doi": doi,
                "summary": parsed_result.get("summary", {}),
                "_metadata": self._create_metadata(paper_text, temperature, max_tokens, output_file_path)
            }
            
            # Save to file if requested
            if save_to_json:
                self._save_to_json(organized_result, custom_filename, organized_result.get("title"))
            
            return organized_result
            
        except Exception as e:
            print(f"❌ Error in ask_json: {e}")
            return self._create_error_response(e, paper_text, doi, output_file_path, custom_filename, save_to_json)
    
    # ============================================
    # MÉTODOS DE PARSE (FALLBACK)
    # ============================================
    
    def _manual_parse(self, response_content):
        """
        Manual fallback to extract JSON from response when LangChain parser fails.
        
        This method serves as a robust backup parser when the primary LangChain
        JsonOutputParser encounters errors. It handles common response formatting
        issues that can occur with LLM outputs, such as:
            - Markdown code blocks wrapping the JSON (```json ... ```)
            - Extra text before or after the JSON content
            - Incomplete or malformed JSON structures
        
        The parsing strategy:
            1. Remove markdown code block markers (```json and ```)
            2. Extract the first valid JSON object using regex pattern matching
            3. Parse the extracted string with json.loads()
            4. Return empty result structure if parsing fails at any step
        
        Args:
            response_content (str): Raw response string from the AI model. This may
                contain JSON alone, JSON wrapped in markdown, or non-JSON text.
        
        Returns:
            dict: Parsed JSON content if successful, otherwise an empty result
            structure from _get_empty_result() containing None values for all fields.
        
        Example:
            >>> # Handles markdown-wrapped JSON
            >>> response = "```json\n{\"title\": \"AI Paper\"}\n```"
            >>> result = self._manual_parse(response)
            >>> print(result)  # {'title': 'AI Paper'}
            
            >>> # Handles plain JSON
            >>> response = "{\"title\": \"AI Paper\"}"
            >>> result = self._manual_parse(response)
            >>> print(result)  # {'title': 'AI Paper'}
            
            >>> # Handles malformed response
            >>> response = "The paper is about AI. {\"title\": \"AI Paper\"}"
            >>> result = self._manual_parse(response)
            >>> print(result)  # {'title': 'AI Paper'}
        
        Note:
            This method uses regex with re.DOTALL flag to match JSON across multiple
            lines. The empty result structure is returned as a fallback to prevent
            cascading failures in the analysis pipeline.
        """
        try:
            # Clean markdown code blocks
            clean = re.sub(r'```json\n?', '', response_content)
            clean = re.sub(r'```', '', clean)
            
            # Find JSON object
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                print("✅ Manual parse succeeded")
                return result
            else:
                print("❌ No JSON found in response")
                return self._get_empty_result()
                
        except Exception as e:
            print(f"❌ Manual parse failed: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self):
        """Returns an empty result structure."""
        return {
            "title": None,
            "journal": None,
            "publication_date": None,
            "authors": [],
            "summary": {
                "objective": None,
                "methods": None,
                "results": None,
                "conclusion": None
            }
        }
    
    # ============================================
    # MÉTODOS DE METADADOS
    # ============================================
    
    def _create_metadata(self, paper_text, temperature, max_tokens, output_file_path):
        """
    Creates a comprehensive metadata dictionary for the analysis result.
    
    This method generates structured metadata that captures important context
    about the analysis process, including model configuration, timing, and
    source information. This metadata is useful for:
        - Reproducibility (recording which model and parameters were used)
        - Debugging (tracking input size and processing time)
        - Audit trails (knowing which file was analyzed and when)
    
    The metadata includes two categories of information:
        1. Analysis parameters: model name, temperature, max_tokens, timestamp
        2. Source information: file path, name, directory (if available)
    
    Args:
        paper_text (str): The original paper text that was analyzed. Used to
            calculate the text length for the metadata.
        temperature (float): The temperature setting used for the AI model
            (controls randomness, typically 0.0-1.0).
        max_tokens (int): The maximum number of tokens allowed for the response.
        output_file_path (str, optional): The file system path to the original
            PDF or source document. If None, source_file will be set to None.
    
    Returns:
        dict: A metadata dictionary with the following structure:
            {
                "model": str,           # AI model identifier
                "temperature": float,   # Temperature parameter used
                "max_tokens": int,      # Max tokens limit
                "timestamp": str,       # ISO format timestamp
                "text_length": int,     # Length of input text in characters
                "source_file": {        # None if output_file_path is None
                    "path": str,        # Full file system path
                    "name": str,        # Base filename
                    "directory": str    # Parent directory path
                }
            }
    
    Example:
        >>> metadata = self._create_metadata(
        ...     paper_text="Full paper content...",
        ...     temperature=0.7,
        ...     max_tokens=8000,
        ...     output_file_path="/data/papers/paper.pdf"
        ... )
        >>> print(metadata['model'])
        'stepfun/step-3.5-flash'
        >>> print(metadata['text_length'])
        12500
        >>> print(metadata['source_file']['name'])
        'paper.pdf'
    
    Note:
        The timestamp uses datetime.now().isoformat() which produces a string
        like "2024-01-15T14:30:45.123456". The text_length is measured in
        characters (not tokens) for simplicity and easier interpretation.
    """
        metadata = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(paper_text)
        }
        
        # Add source file info if provided
        if output_file_path:
            metadata["source_file"] = {
                "path": output_file_path,
                "name": os.path.basename(output_file_path),
                "directory": os.path.dirname(output_file_path)
            }
        else:
            metadata["source_file"] = None
        
        return metadata
    
    # ============================================
    # MÉTODOS DE SALVAMENTO
    # ============================================
    
    def _save_to_json(self, data, filename=None, paper_name=None):
        """
        Saves a dictionary object to a JSON file with automatic filename generation.
        
        Args:
            data (dict): The dictionary data to be saved as JSON.
            filename (str, optional): Custom filename for the JSON output.
            paper_name (str, optional): The paper title used to generate a sanitized filename.
        
        Returns:
            str or None: The full filepath where the JSON was saved if successful.
        """
        try:
            # Generate filename if not provided
            if filename is None:
                if paper_name and paper_name.strip():
                    # Sanitize paper name for filename
                    safe_name = "".join(c for c in paper_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_name = safe_name.replace(' ', '_')[:50]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_name}_{timestamp}.json"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"paper_analysis_{timestamp}.json"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Full path
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ JSON saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            return None