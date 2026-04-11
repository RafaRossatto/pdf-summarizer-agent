import os
import json
import re
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Imports LangChain 1.x
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ============================================
# DEFINIÇÃO DOS SCHEMAS (Pydantic)
# ============================================

class Author(BaseModel):
    """Schema para um autor do artigo"""
    name: str = Field(description="Nome completo do autor")
    affiliation: Optional[str] = Field(description="Instituição/universidade do autor", default=None)


class PaperAnalysis(BaseModel):
    """Schema completo do artigo científico"""
    title: str = Field(description="Título exato do artigo científico")
    journal: Optional[str] = Field(description="Nome da revista onde foi publicado", default=None)
    publication_date: Optional[str] = Field(description="Data de publicação (ano ou data completa)", default=None)
    authors: List[Author] = Field(description="Lista de autores do artigo", default=[])
    summary: dict = Field(description="Resumo do artigo com objective, methods, results, conclusion")


# ============================================
# CLASSE PRINCIPAL
# ============================================

class StepFunAssistant:
    """
    A client class for interacting with the StepFun AI model to analyze scientific papers.
    
    This class uses LangChain for prompt templating and output parsing to extract
    structured information from scientific papers.
    """
    
    def __init__(self, model="stepfun/step-3.5-flash", output_dir="outputs"):
        """
        Initializes the StepFunAssistant with configuration and creates the API client.
        
        Args:
            model (str): The AI model identifier to use for analysis.
            output_dir (str): Directory path where JSON outputs will be saved.
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
        
        # Configura os componentes do LangChain
        self._setup_langchain()
    
    # ============================================
    # MÉTODOS DE CONFIGURAÇÃO
    # ============================================
    
    def _validate_config(self):
        """Validates that required environment variables are present."""
        if not self.api_key:
            raise ValueError("Error: MY_KEY not found in .env file")
        if not self.base_url:
            raise ValueError("Error: BASE_URL not found in .env file")

    def _initialize_client(self):
        """Creates and configures the OpenAI-compatible client instance."""
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
        """Cria as mensagens usando o template LangChain"""
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
        
        Args:
            response_content (str): Raw response from the AI model
            
        Returns:
            dict: Parsed JSON content
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
        """Creates a comprehensive metadata dictionary for the analysis result."""
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
    
    # # ============================================
    # # MÉTODOS DE ERRO
    # # ============================================
    
    # def _create_error_response(self, error, paper_text, doi, output_file_path, custom_filename, save_to_json):
    #     """Creates a structured error response when analysis fails."""
    #     error_result = {
    #         "error": f"API Error: {error}",
    #         "title": None,
    #         "journal": None,
    #         "publication_date": None,
    #         "authors": [],
    #         "doi": doi,
    #         "summary": {},
    #         "_metadata": self._create_metadata(paper_text, 0, 0, output_file_path)
    #     }
        
    #     # Save error to file if requested
    #     if save_to_json:
    #         if custom_filename:
    #             json_filename = custom_filename
    #         elif output_file_path:
    #             base_name = os.path.splitext(os.path.basename(output_file_path))[0]
    #             json_filename = f"{base_name}_error.json"
    #         else:
    #             json_filename = None
            
    #         self._save_to_json(error_result, json_filename, "error_response")
        
    #     return error_result