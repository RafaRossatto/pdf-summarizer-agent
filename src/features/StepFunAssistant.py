import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

class StepFunAssistant:
    def __init__(self, model="stepfun/step-3.5-flash:free", output_dir="outputs"):
        """Initializes settings and the OpenAI client."""
        load_dotenv()
        
        self.api_key = os.getenv("MY_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._validate_config()
        self.client = self._initialize_client()

    def _validate_config(self):
        """Checks if required environment variables are present."""
        if not self.api_key:
            raise ValueError("Error: MY_KEY not found in .env file")
        if not self.base_url:
            raise ValueError("Error: BASE_URL not found in .env file")

    def _initialize_client(self):
        """Creates the OpenAI client instance."""
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"Client initialized successfully using model: {self.model}")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize client: {e}")

    def create_paper_analysis_messages(self, paper_text):
        """Creates the messages structure for paper analysis."""
        messages = [
                    {"role": "system", "content": """You are a scientific article analysis assistant specialized in extracting key information from academic papers.

                    Your task is to analyze the provided paper text and extract:
                    1. The exact title
                    2. The DOI (Digital Object Identifier) if present
                    3. A comprehensive summary

                    Return your response in JSON format as shown below. Do not add any text outside the JSON structure."""},
                                
                                {"role": "user", "content": f"""
                    Extract from this paper:

                    {paper_text}

                    Return in this exact JSON format:
                    {{
                        "title": "",
                        "doi": "",
                        "summary": {{
                            "objective": "",
                            "methods": "",
                            "results": "",
                            "conclusion": ""
                        }}
                    }}

                    If DOI is not found, use null. Keep each summary field to 1-2 sentences."""}
                ]
        
        return messages

    def _save_to_json(self, data, filename=None, paper_name=None):
        """Saves the dictionary to a JSON file."""
        try:
            # Generate filename if not provided
            if filename is None:
                if paper_name:
                    # Sanitize paper name for filename
                    safe_name = "".join(c for c in paper_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_name}.json"
                    
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

    def ask_json(self, paper_text, temperature=0.7, max_tokens=5000, save_to_json=True, custom_filename=None, original_file_path=None):
        """Sends the paper text and returns structured JSON response.
        
        Args:
            paper_text (str): The extracted text from the PDF
            temperature (float): Temperature for model response
            max_tokens (int): Maximum tokens in response
            save_to_json (bool): Whether to save result to JSON file
            custom_filename (str, optional): Custom filename for JSON output
            original_file_path (str, optional): Original PDF file path
        """
        try:
            messages = self.create_paper_analysis_messages(paper_text)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse the response as JSON
            response_content = response.choices[0].message.content
            
            # Try to extract JSON if there's any extra text
            try:
                # Find JSON in the response (in case model adds extra text)
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = response_content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response_content)
                
                # Extract file information
                file_name = None
                file_directory = None
                if original_file_path:
                    file_name = os.path.basename(original_file_path)
                    file_directory = os.path.dirname(original_file_path)
                
                # Add metadata to the result
                result["_metadata"] = {
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(paper_text),
                    "source_file": {
                        "path": original_file_path,
                        "name": file_name,
                        "directory": file_directory
                    } if original_file_path else None
                }
                
                # Save to JSON file if requested
                if save_to_json:
                    paper_title = result.get("title", None)
                    # Use custom filename or generate from original file name
                    if custom_filename:
                        json_filename = custom_filename
                    elif original_file_path:
                        # Generate filename from original PDF name
                        base_name = os.path.splitext(file_name)[0]
                        json_filename = f"{base_name}.json"
                    else:
                        json_filename = None
                    
                    self._save_to_json(result, json_filename, paper_title)
                
                return result
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                file_name = None
                file_directory = None
                if original_file_path:
                    file_name = os.path.basename(original_file_path)
                    file_directory = os.path.dirname(original_file_path)
                
                error_result = {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_content,
                    "_metadata": {
                        "model": self.model,
                        "timestamp": datetime.now().isoformat(),
                        "text_length": len(paper_text),
                        "source_file": {
                            "path": original_file_path,
                            "name": file_name,
                            "directory": file_directory
                        } if original_file_path else None
                    }
                }
                
                if save_to_json:
                    if custom_filename:
                        json_filename = custom_filename
                    elif original_file_path:
                        base_name = os.path.splitext(file_name)[0]
                        json_filename = f"{base_name}_error.json"
                    else:
                        json_filename = None
                        
                    self._save_to_json(error_result, json_filename, "error_response")
                
                return error_result
                
        except Exception as e:
            file_name = None
            file_directory = None
            if original_file_path:
                file_name = os.path.basename(original_file_path)
                file_directory = os.path.dirname(original_file_path)
            
            error_result = {
                "error": f"API Error: {e}",
                "title": None,
                "doi": None,
                "summary": None,
                "_metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(paper_text),
                    "source_file": {
                        "path": original_file_path,
                        "name": file_name,
                        "directory": file_directory
                    } if original_file_path else None
                }
            }
            
            if save_to_json:
                if custom_filename:
                    json_filename = custom_filename
                elif original_file_path:
                    base_name = os.path.splitext(file_name)[0]
                    json_filename = f"{base_name}_error.json"
                else:
                    json_filename = None
                    
                self._save_to_json(error_result, json_filename, "error_response")
            
            return error_result

    def ask_json_with_retry(self, paper_text, temperature=0.7, max_tokens=5000, max_retries=3, save_to_json=True, original_file_path=None):
        """Sends prompt with automatic retry on rate limits and returns JSON.
        
        Args:
            paper_text (str): The extracted text from the PDF
            temperature (float): Temperature for model response
            max_tokens (int): Maximum tokens in response
            max_retries (int): Number of retry attempts
            save_to_json (bool): Whether to save result to JSON file
            original_file_path (str, optional): Original PDF file path
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = self.ask_json(paper_text, temperature, max_tokens, save_to_json, 
                                    custom_filename=None, original_file_path=original_file_path)
                
                # Check if result contains an error
                if "error" in result and "429" in str(result.get("error", "")):
                    wait_time = (2 ** attempt) * 3
                    print(f"Rate limit detected. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                elif "error" in result and attempt < max_retries - 1:
                    # Other errors, maybe retry
                    wait_time = (2 ** attempt) * 2
                    print(f"Error detected: {result['error']}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return result
                    
            except Exception as e:
                last_error = str(e)
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    print(f"Rate limit detected. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    file_name = None
                    if original_file_path:
                        file_name = os.path.basename(original_file_path)
                    
                    error_result = {
                        "error": f"API Error: {e}",
                        "title": None,
                        "doi": None,
                        "summary": None,
                        "_metadata": {
                            "model": self.model,
                            "timestamp": datetime.now().isoformat(),
                            "source_file": {
                                "path": original_file_path,
                                "name": file_name
                            } if original_file_path else None
                        }
                    }
                    
                    if save_to_json:
                        if original_file_path:
                            base_name = os.path.splitext(file_name)[0]
                            json_filename = f"{base_name}_error.json"
                        else:
                            json_filename = None
                            
                        self._save_to_json(error_result, json_filename, "error_response")
                    
                    return error_result
        
        file_name = None
        if original_file_path:
            file_name = os.path.basename(original_file_path)
        
        error_result = {
            "error": f"Failed after {max_retries} retries. Last error: {last_error}",
            "title": None,
            "doi": None,
            "summary": None,
            "_metadata": {
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "source_file": {
                    "path": original_file_path,
                    "name": file_name
                } if original_file_path else None
            }
        }
        
        if save_to_json:
            if original_file_path:
                base_name = os.path.splitext(file_name)[0]
                json_filename = f"{base_name}_error.json"
            else:
                json_filename = None
                
            self._save_to_json(error_result, json_filename, "error_response")
        
        return error_result

    # def ask_json_with_retry(self, paper_text, temperature=0.7, max_tokens=5000, max_retries=3, save_to_json=True):
    #     """Sends prompt with automatic retry on rate limits and returns JSON."""
    #     last_error = None
        
    #     for attempt in range(max_retries):
    #         try:
    #             result = self.ask_json(paper_text, temperature, max_tokens, save_to_json)
                
    #             # Check if result contains an error
    #             if "error" in result and "429" in str(result.get("error", "")):
    #                 wait_time = (2 ** attempt) * 3
    #                 print(f"Rate limit detected. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
    #                 time.sleep(wait_time)
    #                 continue
    #             elif "error" in result and attempt < max_retries - 1:
    #                 # Other errors, maybe retry
    #                 wait_time = (2 ** attempt) * 2
    #                 print(f"Error detected: {result['error']}. Retrying in {wait_time} seconds...")
    #                 time.sleep(wait_time)
    #                 continue
    #             else:
    #                 return result
                    
    #         except Exception as e:
    #             last_error = str(e)
    #             if "429" in str(e) and attempt < max_retries - 1:
    #                 wait_time = (2 ** attempt) * 3
    #                 print(f"Rate limit detected. Waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
    #                 time.sleep(wait_time)
    #             else:
    #                 error_result = {
    #                     "error": f"API Error: {e}",
    #                     "title": None,
    #                     "doi": None,
    #                     "summary": None,
    #                     "_metadata": {
    #                         "model": self.model,
    #                         "timestamp": datetime.now().isoformat()
    #                     }
    #                 }
                    
    #                 if save_to_json:
    #                     self._save_to_json(error_result, "error_response")
                    
    #                 return error_result
        
    #     error_result = {
    #         "error": f"Failed after {max_retries} retries. Last error: {last_error}",
    #         "title": None,
    #         "doi": None,
    #         "summary": None,
    #         "_metadata": {
    #             "model": self.model,
    #             "timestamp": datetime.now().isoformat()
    #         }
    #     }
        
    #     if save_to_json:
    #         self._save_to_json(error_result, "error_response")
        
    #     return error_result

    # # Método original mantido para compatibilidade
    # def ask(self, question, temperature=0.7, max_tokens=5000):
    #     """Sends a prompt to the model and returns the raw response."""
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": """You are a scientific article analysis assistant. 
    #                 Extract from the provided text and return exactly in this format:
                    
    #                 Title of the paper: [full title exactly as appears in the paper]
    #                 DOI: [DOI if present, otherwise write "Not found"]
    #                 Resume of the paper: [concise 3-5 sentence summary covering: main objective, methodology, key results, and conclusions]
                    
    #                 Be precise and concise."""},
    #                 {"role": "user", "content": question}
    #             ],
    #             temperature=temperature,
    #             max_tokens=max_tokens
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         return f"API Error: {e}"