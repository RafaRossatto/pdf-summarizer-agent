# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# class StepFunAssistant:
#     def __init__(self, model="stepfun/step-3.5-flash:free"):
#         """Initializes settings and the OpenAI client."""
#         load_dotenv()
        
#         self.api_key = os.getenv("MY_KEY")
#         self.base_url = os.getenv("BASE_URL")
#         self.model = model
        
#         self._validate_config()
#         self.client = self._initialize_client()

#     def _validate_config(self):
#         """Checks if required environment variables are present."""
#         if not self.api_key:
#             raise ValueError(" Error: MY_KEY not found in .env file")
#         if not self.base_url:
#             raise ValueError(" Error: BASE_URL not found in .env file")

#     def _initialize_client(self):
#         """Creates the OpenAI client instance."""
#         try:
#             client = OpenAI(api_key=self.api_key, base_url=self.base_url)
#             print(f" Client initialized successfully using model: {self.model}")
#             return client
#         except Exception as e:
#             raise RuntimeError(f" Failed to initialize client: {e}")

#     def ask(self, question, temperature=0.7, max_tokens=5000):
#         """Sends a prompt to the model and returns the response."""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 # messages=[
#                 #     # {"role": "system", "content": "You are a helpful assistant that provides concise responses."},
#                 #     {"role": "system", "content": "You are a article reaer assistant, you gonna recive a text e will return:"
#                 #     "Title of the paper: "
#                 #     "Doi: "
#                 #     "Resume of the paper: "},
#                 #     {"role": "user", "content": question}
#                 # ],
#                 messages = [
#                     {"role": "system", "content": """You are a scientific article analysis assistant. 
#                     Extract from the provided text and return exactly in this format:
                    
#                     Title of the paper: [full title exactly as appears in the paper]
#                     DOI: [DOI if present, otherwise write "Not found"]
#                     Resume of the paper: [concise 3-5 sentence summary covering: main objective, methodology, key results, and conclusions]
                    
#                     Be precise and concise."""},
#                     {"role": "user", "content": question}
#                 ],
#                 temperature=temperature,
#                 max_tokens=max_tokens
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"API Error: {e}"

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

class StepFunAssistant:
    def __init__(self, model="stepfun/step-3.5-flash:free"):
        """Initializes settings and the OpenAI client."""
        load_dotenv()
        
        self.api_key = os.getenv("MY_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = model
        
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

    def ask_json(self, paper_text, temperature=0.7, max_tokens=5000):
        """Sends the paper text and returns structured JSON response."""
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
                    return json.loads(json_str)
                else:
                    return json.loads(response_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_content
                }
                
        except Exception as e:
            return {
                "error": f"API Error: {e}",
                "title": None,
                "doi": None,
                "summary": None
            }

    def ask_json_with_retry(self, paper_text, temperature=0.7, max_tokens=5000, max_retries=3):
        """Sends prompt with automatic retry on rate limits and returns JSON."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = self.ask_json(paper_text, temperature, max_tokens)
                
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
                    return {
                        "error": f"API Error: {e}",
                        "title": None,
                        "doi": None,
                        "summary": None
                    }
        
        return {
            "error": f"Failed after {max_retries} retries. Last error: {last_error}",
            "title": None,
            "doi": None,
            "summary": None
        }

    # Método original mantido para compatibilidade
    def ask(self, question, temperature=0.7, max_tokens=5000):
        """Sends a prompt to the model and returns the raw response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a scientific article analysis assistant. 
                    Extract from the provided text and return exactly in this format:
                    
                    Title of the paper: [full title exactly as appears in the paper]
                    DOI: [DOI if present, otherwise write "Not found"]
                    Resume of the paper: [concise 3-5 sentence summary covering: main objective, methodology, key results, and conclusions]
                    
                    Be precise and concise."""},
                    {"role": "user", "content": question}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {e}"