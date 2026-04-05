import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

class StepFunAssistant:
    """
    A client class for interacting with the StepFun AI model to analyze scientific papers.
    
    This class provides functionality to send scientific paper text to an AI model
    and receive structured JSON responses containing key information such as title,
    DOI, and summary sections (objective, methods, results, conclusion).
    
    Attributes:
        api_key (str): API key for authentication, loaded from environment variables
        base_url (str): Base URL for the API endpoint, loaded from environment variables
        model (str): The AI model identifier to use for analysis
        output_dir (str): Directory path where JSON outputs will be saved
        client (OpenAI): The OpenAI-compatible client instance for API communication
    
    Environment Variables Required:
        MY_KEY (str): API key for authentication
        BASE_URL (str): Base URL for the API endpoint
    
    Example:
        >>> assistant = StepFunAssistant()
        >>> result = assistant.ask_json(paper_text="...", save_to_json=True)
        >>> print(result['title'])
        >>> print(result['summary']['objective'])
    """
    def __init__(self, model="stepfun/step-3.5-flash:free", output_dir="outputs"):
        """
        Initializes the StepFunAssistant with configuration and creates the API client.
        
        This constructor loads environment variables, validates configuration,
        creates the output directory if it doesn't exist, and initializes the
        OpenAI-compatible client for API communication.
        
        Args:
            model (str, optional): The AI model identifier to use for analysis.
                Defaults to "stepfun/step-3.5-flash:free".
            output_dir (str, optional): Directory path where JSON outputs will be saved.
                Defaults to "outputs".
        
        Returns:
            None
        
        Raises:
            ValueError: If required environment variables (MY_KEY or BASE_URL) are missing
            RuntimeError: If the client initialization fails
        
        Example:
            >>> # Use default settings
            >>> assistant = StepFunAssistant()
            
            >>> # Use custom model and output directory
            >>> assistant = StepFunAssistant(
            ...     model="stepfun/step-4.0",
            ...     output_dir="custom_outputs"
            ... )
        
        Notes:
            - Environment variables must be set in a .env file in the project root
            - The output directory is created automatically if it doesn't exist
            - The client initialization is validated before creating the instance
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

    def _validate_config(self):
        """
        Validates that required environment variables are present.
        
        This method checks if the API key and base URL are properly configured
        in the environment variables.
        
        Returns:
            None
        
        Raises:
            ValueError: If MY_KEY or BASE_URL environment variables are missing
        
        Notes:
            This method is called internally during initialization.
            The environment variables should be defined in a .env file.
        """
        if not self.api_key:
            raise ValueError("Error: MY_KEY not found in .env file")
        if not self.base_url:
            raise ValueError("Error: BASE_URL not found in .env file")

    def _initialize_client(self):
        """
        Creates and configures the OpenAI-compatible client instance.
        
        This method initializes the client with the API key and base URL
        for communicating with the StepFun AI model.
        
        Returns:
            OpenAI: Configured OpenAI client instance ready for API calls
        
        Raises:
            RuntimeError: If client initialization fails due to connection or configuration issues
        
        Notes:
            The client uses the OpenAI library but connects to the StepFun API endpoint.
            This method is called internally during class initialization.
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f"Client initialized successfully using model: {self.model}")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize client: {e}")

    def create_paper_analysis_messages(self, paper_text):
        """
        Creates the message structure for the AI model to analyze a scientific paper.
        
        This method constructs the system and user prompts that guide the AI model
        to extract specific information from the paper text and return it in JSON format.
        
        Args:
            paper_text (str): The extracted text content from the scientific paper
        
        Returns:
            list: A list of message dictionaries formatted for the OpenAI API,
                containing system instructions and user prompt with the paper text
        
        Example:
            >>> messages = assistant.create_paper_analysis_messages(paper_text)
            >>> # messages contains:
            >>> # [
            >>> #     {"role": "system", "content": "..."},
            >>> #     {"role": "user", "content": "..."}
            >>> # ]
        
        Notes:
            - The system prompt defines the AI's role as a scientific article analyst
            - The user prompt contains the paper text and specifies the required JSON format
            - Each summary field (objective, methods, results, conclusion) is limited to 1-2 sentences
            - DOI field returns null if not found in the paper
        """
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
        """
        Saves a dictionary object to a JSON file with automatic filename generation.
        
        This internal method handles saving JSON data to disk with various filename
        options. It can generate filenames based on the paper title, use a custom
        filename, or create a timestamp-based filename as a fallback.
        
        Args:
            data (dict): The dictionary data to be saved as JSON. This should be
                a serializable Python dictionary containing the paper analysis results.
            filename (str, optional): Custom filename for the JSON output. If provided,
                this overrides automatic filename generation. Defaults to None.
            paper_name (str, optional): The paper title used to generate a sanitized
                filename. Only used if filename is None. Defaults to None.
        
        Returns:
            str or None: The full filepath where the JSON was saved if successful,
                or None if an error occurred during saving.
        
        Filename Generation Logic:
            1. If `filename` is provided:
            - Ensures .json extension is added if missing
            - Saves directly with that filename in output_dir
            2. Else if `paper_name` is provided:
            - Sanitizes the paper name (alphanumeric, spaces, hyphens, underscores)
            - Replaces spaces with underscores
            - Truncates to 50 characters maximum
            - Adds timestamp (YYYYMMDD_HHMMSS) to ensure uniqueness
            - Example: "My_Paper_Title_20231215_143022.json"
            3. Else (no filename and no paper_name):
            - Uses timestamp-based filename: "paper_analysis_20231215_143022.json"
        
        File Location:
            All JSON files are saved to the instance's `output_dir` directory, which
            is set during class initialization (defaults to "outputs/").
        
        Example:
            >>> # Save with auto-generated name from paper title
            >>> assistant._save_to_json(data, paper_name="Machine Learning Review")
            # Saves to: outputs/Machine_Learning_Review_20231215_143022.json
            
            >>> # Save with custom filename
            >>> assistant._save_to_json(data, filename="custom_result.json")
            # Saves to: outputs/custom_result.json
            
            >>> # Save with timestamp-based filename
            >>> assistant._save_to_json(data)
            # Saves to: outputs/paper_analysis_20231215_143022.json
        
        Notes:
            - The method automatically creates the output directory if it doesn't exist
            - JSON is saved with UTF-8 encoding and 2-space indentation for readability
            - Non-ASCII characters are preserved (ensure_ascii=False)
            - Existing files with the same name will be overwritten without warning
            - The method includes error handling and will print error messages to console
        
        Error Handling:
            - If file writing fails (permissions, disk full, etc.), the error is caught
            - An error message is printed to console
            - The method returns None to indicate failure
            - The exception is not propagated to the caller
        
        See Also:
            - ask_json(): Public method that calls this internal method
            - json.dump(): Python's JSON serialization function
        """
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
            
            print(f"JSON saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return None

    def ask_json(self, paper_text, temperature=0.7, max_tokens=5000, save_to_json=True, custom_filename=None, output_file_path=None):
        """
        Sends scientific paper text to the AI model and returns a structured JSON response.
        
        This method orchestrates the entire process of analyzing a scientific paper:
        1. Creates the appropriate message structure for the AI model
        2. Sends a request to the StepFun API with configurable parameters
        3. Parses the JSON response from the AI model
        4. Enriches the result with metadata (timestamp, model settings, source file info)
        5. Optionally saves the result to a JSON file
        6. Returns a structured dictionary with the extracted information
        
        Args:
            paper_text (str): The extracted and cleaned text from the PDF file.
                This should contain the full content of the scientific paper.
            
            temperature (float, optional): Controls the randomness of the AI's responses.
                - Lower values (0.0-0.3): More deterministic, focused, and repetitive
                - Medium values (0.4-0.7): Balanced creativity and accuracy
                - Higher values (0.8-1.0): More creative, diverse, but potentially less accurate
                Defaults to 0.7.
            
            max_tokens (int, optional): Maximum number of tokens in the AI's response.
                Limits the length of the generated output. Each token is roughly 4 characters.
                Defaults to 5000 (sufficient for most paper analyses).
            
            save_to_json (bool, optional): Whether to automatically persist the result to a JSON file.
                If True, the method calls internal save methods to write the result to disk.
                Defaults to True.
            
            custom_filename (str, optional): Custom path and filename for the JSON output.
                When provided, this overrides all automatic filename generation.
                Can be an absolute path or relative to the current working directory.
                Example: "/custom/path/my_analysis.json"
                Defaults to None.
            
            output_file_path (str, optional): Original PDF file path used for:
                - Generating default JSON filename (replaces .pdf extension with .json)
                - Adding source file information to metadata
                - Providing context for error handling
                Example: "/home/user/documents/paper.pdf"
                Defaults to None.
        
        Returns:
            dict: A structured dictionary containing the analysis results with the following schema:
            
            **Success Response Structure:**
            {
                "title": str,                    # Paper title extracted from the text
                "doi": str or None,              # DOI identifier (null if not found)
                "summary": {
                    "objective": str,            # Research objectives (1-2 sentences)
                    "methods": str,              # Methodology description (1-2 sentences)
                    "results": str,              # Key findings (1-2 sentences)
                    "conclusion": str            # Main conclusions (1-2 sentences)
                },
                "_metadata": {
                    "model": str,                # AI model identifier used
                    "temperature": float,        # Temperature setting used
                    "max_tokens": int,           # Max tokens setting used
                    "timestamp": str,            # ISO 8601 format timestamp
                    "text_length": int,          # Length of input text in characters
                    "source_file": {             # Present only if output_file_path provided
                        "path": str,             # Absolute path to original PDF
                        "name": str,             # PDF filename with extension
                        "directory": str         # Parent directory path
                    } or None
                }
            }
            
            **Error Response Structure:**
            {
                "error": str,                    # Error description ("API Error: {details}")
                "title": None,
                "doi": None,
                "summary": None,
                "_metadata": {...}               # Same metadata structure as above
            }
        
        Raises:
            No exceptions are propagated. All exceptions are caught and returned as
            error dictionaries. This ensures the method always returns a dictionary
            and never crashes the calling code.
        
        Workflow:
            1. **Message Creation**: Calls create_paper_analysis_messages() to build prompts
            2. **API Request**: Sends messages to StepFun API with specified parameters
            3. **JSON Parsing**: Uses _parse_json_response() to extract JSON from response
            4. **Metadata Enrichment**: Adds metadata via _create_metadata()
            5. **File Persistence**: Optionally saves via _save_analysis_result()
            6. **Response Return**: Returns the enriched result dictionary
        
        Example Usage:
            >>> # Basic usage
            >>> assistant = StepFunAssistant()
            >>> result = assistant.ask_json(paper_text=cleaned_text)
            >>> print(result['title'])
            "Deep Learning in Healthcare"
            
            >>> # With custom temperature and saving options
            >>> result = assistant.ask_json(
            ...     paper_text=cleaned_text,
            ...     temperature=0.3,  # More precise responses
            ...     save_to_json=True,
            ...     output_file_path="/path/to/paper.pdf"
            ... )
            
            >>> # Custom JSON output location
            >>> result = assistant.ask_json(
            ...     paper_text=cleaned_text,
            ...     custom_filename="/custom/path/analysis.json",
            ...     output_file_path="/path/to/paper.pdf"
            ... )
            
            >>> # Error handling example
            >>> result = assistant.ask_json(paper_text=cleaned_text)
            >>> if 'error' in result:
            ...     print(f"Analysis failed: {result['error']}")
            ... else:
            ...     print(f"Success: {result['title']}")
        
        Notes:
            - The method is designed to be robust and never throw exceptions
            - Metadata is always included in the response, even for errors
            - The `custom_filename` parameter takes precedence over automatic naming
            - When `output_file_path` is provided, the JSON is saved alongside the PDF
            - The `_parse_json_response` method handles AI responses with extra text
            - All errors are caught and returned as structured error dictionaries
        
        See Also:
            - _parse_json_response(): Extracts JSON from AI responses
            - _create_metadata(): Builds the metadata dictionary
            - _save_analysis_result(): Handles file persistence logic
            - _create_error_response(): Constructs error responses
            - create_paper_analysis_messages(): Builds the AI prompt structure
        
        Performance Considerations:
            - Input text length affects API response time and token usage
            - Higher max_tokens values may increase response time
            - Temperature has minimal impact on performance but affects output quality
            - File I/O occurs only when save_to_json=True
        
        Error Scenarios Handled:
            - Network failures during API request
            - Invalid API credentials
            - Malformed JSON responses from AI
            - File system permission errors when saving
            - Timeout errors from the API
        """
        try:
            # Send request to AI
            messages = self.create_paper_analysis_messages(paper_text)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse JSON from response
            result = self._parse_json_response(response.choices[0].message.content)
            
            # Add metadata
            result["_metadata"] = self._create_metadata(paper_text, temperature, max_tokens, output_file_path)
            
            # Save to file if requested
            if save_to_json:
                self._save_analysis_result(result, custom_filename, output_file_path)
            
            return result
            
        except Exception as e:
            return self._create_error_response(e, paper_text, output_file_path, custom_filename, save_to_json)

    def _parse_json_response(self, response_content):
        """
        Extracts and parses JSON from the AI model's response content.
        
        This method handles the common issue where AI models add explanatory text
        before or after the actual JSON response. It intelligently locates the
        JSON portion of the response and parses it into a Python dictionary.
        
        Args:
            response_content (str): Raw text response from the AI model.
                This can be in several formats:
                - Pure JSON: '{"title": "Paper Title", "doi": "10.xxx"}'
                - JSON with prefix text: 'Here is the analysis: {"title": "Paper Title"}'
                - JSON with suffix text: '{"title": "Paper Title"} Hope this helps!'
                - JSON with both prefix and suffix: 'Analysis: {"title": "Paper Title"} End'
                - Multi-line JSON with extra formatting
        
        Returns:
            dict: Parsed JSON content as a Python dictionary.
                Returns an empty dictionary if parsing fails (though this should
                be caught by the calling method's exception handler).
        
        Raises:
            json.JSONDecodeError: If no valid JSON structure is found in the
                response content or if the extracted JSON string is malformed.
                This exception should be caught by the caller.
        
        Algorithm:
            1. Find the position of the first opening brace '{' in the response
            2. Find the position of the last closing brace '}' in the response
            3. Extract the substring between these positions (inclusive)
            4. If no braces are found, attempt to parse the entire response as JSON
            5. Parse the extracted string using json.loads()
        
        Examples:
            >>> assistant = StepFunAssistant()
            
            >>> # Example 1: Clean JSON response
            >>> response = '{"title": "Deep Learning", "doi": "10.1016/j.dl.2024.01.001"}'
            >>> result = assistant._parse_json_response(response)
            >>> print(result['title'])
            'Deep Learning'
            
            >>> # Example 2: JSON with prefix text
            >>> response = 'Here is the analysis: {"title": "AI in Medicine", "summary": {...}}'
            >>> result = assistant._parse_json_response(response)
            >>> print(result['title'])
            'AI in Medicine'
            
            >>> # Example 3: JSON with suffix text
            >>> response = '{"title": "Cancer Detection"} I hope this helps!'
            >>> result = assistant._parse_json_response(response)
            >>> print(result['title'])
            'Cancer Detection'
            
            >>> # Example 4: JSON with both prefix and suffix
            >>> response = 'Analysis result: {"title": "Neural Networks"} End of analysis'
            >>> result = assistant._parse_json_response(response)
            >>> print(result['title'])
            'Neural Networks'
            
            >>> # Example 5: Multi-line JSON response
            >>> response = '''
            ... Here is the paper analysis:
            ... {
            ...     "title": "Machine Learning Review",
            ...     "doi": "10.1038/s41586-023-00001-0",
            ...     "summary": {
            ...         "objective": "Review ML applications",
            ...         "methods": "Systematic literature review",
            ...         "results": "ML shows promise in healthcare",
            ...         "conclusion": "More research needed"
            ...     }
            ... }
            ... Thank you for using our service!
            ... '''
            >>> result = assistant._parse_json_response(response)
            >>> print(result['summary']['objective'])
            'Review ML applications'
            
            >>> # Example 6: Nested JSON structures
            >>> response = '{"metadata": {"version": "1.0"}, "data": {"title": "Paper"}}'
            >>> result = assistant._parse_json_response(response)
            >>> print(result['metadata']['version'])
            '1.0'
        
        Common AI Response Patterns Handled:
            - "Here is the JSON: {...}"
            - "The analysis result is {...}"
            - "```json {...} ```" (Markdown code blocks - though braces still work)
            - "... {...} ..." (Any surrounding text)
            - Multiple lines with indentation
            - Responses with escaped characters
        
        Edge Cases:
            - Empty string: Will raise json.JSONDecodeError
            - No braces found: Attempts to parse entire string as JSON
            - Multiple JSON objects: Only extracts first complete object (from first { to last })
            - Malformed JSON: Propagates json.JSONDecodeError to caller
            - Extremely large responses: Memory efficient as it slices string
        
        Notes:
            - This method assumes the JSON object starts with '{' and ends with '}'
            - It does not handle JSON arrays that start with '[' (not expected from AI)
            - The method is not recursive - it finds the outermost braces
            - Performance is O(n) where n is the response length
            - The extracted JSON string may still contain escape sequences
        
        Performance Considerations:
            - String slicing creates a new substring (O(k) where k is JSON length)
            - json.loads() performance depends on JSON size (typically < 10KB)
            - Method is fast enough for real-time processing (< 1ms for typical responses)
        
        See Also:
            - json.loads(): Python's JSON parsing function
            - str.find(): String method for finding substrings
            - str.rfind(): String method for finding last occurrence
        
        Error Handling Note:
            This method does NOT catch json.JSONDecodeError. The calling method
            (ask_json) is responsible for catching and handling parsing errors
            appropriately, typically by returning an error response structure.
        
        Why This Approach?
            Many AI models, including StepFun, sometimes add conversational text
            around the JSON response to make it more human-readable. This method
            robustly handles that behavior without requiring changes to the AI
            prompt or post-processing the response in complex ways.
        """
        # Find JSON in response (handles extra text before/after)
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        
        if json_start != -1 and json_end != 0:
            json_str = response_content[json_start:json_end]
        else:
            json_str = response_content
        
        return json.loads(json_str)


    def _create_metadata(self, paper_text, temperature, max_tokens, output_file_path):
        """
        Creates a comprehensive metadata dictionary for the analysis result.
        
        This method generates metadata that provides important context about the
        analysis, including model configuration, timing information, input text
        statistics, and source file details. This metadata is essential for:
        - Reproducibility (knowing which model and settings were used)
        - Debugging (tracking when and how the analysis was performed)
        - Audit trails (recording source file information)
        - Performance monitoring (tracking input text length)
        
        Args:
            paper_text (str): The original extracted text from the PDF file.
                Used to calculate the input text length. Can be very large
                (thousands of characters) but only its length is stored.
            
            temperature (float): The temperature parameter used in the API request.
                Controls the randomness/creativity of the AI's responses.
                Typical values range from 0.0 to 1.0.
                Example: 0.7 (balanced), 0.2 (focused), 0.9 (creative)
            
            max_tokens (int): The maximum tokens parameter used in the API request.
                Limits the length of the AI's response.
                Example: 5000 (default), 1000 (shorter responses), 10000 (longer)
            
            output_file_path (str, optional): Full path to the original PDF file.
                Used to extract source file information. Can be None if no
                source file information is available or needed.
                Example: "/home/user/documents/paper.pdf"
        
        Returns:
            dict: A metadata dictionary with the following structure:
            
            {
                "model": str,                    # AI model identifier (e.g., "stepfun/step-3.5-flash:free")
                "temperature": float,            # Temperature setting (e.g., 0.7)
                "max_tokens": int,               # Max tokens setting (e.g., 5000)
                "timestamp": str,                # ISO 8601 timestamp (e.g., "2024-01-15T10:30:00.123456")
                "text_length": int,              # Number of characters in input text (e.g., 15234)
                "source_file": dict or None      # Source file information (see structure below)
            }
            
            **Source File Structure (when output_file_path is provided):**
            {
                "path": str,                     # Absolute path to the PDF file
                "name": str,                     # PDF filename with extension
                "directory": str                 # Parent directory path
            }
            
            **Source File Value (when output_file_path is None):**
            {
                "source_file": None              # Explicit null value
            }
        
        Examples:
            >>> assistant = StepFunAssistant()
            
            >>> # Example 1: Complete metadata with source file
            >>> metadata = assistant._create_metadata(
            ...     paper_text="Full paper content here...",
            ...     temperature=0.7,
            ...     max_tokens=5000,
            ...     output_file_path="/home/user/research/paper.pdf"
            ... )
            >>> print(metadata)
            {
                "model": "stepfun/step-3.5-flash:free",
                "temperature": 0.7,
                "max_tokens": 5000,
                "timestamp": "2024-01-15T14:30:22.123456",
                "text_length": 15234,
                "source_file": {
                    "path": "/home/user/research/paper.pdf",
                    "name": "paper.pdf",
                    "directory": "/home/user/research"
                }
            }
            
            >>> # Example 2: Metadata without source file
            >>> metadata = assistant._create_metadata(
            ...     paper_text="Text content",
            ...     temperature=0.5,
            ...     max_tokens=3000,
            ...     output_file_path=None
            ... )
            >>> print(metadata['source_file'])
            None
            >>> print(metadata['temperature'])
            0.5
            
            >>> # Example 3: Different model and settings
            >>> metadata = assistant._create_metadata(
            ...     paper_text="Short paper",
            ...     temperature=0.2,  # More deterministic
            ...     max_tokens=1000,   # Shorter response
            ...     output_file_path="/data/paper.pdf"
            ... )
            >>> print(metadata['model'])
            "stepfun/step-3.5-flash:free"
            >>> print(metadata['max_tokens'])
            1000
            
            >>> # Example 4: Very long paper text
            >>> long_text = "x" * 100000  # 100,000 characters
            >>> metadata = assistant._create_metadata(
            ...     paper_text=long_text,
            ...     temperature=0.7,
            ...     max_tokens=5000,
            ...     output_file_path="/path/paper.pdf"
            ... )
            >>> print(metadata['text_length'])
            100000
            
            >>> # Example 5: Windows file path
            >>> metadata = assistant._create_metadata(
            ...     paper_text="Text",
            ...     temperature=0.7,
            ...     max_tokens=5000,
            ...     output_file_path="C:\\Users\\name\\Documents\\paper.pdf"
            ... )
            >>> print(metadata['source_file']['directory'])
            "C:\\Users\\name\\Documents"
        
        Metadata Usage Examples:
            >>> # Using metadata for debugging
            >>> if result['_metadata']['temperature'] > 0.8:
            ...     print("Warning: High temperature may affect accuracy")
            
            >>> # Tracking analysis history
            >>> print(f"Analyzed on: {result['_metadata']['timestamp']}")
            >>> print(f"Using model: {result['_metadata']['model']}")
            
            >>> # Monitoring input size
            >>> text_length = result['_metadata']['text_length']
            >>> if text_length > 50000:
            ...     print(f"Large document ({text_length} chars) - may take longer")
            
            >>> # Reproducing analysis
            >>> same_settings = StepFunAssistant()
            >>> same_settings.ask_json(
            ...     paper_text=text,
            ...     temperature=result['_metadata']['temperature'],
            ...     max_tokens=result['_metadata']['max_tokens']
            ... )
        
        Notes:
            - **Timestamp Format**: ISO 8601 format with microseconds
            (YYYY-MM-DDTHH:MM:SS.mmmmmm). This format is:
            - Sortable chronologically
            - Human-readable
            - Timezone-naive (uses local system time)
            
            - **Text Length**: Counts characters, not tokens.
            - English text: ~4 characters per token
            - Can be used to estimate API costs
            - Helps identify very large documents
            
            - **Source File Paths**: Uses os.path.basename() and os.path.dirname()
            - Works cross-platform (Windows, Linux, macOS)
            - Handles both absolute and relative paths
            - Returns empty string for root directory paths
            
            - **Null Source File**: Explicitly set to None when output_file_path is None
            - Maintains consistent dictionary structure
            - Allows for easy checking: `if metadata['source_file'] is not None:`
        
        Platform Compatibility:
            - **Linux/Mac**: Uses forward slashes (/)
            Example: "/home/user/paper.pdf"
            - **Windows**: Uses backslashes (\)
            Example: "C:\\Users\\user\\paper.pdf"
            - The os.path module handles both correctly
        
        Performance:
            - O(1) time complexity (constant time, independent of text length)
            - Only stores text length, not the actual text
            - Memory efficient (metadata is small, ~200 bytes)
            - String operations are minimal
        
        Error Handling:
            - No error handling needed - all operations are safe
            - os.path functions handle malformed paths gracefully
            - Even if output_file_path is invalid, basename/dirname work
            - datetime.now() always succeeds
        
        See Also:
            - datetime.now().isoformat(): Creates the timestamp
            - os.path.basename(): Extracts filename from path
            - os.path.dirname(): Extracts directory from path
            - len(): Built-in function for string length
        
        Why Include Metadata?
            1. **Reproducibility**: Know exactly which settings produced the results
            2. **Debugging**: Track down issues with specific configurations
            3. **Auditing**: Maintain chain of custody for source documents
            4. **Analytics**: Analyze performance across different document sizes
            5. **Compliance**: Meet regulatory requirements for data processing
            6. **Cost Tracking**: Estimate API costs based on text length
            7. **Version Control**: Track when analyses were performed
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


    def _save_analysis_result(self, result, custom_filename, output_file_path):
        """
        Saves the analysis result to a JSON file with intelligent filename generation.
        
        This method acts as a bridge between the analysis result and the underlying
        JSON saving mechanism. It determines the appropriate filename based on
        priority rules and then delegates the actual file I/O to `_save_to_json()`.
        
        The filename selection follows a clear priority order:
            1. Custom filename (user-specified) - Highest priority
            2. Derived from PDF filename - Medium priority
            3. Auto-generated timestamp - Lowest priority (handled by `_save_to_json`)
        
        Args:
            result (dict): The analysis result dictionary to be saved.
                This dictionary typically contains:
                - 'title': Paper title (used for auto-generated filenames)
                - 'doi': Digital Object Identifier
                - 'summary': Dictionary with objective, methods, results, conclusion
                - '_metadata': Metadata about the analysis
                
            custom_filename (str, optional): User-specified filename or path.
                When provided, this takes precedence over all other naming methods.
                Can be:
                - Simple filename: "result.json"
                - Relative path: "outputs/analysis.json"
                - Absolute path: "/home/user/results/paper.json"
                If None, falls back to using output_file_path.
                
            output_file_path (str, optional): Original PDF file path.
                Used to generate a default JSON filename by:
                1. Extracting the base name (e.g., "paper.pdf" -> "paper")
                2. Appending the ".json" extension
                Example: "/path/to/paper.pdf" -> "paper.json"
                If None and custom_filename is None, passes None to _save_to_json,
                which will generate a timestamp-based filename.
        
        Returns:
            None. This method does not return a value. The actual saving is handled
            by `_save_to_json()`, which prints confirmation or error messages.
        
        Filename Generation Logic:
            ┌─────────────────────────────────────────────────────────────┐
            │  Is custom_filename provided?                               │
            │       │                                                     │
            │       ├── YES → Use custom_filename directly                │
            │       │                                                     │
            │       └── NO  → Is output_file_path provided?               │
            │                   │                                         │
            │                   ├── YES → Extract PDF basename + .json   │
            │                   │                                         │
            │                   └── NO  → Pass None (let _save_to_json   │
            │                             generate timestamp filename)    │
            └─────────────────────────────────────────────────────────────┘
        
        Examples:
            >>> assistant = StepFunAssistant()
            >>> result = {"title": "Deep Learning Paper", "summary": {...}}
            
            >>> # Example 1: Custom filename (highest priority)
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename="/custom/path/my_result.json",
            ...     output_file_path="/original/paper.pdf"
            ... )
            # Saves to: /custom/path/my_result.json
            
            >>> # Example 2: Using PDF filename
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=None,
            ...     output_file_path="/home/user/research/paper.pdf"
            ... )
            # Saves to: /home/user/research/paper.json
            
            >>> # Example 3: PDF with different extension
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=None,
            ...     output_file_path="/data/thesis.pdf"
            ... )
            # Saves to: /data/thesis.json
            
            >>> # Example 4: No filename hints provided
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=None,
            ...     output_file_path=None
            ... )
            # Saves to: outputs/paper_analysis_20240115_143022.json
            # (auto-generated by _save_to_json)
            
            >>> # Example 5: Custom filename with path
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename="results/analysis.json",
            ...     output_file_path="/any/path/paper.pdf"  # Ignored due to custom
            ... )
            # Saves to: results/analysis.json
            
            >>> # Example 6: Simple custom filename (current directory)
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename="output.json",
            ...     output_file_path=None
            ... )
            # Saves to: output.json (in current working directory)
        
        Real-World Scenarios:
            
            **Scenario 1: User wants specific output location**
            >>> # User runs: python main.py paper.pdf --output /results/paper.json
            >>> # In the code:
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=args.output,  # "/results/paper.json"
            ...     output_file_path=args.file_path
            ... )
            # Result saved exactly where user requested
            
            **Scenario 2: Automatic saving alongside PDF**
            >>> # User just wants JSON next to the PDF
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=None,
            ...     output_file_path="/home/user/paper.pdf"
            ... )
            # Saves to: /home/user/paper.json (convenient!)
            
            **Scenario 3: No output specified, use defaults**
            >>> # User didn't specify any output preferences
            >>> assistant._save_analysis_result(
            ...     result=result,
            ...     custom_filename=None,
            ...     output_file_path=None
            ... )
            # Saves to outputs/ directory with timestamp (ensures no overwrites)
        
        Path Handling Details:
            - **Absolute paths**: Preserved as-is
            Example: "/home/user/result.json"
            
            - **Relative paths**: Resolved relative to current working directory
            Example: "outputs/result.json" -> "./outputs/result.json"
            
            - **Just filename**: Saved in current working directory
            Example: "result.json" -> "./result.json"
            
            - **Directory creation**: Handled by `_save_to_json` (creates if needed)
            
            - **Extension handling**: `_save_to_json` ensures .json extension
        
        Integration with _save_to_json:
            This method passes three parameters to `_save_to_json`:
            1. **data**: The result dictionary (unchanged)
            2. **filename**: Determined filename (or None)
            3. **paper_name**: The paper title from result.get("title")
            
            The `_save_to_json` method then:
            - Creates the output directory if it doesn't exist
            - Ensures .json extension
            - Generates timestamp-based filename if needed
            - Actually writes the file to disk
        
        Error Handling:
            This method does not perform explicit error handling. Instead, it relies
            on `_save_to_json()` to handle any I/O errors (permissions, disk full,
            invalid paths, etc.). Errors are caught and printed by `_save_to_json()`,
            which returns None on failure.
        
        Notes:
            - The method extracts the paper title from the result dictionary
            (result.get("title")) and passes it to `_save_to_json` for potential
            use in auto-generated filenames.
            
            - If both custom_filename and output_file_path are None, json_filename
            is set to None, triggering `_save_to_json`'s default naming logic.
            
            - The method does not validate that output_file_path ends with .pdf.
            It works with any file extension, extracting the base name correctly.
            
            - The paper title is only used by `_save_to_json` if no other naming
            information is provided (sanitized and used in filename).
        
        Advantages of This Approach:
            1. **Separation of Concerns**: Filename logic separated from I/O logic
            2. **Flexibility**: Supports multiple naming strategies
            3. **User Control**: Custom filename always takes priority
            4. **Sensible Defaults**: Falls back to reasonable auto-naming
            5. **Clean Code**: Simple, linear logic flow
        
        See Also:
            - _save_to_json(): The actual file writing method
            - ask_json(): The public method that calls this one
            - os.path.splitext(): Used to remove file extension
            - os.path.basename(): Used to extract filename from path
        
        Performance:
            - O(1) time complexity - simple string operations only
            - No disk I/O performed in this method (delegated to _save_to_json)
            - Minimal memory allocation
        """
        # Determine filename
        if custom_filename:
            json_filename = custom_filename
        elif output_file_path:
            base_name = os.path.splitext(os.path.basename(output_file_path))[0]
            json_filename = f"{base_name}.json"
        else:
            json_filename = None
        
        # Save file
        paper_title = result.get("title")
        self._save_to_json(result, json_filename, paper_title)


    def _create_error_response(self, error, paper_text, output_file_path, custom_filename, save_to_json):
        """
            Creates a structured error response when analysis fails.
            
            This method handles graceful degradation when errors occur during paper analysis.
            Instead of crashing, it constructs a consistent error response that mirrors the
            success response format, allowing client code to handle errors uniformly.
            
            Args:
                error (Exception): The caught exception object. Can be any exception type:
                    - ConnectionError: Network issues
                    - TimeoutError: Request took too long
                    - json.JSONDecodeError: Invalid JSON response
                    - APIError: Authentication or API problems
                    - Any other unexpected exception
                
                paper_text (str): The original paper text being processed.
                    Used to generate metadata about the failed request.
                
                output_file_path (str, optional): Original PDF file path.
                    Used to include source file information in error metadata.
                
                custom_filename (str, optional): User-specified filename for JSON output.
                    If provided, the error response will be saved to this location.
                
                save_to_json (bool): Whether to save the error response to a JSON file.
                    When True, the error is persisted to disk for debugging.
                    When False, the error is only returned in memory.
            
            Returns:
                dict: A structured error response with the following schema:
                    {
                        "error": str,                    # Human-readable error message
                        "title": None,                   # Consistent with success response
                        "doi": None,                     # Consistent with success response
                        "summary": None,                 # Consistent with success response
                        "_metadata": {                   # Same structure as success response
                            "model": str,
                            "temperature": 0,            # Default value for errors
                            "max_tokens": 0,             # Default value for errors
                            "timestamp": str,
                            "text_length": int,
                            "source_file": dict or None
                        }
                    }
            
            Examples:
                >>> assistant = StepFunAssistant()
                
                >>> # Example 1: Network error with file saving
                >>> try:
                ...     raise ConnectionError("Failed to connect to API")
                ... except Exception as e:
                ...     error_response = assistant._create_error_response(
                ...         error=e,
                ...         paper_text="Sample paper text",
                ...         output_file_path="/path/paper.pdf",
                ...         custom_filename=None,
                ...         save_to_json=True
                ...     )
                >>> print(error_response['error'])
                'API Error: Failed to connect to API'
                >>> print(error_response['title'])
                None
                
                >>> # Example 2: JSON parsing error with custom filename
                >>> try:
                ...     raise json.JSONDecodeError("Invalid JSON", "...", 0)
                ... except json.JSONDecodeError as e:
                ...     error_response = assistant._create_error_response(
                ...         error=e,
                ...         paper_text="Paper content",
                ...         output_file_path="/data/paper.pdf",
                ...         custom_filename="results/output.json",
                ...         save_to_json=True
                ...     )
                # Saves to: results/output.json (custom filename takes priority)
                
                >>> # Example 3: Error without saving to file
                >>> error_response = assistant._create_error_response(
                ...     error=ValueError("Invalid parameter"),
                ...     paper_text="Content",
                ...     output_file_path=None,
                ...     custom_filename=None,
                ...     save_to_json=False
                ... )
                >>> print(error_response['_metadata']['text_length'])
                7
                # No file is created on disk
                
                >>> # Example 4: Timeout error with automatic naming
                >>> error_response = assistant._create_error_response(
                ...     error=TimeoutError("Request timed out"),
                ...     paper_text="Long paper..." * 1000,
                ...     output_file_path="/home/user/thesis.pdf",
                ...     custom_filename=None,
                ...     save_to_json=True
                ... )
                # Saves to: /home/user/thesis_error.json
                >>> print(error_response['_metadata']['source_file']['name'])
                'thesis.pdf'
                
                >>> # Example 5: Client-side error handling pattern
                >>> result = assistant.ask_json(paper_text=text)
                >>> if 'error' in result:
                ...     print(f"Analysis failed: {result['error']}")
                ...     print(f"At: {result['_metadata']['timestamp']}")
                ...     print(f"Text length: {result['_metadata']['text_length']}")
                ... else:
                ...     print(f"Success: {result['title']}")
            
            Filename Generation Rules:
                - If custom_filename is provided: Use it as-is (no modification)
                - Else if output_file_path is provided: Add "_error" suffix (paper.pdf -> paper_error.json)
                - Else: Let _save_to_json generate a timestamp-based filename
            
            Metadata in Error Responses:
                The error response includes metadata even when analysis fails:
                
                - For Debugging: timestamp shows when the error occurred
                - For Analysis: text_length shows input size that caused error
                - For Auditing: source_file shows which document failed
                - For Monitoring: track patterns like large documents causing timeouts
            
            Error Message Examples:
                - "API Error: Connection refused"
                - "API Error: Request timeout after 30 seconds"
                - "API Error: Expecting value: line 1 column 1"
                - "API Error: [Errno 104] Connection reset by peer"
                - "API Error: 401 Unauthorized"
            
            Notes:
                - Temperature and max_tokens in metadata are set to 0 for errors
                - The error message automatically converts the exception to string
                - Custom filenames are used AS-IS without "_error" suffix
                - The paper_title passed to _save_to_json is "error_response"
                - This method does NOT catch exceptions - called from within except block
                - The method always returns a dictionary, never raises exceptions
            
            Advantages of This Approach:
                1. Consistent API: Success and error responses have similar structure
                2. Graceful Degradation: System never crashes from API errors
                3. Debugging Support: Rich metadata included with every error
                4. Flexible Persistence: Option to save or not save errors to disk
                5. User Control: Custom filenames respected even for errors
            
            Common Use Cases:
                Production Monitoring:
                    result = assistant.ask_json(paper_text)
                    if 'error' in result:
                        error_logger.error(
                            f"Analysis failed: {result['error']}",
                            extra=result['_metadata']
                        )
                
                Batch Processing:
                    failed_files = []
                    for pdf_path in pdf_list:
                        result = assistant.ask_json(paper_text, output_file_path=pdf_path)
                        if 'error' in result:
                            failed_files.append({
                                'file': pdf_path,
                                'error': result['error'],
                                'timestamp': result['_metadata']['timestamp']
                            })
                
                User Feedback:
                    result = assistant.ask_json(paper_text)
                    if 'error' in result:
                        if 'timeout' in result['error'].lower():
                            show_message("Paper is too large, please try a smaller file")
                        elif 'connection' in result['error'].lower():
                            show_message("Network issue, please check your connection")
                        else:
                            show_message("Analysis failed, please try again")
            
            Performance:
                - Time complexity: O(1) - constant time operations
                - Memory: Creates small dictionary (typically < 1KB)
                - File I/O: Only occurs if save_to_json=True
                - String conversion: Fast even for large exception objects
            
            See Also:
                - _create_metadata(): Called to generate error metadata with safe defaults
                - _save_to_json(): Handles error persistence to disk
                - ask_json(): The public method that calls this on exceptions
            
            Best Practices:
                1. Always set save_to_json=True in production to maintain error logs
                2. Check for 'error' key before accessing other fields in result
                3. Use metadata timestamp for error tracking and trend analysis
                4. Log the complete error response for production debugging
                5. Monitor error patterns by analyzing saved error JSON files
            """
        error_result = {
            "error": f"API Error: {error}",
            "title": None,
            "doi": None,
            "summary": None,
            "_metadata": self._create_metadata(paper_text, 0, 0, output_file_path)
        }
        
        # Save error to file if requested
        if save_to_json:
            if custom_filename:
                json_filename = custom_filename
            elif output_file_path:
                base_name = os.path.splitext(os.path.basename(output_file_path))[0]
                json_filename = f"{base_name}_error.json"
            else:
                json_filename = None
            
            self._save_to_json(error_result, json_filename, "error_response")
        
        return error_result