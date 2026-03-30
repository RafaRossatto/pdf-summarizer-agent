import os
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
            raise ValueError(" Error: MY_KEY not found in .env file")
        if not self.base_url:
            raise ValueError(" Error: BASE_URL not found in .env file")

    def _initialize_client(self):
        """Creates the OpenAI client instance."""
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            print(f" Client initialized successfully using model: {self.model}")
            return client
        except Exception as e:
            raise RuntimeError(f" Failed to initialize client: {e}")

    def ask(self, question, temperature=0.7, max_tokens=5000):
        """Sends a prompt to the model and returns the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise responses."},
                    {"role": "user", "content": question}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {e}"