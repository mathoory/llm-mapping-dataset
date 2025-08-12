from google import genai
from google.genai import types

class LLM:
    def __init__(self, type="gemini-2.5-flash", api_key_path="key.secret"):
        self.type = type
        self.api_key_path = api_key_path
        self.client = self._create_client()

    def _create_client(self):
        with open(self.api_key_path, "r") as f:
            api_key = f.read().strip()
        return genai.Client(api_key=api_key)

    def query(self, input):
        response = self.client.models.generate_content(
            model=self.type,
            contents=input,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            ),
        )

        return response.text