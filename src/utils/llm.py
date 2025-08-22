import json
from google import genai
from google.genai import types


class LLM:
    MODEL_ALIASES = {
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }

    def __init__(self, model="gemini-2.5-flash", api_key_path="key.secret"):
        # Map alias to actual model name if present
        self.model = self.MODEL_ALIASES.get(model, model)
        self.api_key_path = api_key_path
        self.client = self._create_client()

    def _create_client(self):
        with open(self.api_key_path, "r") as f:
            api_key = f.read().strip()
        return genai.Client(api_key=api_key)

    def query(self, input, parse=False):
        response = self.client.models.generate_content(
            model=self.model,
            contents=input,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            ),
        )

        if parse:
            return self.parse_response(response.text)
        
        return response.text

    def parse_response(self, response):
        # Extract JSON block by slicing from first '{' to last '}'
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                response_json = json.loads(response[start:end+1])
                extracted_output = response_json.get("answer", "").strip()
                # Remove leading/trailing curly brackets if present
                if extracted_output.startswith('{') and extracted_output.endswith('}'):
                    extracted_output = extracted_output[1:-1].strip()
                confidence_score = response_json.get("confidence_score", None)
            except Exception:
                print(f"Error parsing response: {response}")
                extracted_output = ""
                confidence_score = None
        else:
            extracted_output = ""
            confidence_score = None

        return extracted_output, confidence_score