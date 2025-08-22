import time
import json
import logging
from google import genai
from google.genai import types


class LLM:
    # Rate and global limits per model
    RATE_LIMITS = {
        "gemini-2.5-flash": {"per_minute": 10, "global": 250},
        "gemini-2.5-pro": {"per_minute": 5, "global": 100},
    }

    # Instance state for limits
    MODEL_ALIASES = {
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }

    THINKING_BUDGETS = {
        "gemini-2.5-flash": 0,    # Disable thinking for flash
        "gemini-2.5-pro": 128,    # Minimum possible value for pro
    }

    def __init__(self, model="gemini-2.5-flash", api_key_path="key.secret"):
        # Map alias to actual model name if present
        self.model = self.MODEL_ALIASES.get(model, model)
        self.api_key_path = api_key_path
        self.client = self._create_client()
        # Set thinking_budget using dict, fallback to 0
        self.thinking_budget = self.THINKING_BUDGETS.get(self.model, 0)
        self.rate_limit = self.RATE_LIMITS.get(self.model, {"per_minute": float("inf"), "global": float("inf")})

    # ...existing code...

    def _create_client(self):
        with open(self.api_key_path, "r") as f:
            api_key = f.read().strip()
        return genai.Client(api_key=api_key)



    def query(self, input, parse=False):
        # Single query for compatibility
        return self.query_batch([input], parse=parse)[0]

    def query_batch(self, prompts, parse=False):
        # Check global limit
        total = len(prompts)
        if total > self.rate_limit["global"]:
            raise RuntimeError(f"Global query limit exceeded for model {self.model} ({self.rate_limit['global']})")

        batch_size = self.rate_limit["per_minute"]
        results = []
        for i in range(0, total, batch_size):
            batch = prompts[i:i+batch_size]
            batch_results = []
            for prompt in batch:
                logging.info(f"Sending prompt to model")
                from google.genai.errors import ServerError, ClientError
                for attempt in range(3):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                            ),
                        )
                        logging.debug(f"Received response:\n{response}")
                        text = getattr(response, "text", "")
                        if parse:
                            batch_results.append(self.parse_response(text))
                        else:
                            batch_results.append(text)
                        break
                    except ServerError as e:
                        status_code = getattr(e, 'status_code', None)
                        response_json = getattr(e, 'response_json', None)
                        logging.error(f"ServerError (HTTP 500) from model: {e}. Status code: {status_code}, Response JSON: {response_json}. Attempt {attempt+1}/3")
                        if attempt == 2:
                            logging.error(f"All 3 retries failed for prompt, skipping this prompt.")
                            batch_results.append(None)
                        else:
                            time.sleep(2)
                    except ClientError as e:
                        status_code = getattr(e, 'status_code', None)
                        response_json = getattr(e, 'response_json', None)
                        # Try to parse status code from exception string if not present
                        if status_code is None:
                            import re
                            match = re.search(r'(\b\d{3}\b)', str(e))
                            if match:
                                status_code = int(match.group(1))
                        if status_code == 429 or 'RESOURCE_EXHAUSTED' in str(e):
                            logging.error(f"ClientError (HTTP 429) RESOURCE_EXHAUSTED: {e}. Status code: {status_code}, Response JSON: {response_json}. Waiting 60 seconds before retry.")
                            time.sleep(60)
                        else:
                            logging.error(f"ClientError from model: {e}. Status code: {status_code}, Response JSON: {response_json}. Attempt {attempt+1}/3")
                            if attempt == 2:
                                logging.error(f"All 3 retries failed for prompt, skipping this prompt.")
                                batch_results.append(None)
                            else:
                                time.sleep(2)
            results.extend(batch_results)
            if i + batch_size < total:
                logging.info(f"Rate limit reached for model {self.model}: sleeping for 60 seconds...")
                time.sleep(60)
        return results

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