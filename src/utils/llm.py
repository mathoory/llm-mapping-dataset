import time
import json
from tqdm import tqdm
## Remove logging import
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError


class LLM:
    LOG_LEVELS = ["DEBUG", "INFO"]
    log_level = "INFO"

    @classmethod
    def set_log_level(cls, level):
        if level in cls.LOG_LEVELS:
            cls.log_level = level
        else:
            raise ValueError(f"Invalid log level: {level}")

    @classmethod
    def log(cls, msg, level="INFO"):
        levels = {"DEBUG": 0, "INFO": 1}
        if levels[level] >= levels[cls.log_level]:
            # Print a newline to avoid overwriting tqdm progress bar
            print()
            print(f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}")
    # Rate and global limits per model
    RATE_LIMITS = {
        "gemini-2.5-flash": {"per_minute": 10, "global": 250},
        "gemini-2.5-pro": {"per_minute": 2, "global": 50},
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

    def _create_client(self):
        with open(self.api_key_path, "r") as f:
            api_key = f.read().strip()
        return genai.Client(api_key=api_key)



    def query(self, input, parse=False):
        # Single query for compatibility
        return self.query_batch([input], parse=parse)[0]

    def query_batch(self, prompts, parse=False):
        # This function is a generator: it yields each result as soon as it's ready.
        # Using a generator allows partial results to be processed and saved even if interrupted (e.g., Ctrl+C).
        # Check global limit
        total = len(prompts)
        if total > self.rate_limit["global"]:
            raise RuntimeError(f"Global query limit exceeded for model {self.model} ({self.rate_limit['global']})")

        per_minute = self.rate_limit["per_minute"]
        queries_left = per_minute
        for idx, prompt in enumerate(tqdm(prompts, desc="Querying", unit="prompt")):
            self.log(f"Sending prompt to model", "DEBUG")
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                        ),
                    )
                    #time.sleep(0.5)

                    if getattr(response, "text", None) is None:
                        self.log("Received response.text=None, retrying...", "INFO")
                        continue  # retry without decrementing queries_left

                    self.log(f"Received response:\n{response.text}", "DEBUG")
                    text = response.text
                    if parse:
                        yield self.parse_response(text)
                    else:
                        yield text
                    queries_left -= 1
                    break
                except ServerError as e:
                    self._handle_server_error(e, attempt)
                    if attempt == 2:
                        queries_left -= 1
                        # Do not yield, just continue to next prompt
                except ClientError as e:
                    self._handle_client_error(e, attempt)
                    queries_left = per_minute
            # Rate limit check after each prompt
            if queries_left == 0 and idx < total - 1:
                self.log(f"Rate limit reached for model {self.model}: sleeping for 60 seconds...", "INFO")
                time.sleep(60)
                queries_left = per_minute

    def _handle_server_error(self, e, attempt):
        status_code = getattr(e, 'status_code', None)
        response_json = getattr(e, 'response_json', None)
        self.log(f"ServerError (HTTP 500) from model: {e}. Status code: {status_code}, Response JSON: {response_json}. Attempt {attempt+1}/3", "INFO")
        if attempt == 2:
            self.log(f"All 3 retries failed for prompt, skipping this prompt.", "INFO")
        else:
            time.sleep(2)

    def _handle_client_error(self, e, attempt):
        s = str(e)
        if 'PerMinute' in s:
            self.log("Minute quota exceeded (429). Waiting 60 seconds before retry.", "INFO")
            time.sleep(60)
            return
        elif 'PerDay' in s:
            self.log("Day quota exceeded (429). Aborting further requests.", "INFO")
            raise e
        else:
            self.log(f"ClientError from model: {e}. Attempt {attempt+1}/3", "INFO")
            raise e

    def parse_response(self, response):
        # Extract JSON block by slicing from first '{' to last '}'
        start = response.find('{')
        end = response.rfind('}')
        error = None
        extracted_output = ""
        confidence_score = None
        if start != -1 and end != -1 and end > start:
            try:
                response_json = json.loads(response[start:end+1])
                extracted_output = response_json.get("answer", "").strip()
                # Remove leading/trailing curly brackets if present
                if extracted_output.startswith('{') and extracted_output.endswith('}'):
                    extracted_output = extracted_output[1:-1].strip()
                confidence_score = response_json.get("confidence_score", None)
            except Exception as e:
                error = f"json_error"
        else:
            error = "structure_error"
        return {"output": extracted_output, "confidence": confidence_score, "error": error}