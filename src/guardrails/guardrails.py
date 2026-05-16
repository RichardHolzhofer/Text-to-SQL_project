import threading

from llm_guard import scan_input, scan_output
from llm_guard.input_scanners import Anonymize as InputAnonymize
from llm_guard.input_scanners import PromptInjection
from llm_guard.output_scanners import Anonymize as OutputAnonymize
from llm_guard.output_scanners import Regex

from src.logger.logger import logger


class TextToSQLGuardrails:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TextToSQLGuardrails, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing TextToSQLGuardrails (Loading models...)")

        # 1. Input Scanners
        self.input_scanners = [
            InputAnonymize(),
            PromptInjection(),
        ]

        # 2. Output Scanners
        # Regex for UUID/ID masking (common Olist ID formats)
        id_regex = r"[a-f0-9]{32}|[a-f0-9-]{36}"

        self.output_scanners = [
            OutputAnonymize(),
            Regex(patterns=[id_regex], redact=True),
        ]

        self._initialized = True
        logger.info("TextToSQLGuardrails initialized successfully.")

    def scan_user_input(self, text: str) -> str:
        """
        Scans and sanitizes user input before it reaches the LLM.
        """
        try:
            sanitized_text, is_valid, risk_score = scan_input(self.input_scanners, text)
            if not is_valid:
                logger.warning(
                    f"Input guardrail blocked a request. Risk score: {risk_score}"
                )
                return "I'm sorry, but your request was flagged for safety reasons. Please rephrase."
            return sanitized_text
        except Exception as e:
            logger.error(f"Error in input guardrail: {e}")
            return text  # Fallback to original text on error

    def scan_llm_output(self, prompt: str, text: str) -> str:
        """
        Scans and redacts LLM output before it is returned to the user.
        """
        try:
            sanitized_text, is_valid, risk_score = scan_output(
                self.output_scanners, prompt, text
            )
            if not is_valid:
                logger.warning(
                    f"Output guardrail flagged a response. Risk score: {risk_score}"
                )
                # We return the sanitized text anyway if it's just redaction,
                # but we log the risk.
            return sanitized_text
        except Exception as e:
            logger.error(f"Error in output guardrail: {e}")
            return text
