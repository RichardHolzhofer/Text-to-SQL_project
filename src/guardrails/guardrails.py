import re
from functools import lru_cache

from llm_guard.input_scanners import Anonymize, PromptInjection
from llm_guard.input_scanners.anonymize import DEFAULT_ENTITY_TYPES
from llm_guard.input_scanners.anonymize_helpers.regex_patterns import (
    DEFAULT_REGEX_PATTERNS,
)
from llm_guard.output_scanners import Deanonymize
from llm_guard.vault import Vault

from src.exceptions.exception import (
    GuardrailException,
    GuardrailInputError,
    GuardrailOutputError,
)
from src.logger.logger import logger


class TextToSQLGuardrails:
    def __init__(self):
        try:
            logger.info("Initializing TextToSQLGuardrails (Loading models...)")

            # Initialize Vault to store original vs anonymized mappings
            self.vault = Vault()

            custom_patterns = DEFAULT_REGEX_PATTERNS.copy()
            custom_patterns.append(
                {
                    "name": "OLIST_ID",
                    "expressions": [r"\b[a-f0-9]{32}\b"],
                    "context": [
                        "id",
                        "customer",
                        "order",
                        "product",
                        "seller",
                        "unique",
                    ],
                    "score": 1.0,
                    "languages": ["en"],
                }
            )

            # Ensure OLIST_ID is in the entity_types list
            entity_types = DEFAULT_ENTITY_TYPES + ["OLIST_ID"]

            # Whitelist Brazilian state codes and common terms to prevent false positives
            brazilian_states = [
                "AC",
                "AL",
                "AP",
                "AM",
                "BA",
                "CE",
                "DF",
                "ES",
                "GO",
                "MA",
                "MT",
                "MS",
                "MG",
                "PA",
                "PB",
                "PR",
                "PE",
                "PI",
                "RJ",
                "RN",
                "RS",
                "RO",
                "RR",
                "SC",
                "SP",
                "SE",
                "TO",
            ]
            # Include lowercase versions too
            allowed_names = brazilian_states + [s.lower() for s in brazilian_states]

            # 1. Input Scanners
            self.input_scanners = [
                Anonymize(
                    vault=self.vault,
                    regex_patterns=custom_patterns,
                    use_onnx=True,
                    entity_types=entity_types,
                    allowed_names=allowed_names,
                ),
                PromptInjection(use_onnx=True, threshold=0.9),
            ]

            # 2. Output Scanners
            self.output_scanners = [
                Deanonymize(vault=self.vault),
            ]

            logger.info("TextToSQLGuardrails initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing TextToSQLGuardrails: {e}")
            raise GuardrailException(e) from e

    def scan_user_input(self, text: str) -> str:
        """
        Scans and sanitizes user input before it reaches the LLM.
        """
        try:
            sanitized_text = text
            for scanner in self.input_scanners:
                # 1. We must run Anonymize first to redact raw PII and IDs.
                # 2. However, the resulting bracketed tokens like `[REDACTED_...]` are highly
                #    sensitive and flag as false positives in the PromptInjection model (which
                #    associates bracketed capital phrases with adversarial template injections).
                # 3. Therefore, we temporarily substitute these tokens with a neutral placeholder
                #    ("VALUE") before running the PromptInjection check.
                text_to_scan = sanitized_text
                if isinstance(scanner, PromptInjection):
                    text_to_scan = re.sub(
                        r"\[REDACTED_.*?_\d+\]", "VALUE", sanitized_text
                    )

                scan_res_text, is_valid, risk_score = scanner.scan(text_to_scan)

                # We always keep the sanitized_text from Anonymize, but we use the
                # is_valid/risk_score from the scanner (which ran on the cleaned text).
                if isinstance(scanner, Anonymize):
                    sanitized_text = scan_res_text

                if not is_valid:
                    logger.warning(
                        f"Input guardrail '{type(scanner).__name__}' flagged the input. Risk score: {risk_score}"
                    )
                    # For Anonymize, we continue even if is_valid is False (it just means it redacted something)
                    # For other scanners (PromptInjection), we block.
                    if not isinstance(scanner, Anonymize):
                        return "I'm sorry, but your request was flagged for safety reasons. Please rephrase."
            return sanitized_text
        except Exception as e:
            logger.error(f"Error in input guardrail: {e}")
            raise GuardrailInputError(e) from e

    def scan_history(self, history: list) -> list:
        """
        Scans and redacts PII from the chat history.
        This ensures that even if real IDs were saved to the DB for the user,
        the LLM only sees the anonymized versions in subsequent turns.
        """
        if not history:
            return history

        try:
            # We only use the Anonymizer for history (not PromptInjection)
            # because history was already validated in previous turns.
            anonymizer = self.input_scanners[0]

            sanitized_history = []
            for msg in history:
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    # Redact the content
                    sanitized_content, _, _ = anonymizer.scan(msg.content)
                    # Create a copy of the message with sanitized content
                    msg = msg.model_copy(update={"content": sanitized_content})
                sanitized_history.append(msg)
            return sanitized_history
        except Exception as e:
            logger.error(f"Error in history guardrail: {e}")
            return history

    def scan_data(self, data: list[dict]) -> list[dict]:
        """
        Scans and anonymizes database results before they are passed to the LLM.
        Ensures the LLM doesn't see real IDs in the context.
        """
        if not data:
            return data

        try:
            # We use the same Anonymize scanner to keep Vault mappings consistent
            anonymizer = self.input_scanners[0]

            sanitized_data = []
            for row in data:
                new_row = {}
                for key, value in row.items():
                    if isinstance(value, str):
                        # Only scan if it's a string
                        sanitized_val, _, _ = anonymizer.scan(value)
                        new_row[key] = sanitized_val
                    else:
                        new_row[key] = value
                sanitized_data.append(new_row)
            return sanitized_data
        except Exception as e:
            logger.error(f"Error in data guardrail: {e}")
            # In case of error, we return an empty list or redacted info to be safe
            return [{"info": "[DATA_REDACTED_DUE_TO_SECURITY_ERROR]"}]

    def scan_llm_output(self, prompt: str, text: str) -> str:
        """
        Scans and redacts LLM output before it is returned to the user.
        """
        try:
            sanitized_text = text
            for scanner in self.output_scanners:
                sanitized_text, is_valid, risk_score = scanner.scan(
                    prompt, sanitized_text
                )
                if not is_valid:
                    logger.warning(
                        f"Output guardrail '{type(scanner).__name__}' flagged a response. Risk score: {risk_score}"
                    )
            return sanitized_text
        except Exception as e:
            logger.error(f"Error in output guardrail: {e}")
            raise GuardrailOutputError(e) from e


@lru_cache(maxsize=1)
def get_guardrails() -> TextToSQLGuardrails:
    """
    Returns a cached instance of the guardrails.
    Models are loaded only the first time this is called.
    """
    try:
        return TextToSQLGuardrails()
    except Exception as e:
        logger.error(f"Failed to load or retrieve TextToSQLGuardrails: {e}")
        raise GuardrailException(e) from e
