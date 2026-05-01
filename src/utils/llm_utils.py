from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.utils.utils import load_prompt
from src.logger.logger import logger


def convert_to_messages(base_messages: list):
    """
    Helper to convert tuple-based prompts into LangChain Message objects.
    Expects a list of tuples: (role, content)
    """
    langchain_messages = []
    for role, content in base_messages:
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    return langchain_messages


def generate_conversation_title(config, question: str) -> str:
    """
    Generates a concise title for the conversation using the fast LLM from config.
    """
    try:
        # Load and render the prompt
        raw_prompt = load_prompt(
            "src/prompts/generate_title.yaml", {"question": question}
        )

        # Convert to LangChain messages
        messages = convert_to_messages(raw_prompt)

        # Get the fast LLM instance from config
        llm = config.get_fast_llm()

        # Invoke and clean response
        response = llm.invoke(messages)
        title = response.content.strip().replace('"', "")

        logger.info(f"Generated conversation title: '{title}'")

        return title
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        # Fallback to truncated question
        return f"{question[:25]}..."
