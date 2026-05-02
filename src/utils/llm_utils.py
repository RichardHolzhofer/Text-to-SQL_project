from src.logger.logger import logger
from src.utils.utils import get_prompt_template


def run_prompt(
    prompt_name,
    variables,
    config,
    chat_history=None,
    output_schema=None,
    use_fast_llm=False,
):
    """
    Unified helper to load a prompt template, bind the correct LLM (with overrides),
    handle conversational history, and execute the chain with tracing.
    """
    # 1. Load native template
    template = get_prompt_template(prompt_name, config)
    prompt_config = template.metadata.get("config", {})

    # 2. Select LLM (Smart vs Fast vs Override)
    model_override = prompt_config.get("model")
    if model_override:
        llm = config.get_llm(model_override)
    else:
        llm = config.get_fast_llm() if use_fast_llm else config.get_smart_llm()

    # 3. Bind Parameters (Temperature, etc.)
    if "temperature" in prompt_config:
        llm = llm.bind(temperature=float(prompt_config["temperature"]))

    # 4. Handle Structured Output
    if output_schema:
        llm = llm.with_structured_output(output_schema)

    # 5. Handle History
    if chat_history:
        # We extend the template's internal message list
        template.messages.extend(chat_history)

    # 6. Execute Chain
    chain = template | llm
    return chain.invoke(
        variables,
        config={
            "metadata": {"langfuse_prompt": template.metadata.get("langfuse_prompt")}
        },
    )


def generate_conversation_title(config, question: str) -> str:
    """
    Generates a concise title for the conversation using the fast LLM from config.
    """
    try:
        response = run_prompt(
            prompt_name="generate_title",
            variables={"question": question},
            config=config,
            use_fast_llm=True,
        )
        # Handle both AIMessage and raw content
        content = response.content if hasattr(response, "content") else str(response)
        title = content.strip().replace('"', "")

        logger.info(f"Generated conversation title: '{title}'")
        return title
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return f"{question[:25]}..."
