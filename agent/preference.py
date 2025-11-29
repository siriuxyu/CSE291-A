import time
from typing import Dict, Any, cast

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from agent.context import Context
from agent.state import State
from agent.utils import load_chat_model, get_logger
from agent.schemas import UserProfileUpdate
from agent.prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT

logger = get_logger(__name__)


async def extract_preferences(
        state: State, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """
    LangGraph Node: Analyzes the conversation to extract user preferences.
    """
    start_time = time.time()

    # prepare model
    model = load_chat_model(runtime.context.model)
    structured_llm = model.with_structured_output(UserProfileUpdate)

    # 2. prepare recent Messages
    recent_messages = state.messages[-10:]

    messages_for_extraction = [
        SystemMessage(content=PREFERENCE_EXTRACTION_SYSTEM_PROMPT),
        *recent_messages
    ]

    logger.info("Starting preference extraction", extra={
        'function': 'extract_preferences',
        'details': {'message_count': len(recent_messages)}
    })

    try:
        result = await structured_llm.ainvoke(messages_for_extraction)
        result = cast(UserProfileUpdate, result)

        duration = (time.time() - start_time) * 1000

        if result.extracted_preferences:
            logger.info("Preferences extracted", extra={
                'function': 'extract_preferences',
                'duration_ms': round(duration, 2),
                'details': {
                    'count': len(result.extracted_preferences),
                    'types': [p.preference_type for p in result.extracted_preferences],
                    'reasoning': result.reasoning
                }
            })
            return {"extracted_preferences": result.extracted_preferences}
        else:
            logger.debug("No preferences found", extra={'function': 'extract_preferences'})
            return {"extracted_preferences": []}

    except Exception as e:
        logger.error("Error during preference extraction", exc_info=True)
        return {"extracted_preferences": []}