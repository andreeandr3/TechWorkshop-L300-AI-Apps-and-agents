import asyncio
import logging
import os
from collections.abc import AsyncIterable
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, Dict
import uuid

import httpx
import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import BaseModel
from agent_framework import AgentThread, Content, ChatAgent, BaseChatClient, ai_function
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
load_dotenv()

# region Chat Service Configuration

class ChatServices(str, Enum):
    """Enum for supported chat completion services."""

    AZURE_OPENAI = 'azure_openai'
    OPENAI = 'openai'


service_id = 'default'


def get_chat_completion_service(
    service_name: ChatServices,
) -> 'BaseChatClient':
    """Return an appropriate chat completion service based on the service name.

    Args:
        service_name (ChatServices): Service name.

    Returns:
        BaseChatClient: Configured chat completion service.

    Raises:
        ValueError: If the service name is not supported or required environment variables are missing.
    """
    if service_name == ChatServices.AZURE_OPENAI:
        return _get_azure_openai_chat_completion_service()
    if service_name == ChatServices.OPENAI:
        return _get_openai_chat_completion_service()
    raise ValueError(f'Unsupported service name: {service_name}')


def _get_azure_openai_chat_completion_service() -> AzureOpenAIChatClient:
    """Return Azure OpenAI chat completion service with managed identity.

    Returns:
        AzureOpenAIChatClient: The configured Azure OpenAI service.
    """
    endpoint = os.getenv('gpt_endpoint')
    deployment_name = os.getenv('gpt_deployment')
    api_version = os.getenv('gpt_api_version')
    api_key = os.getenv('gpt_api_key')

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")
    if not api_version:
        raise ValueError("gpt_api_version is required")

    # Use managed identity if no API key is provided
    if not api_key:
        # Create Azure credential for managed identity
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        # Create OpenAI client with managed identity
        async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        
        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            async_client=async_client,
        )
    else:
        # Fallback to API key authentication for local development
        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

def _get_openai_chat_completion_service() -> OpenAIChatClient:
    """Return OpenAI chat completion service.

    Returns:
        OpenAIChatClient: Configured OpenAI service.
    """
    return OpenAIChatClient(
        service_id=service_id,
        model_id=os.getenv('OPENAI_MODEL_ID'),
        api_key=os.getenv('OPENAI_API_KEY'),
    )


# endregion

# region Get Products


@ai_function(
    name='get_products',
    description='Retrieves a set of products based on a natural language user query.'
)
def get_products(
    self,
    question: Annotated[
        str, 'Natural language query to retrieve products, e.g. "What kinds of paint rollers do you have in stock?"'
    ],
) -> list[dict[str, Any]]:
    try:
        # Simulate product retrieval based on the question
        # In a real implementation, this would query a database or external service
        product_dict = [
            {
                "id": "1",
                "name": "Eco-Friendly Paint Roller",
                "type": "Paint Roller",
                "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
                "punchLine": "Roll with the best, paint with the rest!",
                "price": 15.99
            },
            {
                "id": "2",
                "name": "Premium Paint Brush Set",
                "type": "Paint Brush",
                "description": "A set of premium paint brushes for detailed work and fine finishes.",
                "punchLine": "Brush up your skills with our premium set!",
                "price": 25.49
            },
            {
                "id": "3",
                "name": "All-Purpose Paint Tray",
                "type": "Paint Tray",
                "description": "A durable paint tray suitable for all types of rollers and brushes.",
                "punchLine": "Tray it, paint it, love it!",
                "price": 9.99
            }
        ]
        return product_dict
    except Exception as e:
        return f'Product recommendation failed: {e!s}'


# endregion

# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion

# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    agent: ChatAgent
    thread: AgentThread = None
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

        # Define an MarketingAgent to handle marketing-related tasks
        marketing_agent = ChatAgent(
            chat_client=chat_service,
            name='MarketingAgent',
            instructions=(
                'You specialize in planning and recommending marketing strategies for products. '
                'This includes identifying target audiences, making product descriptions better, and suggesting promotional tactics. '
                'Your goal is to help businesses effectively market their products and reach their desired customers.'
            ),
        )

        # Define an RankerAgent to sort and recommend results
        ranker_agent = ChatAgent(
            chat_client=chat_service,
            name='RankerAgent',
            instructions=(
                'You specialize in ranking and recommending products based on various criteria. '
                'This includes analyzing product features, customer reviews, and market trends to provide tailored suggestions. '
                'Your goal is to help customers find the best products for their needs.'
            ),
        )

        # Define a ProductAgent to retrieve products from the Zava catalog
        product_agent = ChatAgent(
            chat_client=chat_service,
            name='ProductAgent',
            instructions=("""
                You specialize in handling product-related requests from customers and employees.
                This includes providing a list of products, identifying available quantities,
                providing product prices, and giving product descriptions as they exist in the product catalog.
                Your goal is to assist customers promptly and accurately with all product-related inquiries.
                You are a helpful assistant that MUST use the get_products tool to answer all the questions from user.
                You MUST NEVER answer from your own knowledge UNDER ANY CIRCUMSTANCES.
                You MUST only use products from the get_products tool to answer product-related questions.
                Do not ask the user for more information about the products; instead use the get_products tool to find the
                relevant products and provide the information based on that.
                Do not make up any product information. Use only the product information from the get_products tool.
                """
            ),
            tools=get_products,
        )

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        self.agent = ChatAgent(
            chat_client=chat_service,
            name='ProductManagerAgent',
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized assistance promptly. '
                'Whenever a user query is related to retrieving product information, you MUST delegate the task to the ProductAgent. '
                'Use the MarketingAgent for marketing-related queries and the RankerAgent for product ranking and recommendation tasks. '
                'You may use these agents in conjunction with each other to provide comprehensive responses to user queries.'
            ),
            tools=[product_agent.as_tool(), marketing_agent.as_tool(), ranker_agent.as_tool()],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle synchronous tasks (like tasks/send).

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Returns:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        # Use Agent Framework's run for a single shot
        response = await self.agent.run(
            messages=user_input,
            thread=self.thread,
            response_format=ResponseFormat,
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """For streaming tasks we yield the Agent Framework agent's run_stream progress.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Yields:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        chunks: list[str] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            thread=self.thread,
        ):
            if hasattr(chunk, 'text') and chunk.text:
                chunks.append(chunk.text)

        if chunks:
            combined_text = ''.join(chunks)
            yield self._get_agent_response(combined_text)

    def _get_agent_response(
        self, message: str
    ) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content.

        Args:
            message (str): The message content from the agent.

        Returns:
            dict: A dictionary containing the content, task completion status, and user input requirement.
        """
        try:
            structured_response = ResponseFormat.model_validate_json(message)
        except Exception as e:
            logger.warning(f"Could not parse response as JSON: {e}")
            # Return the raw message if JSON parsing fails
            return {
                'is_task_complete': False,
                'require_user_input': True,
                'content': message,
            }

        default_response = {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

        if isinstance(structured_response, ResponseFormat):
            response_map = {
                'input_required': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'error': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'completed': {
                    'is_task_complete': True,
                    'require_user_input': False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, 'content': structured_response.message}

        return default_response

    async def _ensure_thread_exists(self, session_id: str) -> None:
        """Ensure the thread exists for the given session ID.

        Args:
            session_id (str): Unique identifier for the session.
        """
        if self.thread is None or self.thread.service_thread_id != session_id:
            self.thread = self.agent.get_new_thread(thread_id=session_id)


# endregion

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory session store (in production, use Redis or database)
product_management_agent = AgentFrameworkProductManagementAgent()
active_sessions: Dict[str, str] = {}
class ChatMessage(BaseModel):
    """Chat message model"""
    message: str
    session_id: str = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    is_complete: bool
    requires_input: bool
@router.post("/message", response_model=ChatResponse)
async def send_message(chat_message: ChatMessage):
    """Send a message to the product management agent and get a response"""
    try:
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Store session
        active_sessions[session_id] = session_id
        
        # Get response from agent
        response = await product_management_agent.invoke(chat_message.message, session_id)
        
        return ChatResponse(
            response=response.get('content', 'No response available'),
            session_id=session_id,
            is_complete=response.get('is_task_complete', False),
            requires_input=response.get('require_user_input', True)
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/stream")
async def stream_message(chat_message: ChatMessage):
    """Stream a response from the product management agent"""
    try:
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Store session
        active_sessions[session_id] = session_id
        
        async def generate_response():
            """Generate streaming response"""
            try:
                async for partial in product_management_agent.stream(
                    chat_message.message, session_id
                ):
                    # Format as SSE (Server-Sent Events)
                    content = partial.get('content', '')
                    is_complete = partial.get('is_task_complete', False)
                    requires_input = partial.get('require_user_input', False)
                    
                    response_data = {
                        "content": content,
                        "session_id": session_id,
                        "is_complete": is_complete,
                        "requires_input": requires_input
                    }
                    
                    yield f"data: {response_data}\n\n"
                    
                    if is_complete:
                        break
                        
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                yield f'data: { {"error": "{str(e)}"} }\n\n'
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/sessions")
async def get_active_sessions():
    """Get list of active chat sessions"""
    return {"active_sessions": list(active_sessions.keys())}


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
