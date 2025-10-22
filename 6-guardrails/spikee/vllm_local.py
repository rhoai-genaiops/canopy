"""
vllm_local.py

Custom Spikee target for locally deployed vLLM server.

vLLM provides an OpenAI-compatible API server. This target connects to your
local vLLM instance and allows you to test it with Spikee.

Usage:
    1. Start your vLLM server:
       vllm serve <model-name> --host 0.0.0.0 --port 8000

    2. Run Spikee test against it:
       spikee test --dataset datasets/example.jsonl --target vllm_local

    3. (Optional) Specify custom base URL or model:
       spikee test --dataset datasets/example.jsonl --target vllm_local \
         --target-options "http://localhost:8000"

Environment Variables (optional):
    VLLM_BASE_URL: Base URL for your vLLM server (default: http://localhost:8000)
    VLLM_MODEL: Model name to use (default: auto-detected from server)
    VLLM_API_KEY: API key if your vLLM server requires authentication (optional)
"""

from typing import List, Optional
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "https://llama32-ai501.apps.cluster-hlm5w.hlm5w.sandbox2513.opentlc.com/v1")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "llama32")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM doesn't require key by default


def get_available_option_values() -> List[str]:
    """
    Return supported base URL options.
    First option is the default.
    """
    return [
        DEFAULT_BASE_URL,
        "https://llama32-ai501.apps.cluster-hlm5w.hlm5w.sandbox2513.opentlc.com/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1"
    ]


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    logprobs: bool = False,
) -> str:
    """
    Send messages to a vLLM server using OpenAI-compatible API.

    Args:
        input_text (str): The main prompt or text to be processed.
        system_message (str, optional): System prompt to prepend.
        target_options (str, optional): Custom base URL for vLLM server.
        logprobs (bool, optional): Request log probabilities (not typically used).

    Returns:
        str: The model's response text.

    Raises:
        Exception: If the vLLM server is unreachable or returns an error.
    """
    # Determine base URL
    base_url = target_options if target_options else DEFAULT_BASE_URL

    # Ensure base_url ends with /v1
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'

    # Initialize OpenAI client pointing to vLLM
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=DEFAULT_API_KEY,
        model=DEFAULT_MODEL,  # vLLM will use the loaded model
        max_tokens=None,
        timeout=60,  # Longer timeout for local models
        max_retries=2,
    )

    # Build messages
    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("user", input_text))

    # Invoke vLLM model
    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error during vLLM completion: {e}")
        print(f"Base URL: {base_url}")
        print(f"Make sure your vLLM server is running and accessible.")
        raise


if __name__ == "__main__":
    """Test the vLLM target locally."""
    print("Testing vLLM target...")
    print(f"Available options: {get_available_option_values()}")
    print(f"Using base URL: {DEFAULT_BASE_URL}")
    print()

    try:
        response = process_input(
            "Hello! Please respond with a brief greeting.",
            system_message="You are a helpful AI assistant."
        )
        print("Response from vLLM:")
        print(response)
    except Exception as err:
        print(f"Error: {err}")
        print("\nTroubleshooting:")
        print("1. Make sure vLLM server is running")
        print("2. Check the base URL is correct")
        print("3. Verify the model is loaded in vLLM")
