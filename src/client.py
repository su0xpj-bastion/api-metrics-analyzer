"""OpenAI client factory with org-scoped configuration."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_client() -> OpenAI:
    """Create a configured OpenAI client from environment variables.

    Returns:
        OpenAI: client instance authenticated via OPENAI_API_KEY.

    Raises:
        ValueError: if OPENAI_API_KEY is not set in the environment.

    Example:
        >>> client = get_client()
        >>> type(client)
        <class 'openai.OpenAI'>
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Copy .env.example → .env and fill in your key."
        )
    org_id = os.getenv("OPENAI_ORG_ID")  # optional; scopes billing to org

    return OpenAI(
        api_key=api_key,
        organization=org_id or None,
    )
