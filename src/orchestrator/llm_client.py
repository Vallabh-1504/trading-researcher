import json
import os
import re
from typing import Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Provider constants
class LLMProvider:
    GEMINI = "gemini"
    OLLAMA = "ollama"

# Default models per provider
DEFAULT_MODELS = {
    LLMProvider.GEMINI: "gemini-3.1-flash-lite",
    LLMProvider.OLLAMA: "qwen2.5:7b",
}

# Helper Function for Ollama
def _clean_llm_json_output(raw_text: str) -> str:
    """
    Strips reasoning blocks and markdown fences from LLM output.
    Essential for local models (via Ollama) that format their JSON responses.
    """
    # 1. Strip reasoning tags (e.g., from DeepSeek R1 or Qwen reasoning models)
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

    text = text.strip()
    
    # 2. strip markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
            
    return text.strip()

# Main Client Class
class LLMClient:
    """
    Unified LLM interface.
    Usage:
        client = LLMClient(provider="gemini")
        result = client.generate_structured(system_prompt, user_prompt, schema)
    """

    def __init__(
        self,
        provider: str = LLMProvider.GEMINI,
        model: str | None = None,
    ):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(self.provider)

        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider '{self.provider}'. Valid options: {list(DEFAULT_MODELS.keys())}")

        print(f"[LLMClient] Initializing provider='{self.provider}' model='{self.model}'")
        self._init_client()


    def _init_client(self):
        """Initializes the underlying SDK client based on the chosen provider."""
        if self.provider == LLMProvider.GEMINI:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not found in environment")
            
            self._client = genai.Client(api_key=api_key)

        elif self.provider == LLMProvider.OLLAMA:
            # using openAI Sdk to interact with ollama
            from openai import OpenAI
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            self._client = OpenAI(base_url=base_url, api_key="ollama")

        else:
            raise ValueError(f"Unknown provider '{self.provider}'. ")


    # Public Interface
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> T:
        """
        Generate a structured response that conforms to the given Pydantic schema.
        Will raise a RuntimeError if the model fails to generate or format correctly.
        """
        try:
            if self.provider == LLMProvider.GEMINI:
                return self._gemini_generate(system_prompt, user_prompt, schema)
            
            else:
                return self._ollama_generate(system_prompt, user_prompt, schema)
                
        except Exception as e:
            # Catching at the highest level to prevent silent failures in the graph
            raise RuntimeError(
                f"[LLMClient Error] Failed to generate or parse structured output for {self.provider}. "
                f"Details: {str(e)}"
            ) from e


    # Provider-Specific Implementations
    def _gemini_generate(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0, # Deterministic for research tasks
            ),
        )

        # response.text is guaranteed JSON by the SDK's enforcement
        return schema.model_validate_json(response.text)


    def _ollama_generate(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        """
        Ollama path: uses OpenAI-compatible JSON mode.
    
        Since these don't have native schema enforcement, we inject the full
        JSON schema into the system prompt and rely on JSON mode to constrain
        the output format.
        """
        # Serialize the Pydantic schema to JSON Schema format and inject it
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        enhanced_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with ONLY a valid JSON object. "
            f"Do NOT return the schema definition itself. Return a JSON object that structure.\n"
            f"The JSON must strictly conform to this JSON Schema:\n{schema_json}"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": enhanced_system},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},  # Forcing JSON mode
        )

        raw_text = response.choices[0].message.content or ""
        cleaned_text = _clean_llm_json_output(raw_text)

        return schema.model_validate_json(cleaned_text)


# Module-Level Singleton
"""
The graph nodes need access to the LLM client. Since LangGraph nodes are
plain functions (not class methods), we use a module-level singleton that
is initialized lazily from environment variables on first use.
This avoids passing the client through the state (not JSON-serializable).
"""
_client_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    Returns the shared LLMClient instance. Initializes it from env vars on
    first call. Subsequent calls return the same instance.
    """
    global _client_singleton

    if _client_singleton is None:
        provider = os.getenv("LLM_PROVIDER", LLMProvider.GEMINI)
        model    = os.getenv("LLM_MODEL") or None   # None → use provider default
        _client_singleton = LLMClient(provider=provider, model=model)

    return _client_singleton


# Isolated Module Testing
if __name__ == "__main__":
    from src.orchestrator.schemas import DocumentRelevance
    from src.orchestrator.prompts import get_grader_prompt
    
    print("\nStarting LLMClient Test\n")
    try:
        client = get_llm_client()
        system = get_grader_prompt("KO", "PEP")
        user = "PepsiCo experienced a massive labor strike shutting down 15% of North American volume."
        
        result = client.generate_structured(system, user, DocumentRelevance)
        print("\n[SUCCESS] Structured Output Received:")
        print(result.model_dump_json(indent=2))
        
        assert isinstance(result.is_relevant, bool), "Type enforcement failed."
        print("\n[SUCCESS] Pydantic parsing passed.")
        
    except Exception as e:
        print(f"\n[FAILED] LLMClient threw an exception: {e}")