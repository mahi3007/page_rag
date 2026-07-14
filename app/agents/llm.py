import os
import re
import json
from typing import Type, TypeVar
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Ollama Import
from langchain_ollama import ChatOllama

load_dotenv()

# Ollama Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# Compatibility shim: evaluations check api_key to decide whether to run LLM judge
api_key = "ollama_active"

T = TypeVar("T", bound=BaseModel)


def preprocess_json_data(data):
    """Recursively parses any JSON strings embedded within dictionary values."""
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, str):
                v_stripped = v.strip()
                if (v_stripped.startswith("[") and v_stripped.endswith("]")) or (
                    v_stripped.startswith("{") and v_stripped.endswith("}")
                ):
                    try:
                        parsed = json.loads(v_stripped)
                        new_data[k] = preprocess_json_data(parsed)
                    except Exception:
                        new_data[k] = v
                else:
                    new_data[k] = v
            elif isinstance(v, list):
                new_data[k] = [preprocess_json_data(item) for item in v]
            else:
                new_data[k] = preprocess_json_data(v)
        return new_data
    elif isinstance(data, list):
        return [preprocess_json_data(item) for item in data]
    else:
        return data


def get_llm():
    """Returns a ChatOllama LangChain client."""
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=0.0,
        num_ctx=4096,
    )


def generate_response(prompt: str, system_instruction: str = None) -> str:
    """Generates a text response from the Ollama model via LangChain."""
    try:
        llm = get_llm()
        messages = []
        if system_instruction:
            messages.append(("system", system_instruction))
        messages.append(("user", prompt))

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return (
            f"Error communicating with Ollama "
            f"({OLLAMA_URL}, model={OLLAMA_MODEL}): {str(e)}"
        )


def generate_structured_response(
    prompt: str,
    response_schema: Type[T],
    system_instruction: str = None,
) -> T:
    """
    Generates a structured JSON response from Ollama and validates it against
    a Pydantic schema. Uses Ollama's native JSON mode for reliable output.
    """
    try:
        # Use JSON output mode for reliable structured responses
        llm = ChatOllama(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_ctx=4096,
            format="json",
        )

        is_v1 = not hasattr(response_schema, "model_json_schema")
        schema = response_schema.schema() if is_v1 else response_schema.model_json_schema()
        sys_instruct = system_instruction or "You are a helpful research assistant."
        sys_instruct += (
            f"\n\nYou MUST reply ONLY with a valid JSON object matching the JSON schema below. "
            f"Do not include any other conversational filler or markdown code blocks "
            f"(e.g. do NOT wrap with ```json). "
            f"Ensure all required fields are populated.\n\n"
            f"JSON Schema:\n{json.dumps(schema)}"
        )

        messages = [
            ("system", sys_instruct),
            ("user", prompt),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        # Clean markdown wrappers if returned despite JSON mode
        if content.startswith("```"):
            # Strip ```json ... ``` or ``` ... ``` fences
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        # If still not valid JSON, try extracting the first {...} block
        if not (content.startswith("{") or content.startswith("[")):
            brace_start = content.find("{")
            if brace_start != -1:
                content = content[brace_start:]

        # Parse and validate with Pydantic schema
        data = json.loads(content.strip())
        preprocessed_data = preprocess_json_data(data)
        if is_v1:
            return response_schema.parse_obj(preprocessed_data)
        else:
            return response_schema.model_validate(preprocessed_data)

    except Exception as e:
        print(f"Ollama structured response generation failed: {e}")
        raise e
