import os

import httpx

JINA_READER_BASE = "https://r.jina.ai/"
DEFAULT_TIMEOUT = 60.0


async def run(url: str, max_chars: int = 50000) -> str:
    """Fetch a URL and return its main content via Jina Reader (jina.ai). No API key required; set JINA_API_KEY for higher rate limits."""
    url = (url or "").strip()
    if not url:
        return "Error: URL is empty."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    reader_url = JINA_READER_BASE + url
    headers = {
        "User-Agent": "MCP-Agent-Tools/1.0",
        "X-Respond-With": "markdown",
    }
    api_key = os.getenv("JINA_API_KEY")
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    max_chars = max(1, min(max_chars, 1_000_000))

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
            text = response.text
    except httpx.TimeoutException as e:
        return f"Visit failed (timeout): {e!s}"
    except httpx.HTTPStatusError as e:
        return f"Visit failed (HTTP {e.response.status_code}): {e.response.text[:500] if e.response.text else e!s}"
    except Exception as e:
        return f"Visit failed: {e!s}"

    if not text or not text.strip():
        return "No content returned for this URL."

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"
    return text
