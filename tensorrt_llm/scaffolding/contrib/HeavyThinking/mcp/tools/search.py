import asyncio
import os


async def run(query: str) -> str:
    """Run web search via Tavily API. Requires TAVILY_API_KEY in environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or not api_key.strip():
        return (
            "Error: TAVILY_API_KEY is not set. "
            "Add it to .env or export TAVILY_API_KEY=your_key"
        )

    try:
        from tavily import TavilyClient
    except ImportError as e:
        return f"Error: tavily-python is not installed: {e}. Run: pip install tavily-python"

    client = TavilyClient(api_key=api_key)
    try:
        response = await asyncio.to_thread(client.search, query=query)
    except Exception as e:
        return f"Tavily search failed: {e!s}"

    results = response.get("results")
    if not results:
        return "No results found for that query."

    parts = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        line = f"{title}: {content}".strip()
        if url:
            line += f" (URL: {url})"
        parts.append(line)
    return "\n\n".join(parts)
