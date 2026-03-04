# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio


async def run(query: str, limit: int = 5) -> str:
    """Run academic/scholarly search via Google Scholar (scholarly package). No API key required."""
    try:
        from scholarly import scholarly
    except ImportError as e:
        return f"Error: scholarly is not installed: {e}. Run: pip install scholarly"

    limit = min(max(1, limit), 20)

    def _search_sync():
        """Run sync generator in thread; collect up to `limit` results."""
        results = []
        try:
            gen = scholarly.search_pubs(query)
            for _ in range(limit):
                try:
                    pub = next(gen)
                    results.append(pub)
                except StopIteration:
                    break
        except Exception as e:
            raise e
        return results

    try:
        results = await asyncio.to_thread(_search_sync)
    except Exception as e:
        return f"Google Scholar search failed: {e!s}"

    if not results:
        return "No papers found for that query."

    parts = []
    for i, pub in enumerate(results, 1):
        bib = pub.get("bib") or {}
        title = bib.get("title") or ""
        abstract = bib.get("abstract") or ""
        author_list = bib.get("author") or []
        author_str = ", ".join(author_list) if isinstance(author_list, list) else str(author_list)
        year = bib.get("pub_year") or ""
        num_citations = pub.get("num_citations")
        pub_url = pub.get("pub_url") or ""

        line_parts = [f"[{i}] {title}"]
        if year:
            line_parts.append(f"Year: {year}")
        if num_citations is not None:
            line_parts.append(f"Citations: {num_citations}")
        if author_str:
            line_parts.append(f"Authors: {author_str}")
        if abstract:
            line_parts.append(f"Abstract: {abstract[:500]}{'...' if len(abstract) > 500 else ''}")
        if pub_url:
            line_parts.append(f"URL: {pub_url}")
        parts.append("\n".join(line_parts))

    return "\n\n".join(parts)
