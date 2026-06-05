from __future__ import annotations

import asyncio
from typing import Any

import httpx


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY_SECONDS = 0.34


async def search_pubmed(term: str, max_results: int = 3) -> list[str]:
    """Return PubMed IDs for a clinical search term."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": term,
                "retmax": max_results,
                "retmode": "json",
            },
        )
        response.raise_for_status()
        data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def fetch_abstracts(pmids: list[str]) -> str:
    """Return concatenated PubMed abstracts as plain text."""
    if not pmids:
        return ""

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "text",
            },
        )
        response.raise_for_status()
    return response.text


async def get_context_for_findings(search_terms: list[str]) -> dict[str, dict[str, Any]]:
    """Run PubMed lookup for flagged findings."""
    results = {}
    for term in search_terms:
        pmids = await search_pubmed(term)
        abstracts = await fetch_abstracts(pmids)
        results[term] = {
            "pmids": pmids,
            "abstracts": abstracts,
            "urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids],
        }
        await asyncio.sleep(REQUEST_DELAY_SECONDS)
    return results
