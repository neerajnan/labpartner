from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}


class PubMedError(RuntimeError):
    pass


def base_params() -> dict[str, str]:
    params = {"tool": "labpartner"}
    if email := os.getenv("NCBI_EMAIL"):
        params["email"] = email
    if api_key := os.getenv("NCBI_API_KEY"):
        params["api_key"] = api_key
    return params


async def get_with_retries(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any],
) -> httpx.Response:
    merged_params = {**base_params(), **params}
    last_error = None
    for attempt in range(MAX_RETRIES):
        response = await client.get(f"{BASE_URL}/{endpoint}", params=merged_params)
        if response.status_code not in RETRY_STATUSES:
            response.raise_for_status()
            return response

        last_error = response
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        else:
            delay = REQUEST_DELAY_SECONDS * (2 ** attempt)
        await asyncio.sleep(delay)

    assert last_error is not None
    raise PubMedError(f"PubMed request failed with HTTP {last_error.status_code}.")


async def search_pubmed(term: str, max_results: int = 3) -> list[str]:
    """Return PubMed IDs for a clinical search term."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await get_with_retries(
            client,
            "esearch.fcgi",
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
        response = await get_with_retries(
            client,
            "efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "text",
            },
        )
    return response.text


async def get_context_for_findings(search_terms: list[str]) -> dict[str, dict[str, Any]]:
    """Run PubMed lookup for flagged findings."""
    results = {}
    for term in search_terms:
        pmids = await search_pubmed(term)
        await asyncio.sleep(REQUEST_DELAY_SECONDS)
        abstracts = await fetch_abstracts(pmids)
        results[term] = {
            "pmids": pmids,
            "abstracts": abstracts,
            "urls": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids],
        }
        await asyncio.sleep(REQUEST_DELAY_SECONDS)
    return results
