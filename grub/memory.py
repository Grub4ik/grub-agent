"""Persistent semantic memory for concise user facts."""

from __future__ import annotations

import json
import http.client
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class MemoryError(RuntimeError):
    """Raised when persistent memory or its local embedding model is unavailable."""


class FactMemory:
    """Store and retrieve short facts with Chroma and local Ollama embeddings."""

    COLLECTION_NAME = "user_facts"

    def __init__(
        self,
        path: str | Path | None = None,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "embeddinggemma",
        timeout: float = 20.0,
        max_distance: float = 0.55,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Embedding timeout must be positive.")
        if not 0.0 <= max_distance <= 2.0:
            raise ValueError("Cosine distance threshold must be between 0 and 2.")

        parsed = urlsplit(ollama_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama URL must start with http:// or https://.")
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_distance = max_distance
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._prefix = parsed.path.rstrip("/")

        data_home = os.environ.get("XDG_DATA_HOME")
        default_data_home = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
        storage_path = Path(path).expanduser() if path else default_data_home / "grub-agent" / "chroma"
        storage_path.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(storage_path))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise MemoryError(f"Could not open Chroma memory: {exc}") from exc

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.embedding_model,
            "input": texts,
            "keep_alive": "10m",
            "options": {"num_gpu": 0},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection_type = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self._host, self._port, timeout=self.timeout)
        try:
            connection.request(
                "POST",
                self._prefix + "/api/embed",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response_body = response.read()
            if response.status >= 400:
                detail = response_body.decode("utf-8", errors="replace")
            else:
                detail = ""
            if response.status == 404:
                raise MemoryError(
                    f"Ollama embedding model {self.embedding_model!r} is missing. "
                    f"Download it with: ollama pull {self.embedding_model}"
                )
            if response.status >= 400:
                raise MemoryError(f"Ollama embedding request failed (HTTP {response.status}): {detail}")
            response_data = json.loads(response_body.decode("utf-8"))
        except MemoryError:
            raise
        except (OSError, http.client.HTTPException, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryError(f"Could not get local text embeddings from Ollama: {exc}") from exc
        finally:
            connection.close()

        embeddings = response_data.get("embeddings") if isinstance(response_data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise MemoryError("Ollama returned an unexpected embedding response.")
        try:
            return [[float(value) for value in vector] for vector in embeddings]
        except (TypeError, ValueError) as exc:
            raise MemoryError("Ollama returned an invalid embedding vector.") from exc

    @staticmethod
    def _normalize(fact: str) -> str:
        return re.sub(r"\s+", " ", fact.casefold()).strip()

    def save_fact(self, fact: str, category: str = "general") -> dict[str, Any]:
        fact = " ".join(fact.split())
        if not fact:
            raise ValueError("Fact cannot be empty.")
        if len(fact) > 1000:
            raise ValueError("Fact cannot exceed 1000 characters.")
        if len(category) > 40:
            raise ValueError("Category cannot exceed 40 characters.")

        normalized = self._normalize(fact)
        existing = self._collection.get(where={"normalized": normalized}, include=["metadatas"])
        fact_id = existing["ids"][0] if existing.get("ids") else str(uuid.uuid4())
        old_metadata = (existing.get("metadatas") or [None])[0] or {}
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "category": category,
            "normalized": normalized,
            "created_at": old_metadata.get("created_at", now),
            "updated_at": now,
        }
        embedding = self._embed([fact])[0]
        self._collection.upsert(
            ids=[fact_id],
            documents=[fact],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return {"id": fact_id, "fact": fact, "category": category, "status": "saved"}

    def search_facts(self, query: str, limit: int = 4) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        if not 1 <= limit <= 10:
            raise ValueError("Search limit must be between 1 and 10.")
        if self._collection.count() == 0:
            return []

        embedding = self._embed([query])[0]
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        matches: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for fact_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if distance <= self.max_distance:
                matches.append({"id": fact_id, "fact": document, **(metadata or {}), "distance": distance})
        return matches

    def list_facts(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("List limit must be between 1 and 100.")
        result = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
        return [
            {"id": fact_id, "fact": fact, **(metadata or {})}
            for fact_id, fact, metadata in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        ]

    def forget_fact(self, fact_id: str) -> dict[str, Any]:
        result = self._collection.get(ids=[fact_id], include=["documents"])
        documents = result.get("documents") or []
        if not documents:
            raise ValueError("No saved fact exists with that ID.")
        self._collection.delete(ids=[fact_id])
        return {"id": fact_id, "fact": documents[0], "status": "forgotten"}
