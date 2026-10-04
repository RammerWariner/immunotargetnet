"""Open Targets Platform client used by the keystone-layer refresh.

Ref: Sec. 2.2 (knowledge resources) and Table 1 note (tractability annotations).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Final

from immunotargetnet.evidence.schema import GeneAnnotation, TractabilityBucket, TypedEdge

GRAPHQL_ENDPOINT: Final[str] = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_RETRIES: Final[int] = 2
DEFAULT_PAUSE_SECONDS: Final[float] = 1.0


class OpenTargetsError(RuntimeError):
    """Raised when the platform returns an error or stays unreachable."""


@dataclass(frozen=True)
class InteractionRecord:
    int_a: str
    int_b: str
    symbol_a: str
    symbol_b: str
    score: float
    source_database: str
    role: str


_TARGET_BATCH_FIELDS: Final[str] = """
  approvedSymbol
  biotype
  isEssential
  tractability { modality label value }
"""


class OpenTargetsClient:
    def __init__(
        self,
        *,
        endpoint: str = GRAPHQL_ENDPOINT,
        retries: int = DEFAULT_RETRIES,
        pause_seconds: float = DEFAULT_PAUSE_SECONDS,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._endpoint = endpoint
        self._retries = retries
        self._pause = pause_seconds
        self._timeout = timeout_seconds

    def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self._retries):
            request = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                time.sleep(self._pause * (attempt + 1))
                continue
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                raise OpenTargetsError(f"platform returned {error.code}: {detail}") from error
            if "errors" in body:
                raise OpenTargetsError(f"platform rejected the query: {body['errors']}")
            data = body.get("data")
            if not isinstance(data, dict):
                raise OpenTargetsError("platform returned no data block")
            return data
        raise OpenTargetsError(f"platform unreachable after {self._retries} attempts: {last_error}")

    def map_symbols(self, symbols: list[str], *, batch_size: int = 80) -> dict[str, str]:
        """Resolve approved symbols to Ensembl gene identifiers."""
        resolved: dict[str, str] = {}
        query = """
        query($terms:[String!]!) {
          mapIds(queryTerms:$terms, entityNames:["target"]) {
            mappings { term hits { id name entity } }
          }
        }
        """
        for start in range(0, len(symbols), batch_size):
            chunk = symbols[start : start + batch_size]
            data = self._post(query, {"terms": chunk})
            mappings = data["mapIds"]["mappings"]
            for entry in mappings:
                hits = [hit for hit in entry["hits"] if hit["entity"] == "target"]
                if len(hits) == 1:
                    resolved[entry["term"]] = str(hits[0]["id"])
            time.sleep(self._pause)
        return resolved

    def annotations(self, ensembl_ids: list[str], *, batch_size: int = 60) -> dict[str, GeneAnnotation]:
        query = "query($ids:[String!]!) { targets(ensemblIds:$ids) { id " + _TARGET_BATCH_FIELDS + " } }"
        annotations: dict[str, GeneAnnotation] = {}
        for start in range(0, len(ensembl_ids), batch_size):
            chunk = ensembl_ids[start : start + batch_size]
            data = self._post(query, {"ids": chunk})
            for record in data["targets"]:
                buckets = tuple(
                    TractabilityBucket(
                        modality=str(bucket["modality"]),
                        label=str(bucket["label"]),
                        value=bool(bucket["value"]),
                    )
                    for bucket in record["tractability"]
                )
                annotations[str(record["id"])] = GeneAnnotation(
                    symbol=str(record["approvedSymbol"]),
                    ensembl_id=str(record["id"]),
                    biotype=str(record["biotype"]),
                    tractability=buckets,
                    is_essential=bool(record["isEssential"]),
                )
            time.sleep(self._pause)
        return annotations

    def interactions(self, ensembl_id: str) -> list[InteractionRecord]:
        query = """
        query($id:String!) {
          target(ensemblId:$id) {
            interactions {
              rows { intA intB targetA targetB score sourceDatabase intABiologicalRole }
            }
          }
        }
        """
        data = self._post(query, {"id": ensembl_id})
        target = data["target"]
        if target is None:
            return []
        rows = target["interactions"]["rows"]
        return [
            InteractionRecord(
                int_a=str(row["intA"]),
                int_b=str(row["intB"]),
                symbol_a=str(row["targetA"]),
                symbol_b=str(row["targetB"]),
                score=float(row["score"]),
                source_database=str(row["sourceDatabase"]),
                role=str(row["intABiologicalRole"]),
            )
            for row in rows
        ]

    def typed_edges(
        self,
        ensembl_ids: list[str],
        *,
        partner_universe: set[str],
        min_score: float = 0.4,
        budget_seconds: float = 900.0,
        max_queries: int = 400,
    ) -> tuple[list[TypedEdge], list[str], list[str]]:
        """Collect interaction edges whose two ends both sit in the candidate pool.

        The sweep is bounded by a query count and a wall-clock budget so a throttled host
        stops the collection instead of hanging; the genes left unqueried are returned.
        """
        edges: list[TypedEdge] = []
        seen: set[tuple[str, str, str]] = set()
        failed: list[str] = []
        skipped: list[str] = []
        started = time.monotonic()
        for position, ensembl_id in enumerate(ensembl_ids):
            if position >= max_queries or time.monotonic() - started > budget_seconds:
                skipped = list(ensembl_ids[position:])
                break
            try:
                records = self.interactions(ensembl_id)
            except OpenTargetsError:
                failed.append(ensembl_id)
                continue
            for record in records:
                if record.int_a not in partner_universe or record.int_b not in partner_universe:
                    continue
                if record.score < min_score:
                    continue
                key = (record.int_a, record.int_b, record.source_database)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    TypedEdge(
                        source=record.int_a,
                        target=record.int_b,
                        edge_type=record.source_database,
                        weight=record.score,
                        source_database=record.source_database,
                    )
                )
            time.sleep(self._pause)
        return edges, failed, skipped
