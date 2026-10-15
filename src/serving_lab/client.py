"""Client-side timing for one streaming chat request.

All timestamps are time.perf_counter() values in seconds.
Glossary: TTFT = time to first token, ITL = inter-token latency,
TPOT = time per output token (after the first).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise


@dataclass
class RequestResult:
    """What happened to one request: raw timestamps, outcome, error."""

    t_send: float
    t_first: float | None = None
    t_tokens: list[float] = field(default_factory=list)
    n_chunks: int = 0
    status: int | None = None
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token (s): first content chunk minus send time. None if no token arrived."""
        if self.t_first is None:
            return None
        return self.t_first - self.t_send

    @property
    def itl(self) -> list[float]:
        """Inter-token latency (s): gap between each pair of consecutive tokens. Empty if fewer than 2 tokens."""
        return [b - a for a, b in pairwise(self.t_tokens)]

    @property
    def e2e(self) -> float | None:
        """End-to-end latency (s): last token minus send time. None if no tokens."""
        if not self.t_tokens:
            return None
        return self.t_tokens[-1] - self.t_send

    @property
    def tpot(self) -> float | None:
        """Time per output token (s): One number per request, the average decode speed after the first token"""
        if len(self.t_tokens) < 2:
            return None
        return (self.t_tokens[-1] - self.t_first) / (len(self.t_tokens) - 1)


def parse_sse_line(line: str) -> str | None:
    """Return the token text in one SSE (Server-Sent Events) line, or None if it carries none.

    None for blank lines, comments, role-only and final empty-delta chunks.
    Raises json.JSONDecodeError on a malformed data line.
    """
    if not line.startswith("data: "):
        return None
    payload = line.removeprefix("data: ")
    if payload == "[DONE]":
        return None
    chunk = json.loads(payload)
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0].get("delta", {}).get("content") or None