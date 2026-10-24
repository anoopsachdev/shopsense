"""Ranking metrics over ESCI relevance labels.

Two gain sets map ESCI labels to NDCG gains:

- ESCI_OFFICIAL_GAINS (the headline): Exact 1.0, Substitute 0.1, Complement 0.01,
  Irrelevant 0. These are the gains the esci-data repo's own Task 1 ranking scripts use
  (ranking/prepare_trec_eval_files.py, ranking/train.py), so scores are on the same scale
  as published ESCI results. Substitutes get little credit.
- ESCI_GRADED_GAINS: Exact 3, Substitute 2, Complement 1, Irrelevant 0, the CLAUDE.md
  fallback. Substitutes get two-thirds credit, which pushes scores up.

Products without a label for the query count as irrelevant (gain 0).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

ESCI_OFFICIAL_GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
ESCI_GRADED_GAINS: dict[str, float] = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}

# Gain sets by name, as accepted by scripts/evaluate.py --gains. The first is the headline.
GAIN_SETS: dict[str, dict[str, float]] = {
    "official": ESCI_OFFICIAL_GAINS,
    "graded": ESCI_GRADED_GAINS,
}


def gain(label: str | None, gains: Mapping[str, float] = ESCI_OFFICIAL_GAINS) -> float:
    """Gain for an ESCI label; None (unlabelled) is 0."""
    return 0.0 if label is None else gains[label]


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of gains in rank order: sum of g_i / log2(i + 1)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(
    ranked_ids: Sequence[str],
    labels: Mapping[str, str],
    k: int,
    gains: Mapping[str, float] = ESCI_OFFICIAL_GAINS,
) -> float:
    """NDCG@k of a ranking against one query's labels.

    The ideal ranking is the query's labelled products sorted by gain. Returns 0.0 when
    the query has no product with positive gain.
    """
    ranked_gains = [gain(labels.get(pid), gains) for pid in ranked_ids[:k]]
    ideal_gains = sorted((gain(label, gains) for label in labels.values()), reverse=True)[:k]
    ideal = dcg(ideal_gains)
    return dcg(ranked_gains) / ideal if ideal > 0 else 0.0


def recall_at_k(
    ranked_ids: Sequence[str],
    labels: Mapping[str, str],
    k: int,
    gains: Mapping[str, float] = ESCI_OFFICIAL_GAINS,
) -> float:
    """Share of the query's relevant products (gain > 0) found in the top k.

    Returns 0.0 when the query has no relevant product.
    """
    relevant = {pid for pid, label in labels.items() if gain(label, gains) > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0
