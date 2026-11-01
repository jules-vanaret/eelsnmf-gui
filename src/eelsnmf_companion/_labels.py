from __future__ import annotations

from typing import List, Tuple


def split_common_label_parts(labels: List[str]) -> Tuple[str, List[str]]:
    """Split a list of labels into a shared prefix and per-label suffixes."""
    if len(labels) <= 1:
        return (labels[0] if labels else ""), labels

    parts = [label.split("_") for label in labels]
    min_len = min(len(p) for p in parts)
    common_indices = []
    for i in range(min_len):
        token = parts[0][i]
        if all(p[i] == token for p in parts):
            common_indices.append(i)

    common_tokens = [parts[0][i] for i in common_indices]
    common = "_".join(common_tokens)

    individuals = []
    for tokens in parts:
        individual_tokens = [
            t for i, t in enumerate(tokens) if i not in common_indices
        ]
        individuals.append("_".join(individual_tokens))

    return common, individuals
