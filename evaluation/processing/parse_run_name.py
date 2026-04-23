"""Parse structured experiment names into tidy columns.

Training scripts encode hyper-parameters into run directory names like::

    model_ablation__resnet50__in1k__img256__k4-4_a0.1-0.5_soft-ce_as-cosine_scheduling__seed=42
    vit-wee__in1k__img256__k6-6_a0.1-0.5_mixed_ma0.1_as-cosine_scheduling__seed=43

Sections are separated by ``"__"`` and within a section individual tokens
are separated by ``"_"``.  A token of the form ``"<key>=<value>"`` or
``"<key><value>"`` (e.g. ``"k4-4"``, ``"a0.1-0.5"``, ``"ma0.1"``) is parsed
into a column.  Anything else becomes a positional tag.

The parser is deliberately permissive: unknown tokens are stored verbatim
under ``tags`` rather than dropped.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable

import pandas as pd


# Explicit ``key=value`` tokens, e.g. ``seed=42``.
_KV_EQUALS = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<value>.+)$")

# Keys without ``=``, must be listed here to be parsed.  Order matters (longer
# prefixes first) so that e.g. ``ma`` is matched before ``a``.
_PREFIX_KEYS = (
    "img",    # img256
    "ma",     # ma0.1
    "a",      # a0.1-0.5
    "k",      # k4-4
    "ld",     # ld0.6
)


def _parse_prefix_token(token: str) -> Dict[str, str]:
    for key in _PREFIX_KEYS:
        if token.startswith(key) and len(token) > len(key):
            rest = token[len(key):]
            # ranges like "0.1-0.5" or "4-4" are preserved as strings.
            if rest[0].isdigit() or rest[0] == "-":
                return {key: rest}
    return {}


def parse_run_name(name: str) -> Dict[str, Any]:
    """Decompose a run name into a tidy record.

    Always returns ``{"run_name": name, "tags": [...]}`` plus any parsed keys.
    Numeric values are *not* cast here to keep the parser loss-less; cast
    downstream with :func:`pandas.to_numeric`.
    """
    record: Dict[str, Any] = {"run_name": name}
    tags: list[str] = []

    sections = name.split("__")
    # Section 0 is usually an experiment / model family tag.
    if sections:
        record["experiment_tag"] = sections[0]

    for section in sections[1:]:
        for token in section.split("_"):
            if not token:
                continue
            m = _KV_EQUALS.match(token)
            if m:
                record[m.group("key")] = m.group("value")
                continue
            parsed = _parse_prefix_token(token)
            if parsed:
                record.update(parsed)
            else:
                tags.append(token)

    record["tags"] = tags
    return record


def add_parsed_columns(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """Return a copy of ``df`` with columns parsed from ``df[name_col]``.

    New columns never overwrite existing ones; conflicts are suffixed with
    ``"_parsed"``.
    """
    parsed = pd.DataFrame([parse_run_name(n) for n in df[name_col].astype(str)])
    out = df.copy()
    for col in parsed.columns:
        target = col if col not in out.columns else f"{col}_parsed"
        out[target] = parsed[col].values
    return out


def parse_many(names: Iterable[str]) -> pd.DataFrame:
    """Convenience helper: parse a sequence of names into a DataFrame."""
    return pd.DataFrame([parse_run_name(n) for n in names])
