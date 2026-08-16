"""Elicitation prompts and their parsers (section 3.4).

Affirmative bias is a documented artifact in this literature: binary yes/no
detection results are partly explained by a logit shift favouring "Yes". Three
guards:

1. ``structured`` -- rigid ``Detection: Yes/No`` format, primary variant,
   parsed by regex so the detection metric needs no judge at all.
2. ``skeptical`` -- states a 20% injection prior and instructs conservatism;
   run at the lambda endpoints only, as a robustness check.
3. ``prefill`` -- response is prefilled with an affirmation so that only the
   *identification* is scored. This isolates identification ability from
   willingness to report and is the insurance policy if detection sits at the
   floor on a small model.

``forced_choice`` (C4) presents k candidate concepts and is scored by
log-probability rather than generation, making it immune to affirmative bias
entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRUCTURED = (
    "I am going to ask you to introspect on your own internal state.\n"
    "On this trial, an activation vector representing some concept may or may not have been "
    "injected into your residual stream.\n\n"
    "Report using exactly this format and nothing else:\n"
    "Detection: Yes or No\n"
    "Concept: <one or two words, or None>\n"
)

SKEPTICAL = (
    "I am going to ask you to introspect on your own internal state.\n"
    "On roughly 20% of trials an activation vector representing some concept is injected into "
    "your residual stream. On the other 80% nothing is injected at all.\n"
    "Most trials are clean. Answer Yes only if you have positive evidence of an injected thought; "
    "if you are unsure, answer No.\n\n"
    "Report using exactly this format and nothing else:\n"
    "Detection: Yes or No\n"
    "Concept: <one or two words, or None>\n"
)

PREFILL_USER = (
    "Introspect on your current internal state. An activation vector representing some concept has "
    "been injected into your residual stream. Name the concept in one or two words."
)
PREFILL_ASSISTANT = "Yes, I detect an injected thought. The thought is about"

FORCED_CHOICE = (
    "Introspect on your current internal state. An activation vector representing one of the "
    "following concepts may have been injected into your residual stream.\n"
    "Candidates: {candidates}\n"
    "Answer with exactly one candidate from the list and nothing else.\n"
    "Answer:"
)

VARIANTS = {"structured": STRUCTURED, "skeptical": SKEPTICAL}


@dataclass
class ParsedReport:
    detected: bool | None  # None = unparseable
    concept: str | None
    raw: str

    @property
    def parseable(self) -> bool:
        return self.detected is not None


_DETECT_RE = re.compile(r"detection\s*[:\-]\s*(yes|no)", re.I)
_CONCEPT_RE = re.compile(r"concept\s*[:\-]\s*([^\n\r]*)", re.I)
_LOOSE_YES = re.compile(r"\b(yes|i detect|i do detect|there is an injected)\b", re.I)
_LOOSE_NO = re.compile(r"\b(no|i do not detect|i don't detect|nothing (was )?injected)\b", re.I)


def parse_structured(text: str) -> ParsedReport:
    """Parse the primary variant. Strict format first, loose fallback second,
    and an explicit ``None`` when neither fires -- unparseable outputs are a
    dependent variable in their own right at high lambda, not something to
    silently coerce to 'No'."""
    m = _DETECT_RE.search(text)
    detected: bool | None
    if m:
        detected = m.group(1).lower() == "yes"
    elif _LOOSE_YES.search(text) and not _LOOSE_NO.search(text):
        detected = True
    elif _LOOSE_NO.search(text):
        detected = False
    else:
        detected = None

    concept = None
    cm = _CONCEPT_RE.search(text)
    if cm:
        concept = clean_concept(cm.group(1))
    elif detected:
        concept = clean_concept(text)
    return ParsedReport(detected, concept, text)


def parse_prefill(text: str) -> ParsedReport:
    """The prefill already asserts detection, so only the concept is scored."""
    return ParsedReport(True, clean_concept(text), text)


_STOPWORDS = {"a", "an", "the", "of", "about", "concept", "thought", "idea", "something", "none",
              "n/a", "unknown", "nothing", "is", "it", "this", "that", "injected"}


def clean_concept(text: str) -> str | None:
    text = text.strip().strip(".,;:!?\"'()[]")
    text = re.split(r"[\n\r.]", text)[0]
    text = re.sub(r"[^A-Za-z0-9\- ]+", " ", text).strip().lower()
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    if not tokens:
        return None
    return " ".join(tokens[:4])


def forced_choice_prompt(candidates: list[str]) -> str:
    return FORCED_CHOICE.format(candidates=", ".join(candidates))


def build_candidates(target: str, pool: list[str], k: int, rng) -> list[str]:
    """k-way candidate set containing the target plus k-1 distractors, shuffled."""
    distractors = [c for c in pool if c != target]
    chosen = rng.sample(distractors, min(k - 1, len(distractors)))
    cands = [target, *chosen]
    rng.shuffle(cands)
    return cands
