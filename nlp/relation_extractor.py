
import json
import os
import re
import requests
import spacy
from dotenv import load_dotenv

load_dotenv()

nlp = spacy.load("en_core_sci_sm")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# Pronouns and auxiliary stop tokens that produce invalid relations
INVALID_RELATION_TOKENS = {
    "it", "they", "this", "that", "these", "those", "which", "who", "whom", "whose",
    "there", "here", "what", "we", "i", "you", "he", "she", "one", "all", "both"
}


def is_valid_relation_endpoint(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    cleaned = name.strip().lower()
    if len(cleaned) < 2:
        return False
    if cleaned in INVALID_RELATION_TOKENS:
        return False
    if cleaned.isdigit() or all(not c.isalnum() for c in cleaned):
        return False
    return True


def _dependency_relations(sent, filter_generic: bool = False):
    """Extract explicit subject-verb-object relations from dependencies."""
    results = []

    for token in sent:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [
            child for child in token.children
            if child.dep_ in {"nsubj", "nsubjpass", "csubj", "nsubj:pass"}
        ]
        objects = [
            child for child in token.children
            if child.dep_ in {"dobj", "obj", "attr", "oprd", "dative"}
        ]

        for subj in subjects:
            if not is_valid_relation_endpoint(subj.text):
                continue
            for obj in objects:
                if not is_valid_relation_endpoint(obj.text):
                    continue
                results.append({
                    "subject": subj.text,
                    "subject_label": "UNKNOWN",
                    "relation": token.lemma_.lower(),
                    "object": obj.text,
                    "object_label": "UNKNOWN",
                    "evidence_text": sent.text,
                    "sentence_start": sent.start_char,
                    "sentence_end": sent.end_char,
                    "relation_start": token.idx,
                    "relation_end": token.idx + len(token.text),
                    "extraction_method": "dependency_heuristic",
                    "validation_status": "unvalidated",
                })

    return results


def _llm_relations(sent):
    """Use Ollama when dependency parsing finds no relation."""
    prompt = f"""
Extract explicit relationships stated in this scientific sentence.
Return JSON only, using this schema:
{{"relations":[{{"subject":"exact source span",
"relation":"exact predicate span","object":"exact source span"}}]}}

Do not infer facts. Subject, relation, and object must each occur
exactly in the sentence. Return an empty relations list if needed.

Sentence: {json.dumps(sent.text, ensure_ascii=False)}
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_predict": 384},
        },
        timeout=90,
    )
    response.raise_for_status()
    data = json.loads(response.json()["message"]["content"])
    results = []

    for item in data.get("relations", []):
        if not isinstance(item, dict):
            continue

        subject = item.get("subject", "").strip()
        predicate = item.get("relation", "").strip()
        object_name = item.get("object", "").strip()

        if not all((subject, predicate, object_name)):
            continue

        
        def find_spans(text, phrase):
            return [
                match.start()
                for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE)
            ]


        subject_positions = find_spans(sent.text, subject)
        relation_positions = find_spans(sent.text, predicate)
        object_positions = find_spans(sent.text, object_name)

        if not subject_positions or not relation_positions or not object_positions:
            continue

        subject_pos = subject_positions[0]
        relation_pos = relation_positions[0]
        object_pos = object_positions[-1]
        
        if min(subject_pos, relation_pos, object_pos) < 0:
            continue

        results.append({
            "subject": subject,
            "subject_label": "UNKNOWN",
            "relation": predicate,
            "object": object_name,
            "object_label": "UNKNOWN",
            "evidence_text": sent.text,
            "sentence_start": sent.start_char,
"sentence_end": sent.end_char,
            "subject_start": sent.start_char + subject_pos,
            "subject_end": sent.start_char + subject_pos + len(subject),
            "relation_start": sent.start_char + relation_pos,
            "relation_end": sent.start_char + relation_pos + len(predicate),
            "object_start": sent.start_char + object_pos,
            "object_end": sent.start_char + object_pos + len(object_name),
            "extraction_method": "ollama_llm",
            "validation_status": "unvalidated",
        })

    return results


def extract_relations(text):
    """Extract relations while preserving evidence and source offsets."""
    if not text or not text.strip():
        return []

    doc = nlp(text)
    results = []

    for sent in doc.sents:
        candidates = _dependency_relations(sent)

        if not candidates:
            try:
                candidates = _llm_relations(sent)
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                print(f"Ollama extraction skipped: {exc}")
                candidates = []

        results.extend(candidates)

    unique = {}
    for item in results:
        key = (
            item["sentence_start"],
            item["subject"].lower(),
            item["relation"].lower(),
            item["object"].lower(),
        )
        unique.setdefault(key, item)

    return list(unique.values())
