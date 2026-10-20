def validate_relation(sentence, sentence_start, candidate):
    """
    Validate candidate text spans against an original sentence.

    Offsets are zero-based character offsets into the full source text.
    This validates span grounding, not scientific truth.
    """
    subject = candidate.get("subject", "").strip()
    predicate = candidate.get("predicate", "").strip()
    predicate_text = candidate.get("predicate_text", "").strip()
    object_name = candidate.get("object", "").strip()

    if not all([subject, predicate, predicate_text, object_name]):
        return None

    def find_span(value):
        start = sentence.find(value)
        if start < 0:
            return None
        return start, start + len(value)

    subject_span = find_span(subject)
    predicate_span = find_span(predicate_text)
    object_span = find_span(object_name)

    if not all([subject_span, predicate_span, object_span]):
        return None

    # Prevent the same text occurrence from serving as multiple arguments.
    spans = [subject_span, predicate_span, object_span]

    for i, first in enumerate(spans):
        for second in spans[i + 1:]:
            overlaps = (
                first[0] < second[1]
                and second[0] < first[1]
            )
            if overlaps:
                return None

    return {
        "subject": subject,
        "subject_label": candidate.get("subject_type") or "UNKNOWN",
        "relation": predicate,
        "object": object_name,
        "object_label": candidate.get("object_type") or "UNKNOWN",
        "evidence_text": sentence,
        "sentence_start": sentence_start,
        "sentence_end": sentence_start + len(sentence),
        "subject_start": sentence_start + subject_span[0],
        "subject_end": sentence_start + subject_span[1],
        "relation_start": sentence_start + predicate_span[0],
        "relation_end": sentence_start + predicate_span[1],
        "object_start": sentence_start + object_span[0],
        "object_end": sentence_start + object_span[1],
        "extraction_method": "ollama_llm",
        "validation_status": "unvalidated",
    }