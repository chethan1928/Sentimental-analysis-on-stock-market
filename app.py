def normalize_language_code(lang: Optional[str], default: str = "en") -> str:
    """Normalize language input (name/code) to a 2-3 letter code for translators/Whisper."""
    if not lang or not isinstance(lang, str):
        return default

    value = lang.strip().lower()
    if not value:
        return default

    mapping = load_language_mapping() or {}
    if value in mapping:
        return mapping[value]

    # Handle full tags like "te-IN" or "en_US"
    for sep in ("-", "_"):
        if sep in value:
            base = value.split(sep)[0]
            if base in mapping:
                return mapping[base]
            if base.isalpha() and 2 <= len(base) <= 3:
                return base

    if value.isalpha() and 2 <= len(value) <= 3:
        return value

    return default
