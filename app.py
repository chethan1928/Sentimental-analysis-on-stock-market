async def analyze_vocab_llm(user_text: str, level: str = "Intermediate", model: str = "gpt") -> dict:
    """llm-based vocabulary analysis with cefr levels"""
    prompt = f"""Analyze vocabulary CEFR levels for this interview answer: "{user_text}"

Level: {level}

CRITICAL - SPELLING ERRORS:
If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
- Do NOT assign it a high CEFR level like C2
- Include it in "suggestions" with the CORRECT SPELLING as "better_word"

Calculate percentage of words at each CEFR level. Percentages should sum to 100.

IMPORTANT: In the "feedback" field, DO NOT mention "A1", "A2", "B1", "B2", "C1", "C2" directly.
Instead use:
- A1/A2 words = "basic words" or "simple vocabulary"
- B1/B2 words = "intermediate words" or "good vocabulary"
- C1/C2 words = "advanced words" or "sophisticated vocabulary"

CRITICAL FOR SUGGESTIONS:
- "original_sentence": Extract the EXACT phrase from the user's transcription that contains the weak word
- "improved_sentence": Show the SAME phrase with the better word substituted

Return STRICTLY valid JSON:
{{
  "score": 0-100,
  "overall_level": "A1/A2/B1/B2/C1/C2",
  "total_words": <word count>,
  "cefr_distribution": {{
    "A1": {{"percentage": 20, "words": ["I", "is"]}},
    "A2": {{"percentage": 30, "words": ["work", "name"]}},
    "B1": {{"percentage": 40, "words": ["experience"]}},
    "B2": {{"percentage": 10, "words": ["sophisticated"]}},
    "C1": {{"percentage": 0, "words": []}},
    "C2": {{"percentage": 0, "words": []}}
  }},
  "professional_words_used": ["list", "of", "professional", "terms"],
  "suggestions": [
    {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "original_sentence": "<extract from user's actual text>", "improved_sentence": "<same phrase with better word>"}}
  ],
  "feedback": "Feedback using 'basic', 'intermediate', 'advanced' - NOT A1/B1/C1 labels"
}}

IMPORTANT: For MISSPELLED words, set current_level = "spelling_error" and better_word = correct spelling
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.debug(f"Vocabulary analysis fallback: {e}")
    return {
        "score": 80, "overall_level": "B1", "total_words": len(user_text.split()),
        "cefr_distribution": {
            "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
            "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
            "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
        },
        "professional_words_used": [], "suggestions": [],
        "feedback": "Vocabulary analysis could not be completed."
    }
