
async def analyze_vocab_llm(user_text: str, user_type: str = "student", level: str = "Intermediate", model: str = "gpt") -> dict:
    """llm-based vocabulary analysis with cefr levels and percentages"""
    
    level_context = ""
    if level == "Beginner":
        level_context = "User is at Beginner level. Suggest simple vocabulary improvements."
    elif level == "Intermediate":
        level_context = "User is at Intermediate level. Suggest intermediate-level vocabulary enhancements."
    elif level == "Advanced":
        level_context = "User is at Advanced level. Suggest sophisticated vocabulary alternatives."
    else:
        level_context = "User is at Proficient level. Suggest native-like vocabulary refinements."
    
    user_type_context = ""
    if user_type == "professional":
        user_type_context = "This is a professional user. Suggest business-appropriate vocabulary."
    elif user_type == "student":
        user_type_context = "This is a student. Suggest academic-appropriate vocabulary."
    
    prompt = f"""
    Analyze vocabulary CEFR levels for: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (tailor suggestions to this level)
    - User Type: {user_type} (make suggestions relevant to their context)
    
    CRITICAL - SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
    - Do NOT assign it a high CEFR level like C2
    - Include it in "suggestions" with the CORRECT SPELLING as "better_word"
    
    Calculate percentage of words at each CEFR level.
    Percentages should sum to 100.
    
    CRITICAL FOR SUGGESTIONS:
    - "original_sentence": Extract the EXACT phrase from the user's transcription that contains the weak word
    - "improved_sentence": Show the SAME phrase with the better word substituted
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "overall_level": "A1/A2/B1/B2/C1/C2",
      "total_words": 10,
      "cefr_distribution": {{
        "A1": {{"percentage": 20, "words": ["I", "is"]}},
        "A2": {{"percentage": 30, "words": ["name", "good"]}},
        "B1": {{"percentage": 40, "words": ["actually", "however"]}},
        "B2": {{"percentage": 10, "words": ["sophisticated"]}},
        "C1": {{"percentage": 0, "words": []}},
        "C2": {{"percentage": 0, "words": []}}
      }},
      "feedback": "vocabulary feedback tailored to {level} level and {user_type} context",
      "suggestions": [
        {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "context": "appropriate for {user_type}", "original_sentence": "<extract from user's actual text>", "improved_sentence": "<same phrase with better word>"}}
      ]
    }}
    
    IMPORTANT:
    - For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
    - Tailor suggestions to {level} level (don't suggest C1 words to A1 learners)
    - Make suggestions relevant for {user_type} context
    """
    try:
        raw = await call_llm(prompt, model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                
                default_cefr = {
                    "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
                    "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
                    "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
                }
                if "cefr_distribution" not in data or not isinstance(data.get("cefr_distribution"), dict):
                    data["cefr_distribution"] = default_cefr
                else:
                    
                    for level_key in default_cefr:
                        if level_key not in data["cefr_distribution"]:
                            data["cefr_distribution"][level_key] = default_cefr[level_key]
                        elif not isinstance(data["cefr_distribution"][level_key], dict):
                            data["cefr_distribution"][level_key] = default_cefr[level_key]
                        else:
                            
                            if "percentage" not in data["cefr_distribution"][level_key]:
                                data["cefr_distribution"][level_key]["percentage"] = 0
                            if "words" not in data["cefr_distribution"][level_key]:
                                data["cefr_distribution"][level_key]["words"] = []
                return data
    except:
        pass
    return {
        "score": 70, "overall_level": "B1", "total_words": 0,
        "cefr_distribution": {
            "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
            "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
            "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
        },
        "feedback": "", "suggestions": []
    }
