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
    Analyze vocabulary CEFR levels for this user's spoken text: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (tailor suggestions to this level)
    - User Type: {user_type} (make suggestions relevant to their context)
    
    CRITICAL - FIND WEAK/BASIC WORDS:
    You MUST identify and suggest improvements for weak/basic words found in the transcription:
    - good → excellent/outstanding
    - bad → challenging/difficult  
    - thing → aspect/factor/element
    - do/did → accomplish/execute/perform
    - get/got → obtain/acquire/receive
    - make/made → create/develop/establish
    - very → extremely/highly/remarkably
    - nice → pleasant/wonderful/delightful
    - big → substantial/significant
    - small → minor/minimal
    
    SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
    - Set current_level = "spelling_error"
    - Set better_word = correct spelling
    
    Calculate percentage of words at each CEFR level. Percentages should sum to 100.
    
    CRITICAL FORMATTING FOR SUGGESTIONS - USE ACTUAL TRANSCRIPTION:
    - Look at the user's ACTUAL transcription text above
    - Find the EXACT sentence/line containing the weak word
    - original_sentence: Copy that EXACT line from transcription, mark weak word with ##
    - improved_sentence: Same EXACT line with better word substituted, mark with ##
    - DO NOT make up sentences - use the user's ACTUAL words
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "overall_level": "A1/A2/B1/B2/C1/C2",
      "total_words": <word count>,
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
        {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "context": "appropriate for {user_type}", "original_sentence": "<EXACT line from transcription with ##good##>", "improved_sentence": "<same line with ##excellent##>"}}
      ]
    }}
    
    IMPORTANT RULES:
    - ALWAYS include suggestions if ANY weak/basic words are found in transcription
    - original_sentence MUST be from the user's ACTUAL transcription (not made up)
    - Extract ONLY the specific line containing the word (NOT the whole text)
    - Mark words with ## like ##word## for frontend display
    - For MISSPELLED words: current_level = "spelling_error"
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
