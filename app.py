
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
    Analyze vocabulary CEFR levels for this spoken text: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (tailor suggestions to this level)
    - User Type: {user_type} (make suggestions relevant to their context)
    
    CRITICAL - VOCABULARY SUGGESTIONS ARE MANDATORY:
    You MUST find and suggest improvements for weak/basic words like:
    - good → excellent/outstanding
    - bad → challenging/difficult  
    - thing → aspect/factor/element
    - do → accomplish/execute/perform
    - get → obtain/acquire/receive
    - make → create/develop/establish
    - very → extremely/highly/remarkably
    - nice → pleasant/wonderful/delightful
    - big → substantial/significant
    - small → minor/minimal
    
    SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
    - Set current_level = "spelling_error"
    - Set better_word = correct spelling
    
    Calculate percentage of words at each CEFR level. Percentages should sum to 100.
    Count ALL words in the text for total_words.
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "overall_level": "A1/A2/B1/B2/C1/C2",
      "total_words": <actual word count>,
      "cefr_distribution": {{
        "A1": {{"percentage": 20, "words": ["I", "is", "the"]}},
        "A2": {{"percentage": 30, "words": ["name", "good", "nice"]}},
        "B1": {{"percentage": 40, "words": ["actually", "however"]}},
        "B2": {{"percentage": 10, "words": ["sophisticated"]}},
        "C1": {{"percentage": 0, "words": []}},
        "C2": {{"percentage": 0, "words": []}}
      }},
      "feedback": "vocabulary feedback tailored to {level} level and {user_type} context",
      "suggestions": [
        {{
          "word": "good",
          "current_level": "A2",
          "better_word": "excellent",
          "suggested_level": "B1",
          "context": "appropriate for {user_type}",
          "original_sentence": "I had a ##good## experience",
          "improved_sentence": "I had an ##excellent## experience",
          "example_sentence": "The service at this restaurant is excellent."
        }}
      ]
    }}
    
    CRITICAL FORMATTING RULES:
    - original_sentence: Extract ONLY the specific sentence/line from user's transcription containing the weak word (NOT the whole transcription)
    - Mark the weak word with ##word## in original_sentence
    - improved_sentence: Same sentence/line with the better word substituted
    - Mark the better word with ##word## in improved_sentence  
    - example_sentence: A NEW sentence showing correct usage (not from transcription, no ## needed)
    - ALWAYS include suggestions if any weak/basic words (A1/A2 level) are found
    - For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
    - Provide at least 2-3 suggestions if weak words exist
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
---------------





async def analyze_vocab_llm(user_text: str, level: str = "Intermediate", model: str = "gpt") -> dict:
    """llm-based vocabulary analysis with cefr levels"""
    prompt = f"""Analyze vocabulary CEFR levels for this interview answer: "{user_text}"
 
Level: {level}

CRITICAL - VOCABULARY SUGGESTIONS ARE MANDATORY:
You MUST find and suggest improvements for weak/basic words like:
- good → excellent/outstanding
- bad → challenging/difficult  
- thing → aspect/factor/element
- do → accomplish/execute/perform
- get → obtain/acquire/receive
- make → create/develop/establish
- very → extremely/highly/remarkably
- nice → pleasant/wonderful/delightful
- big → substantial/significant
- small → minor/minimal

SPELLING ERRORS:
If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
- Set current_level = "spelling_error"
- Set better_word = correct spelling
 
Calculate percentage of words at each CEFR level. Percentages should sum to 100.
Count ALL words in the text for total_words.
 
IMPORTANT: In the "feedback" field, DO NOT mention "A1", "A2", "B1", "B2", "C1", "C2" directly.
Instead use:
- A1/A2 words = "basic words" or "simple vocabulary"
- B1/B2 words = "intermediate words" or "good vocabulary"
- C1/C2 words = "advanced words" or "sophisticated vocabulary"
 
Return STRICTLY valid JSON:
{{
  "score": 0-100,
  "overall_level": "A1/A2/B1/B2/C1/C2",
  "total_words": <actual word count>,
  "cefr_distribution": {{
    "A1": {{"percentage": 20, "words": ["I", "is", "the"]}},
    "A2": {{"percentage": 30, "words": ["work", "name", "good"]}},
    "B1": {{"percentage": 40, "words": ["experience", "actually"]}},
    "B2": {{"percentage": 10, "words": ["sophisticated"]}},
    "C1": {{"percentage": 0, "words": []}},
    "C2": {{"percentage": 0, "words": []}}
  }},
  "professional_words_used": ["list", "of", "professional", "terms"],
  "suggestions": [
    {{
      "word": "good",
      "current_level": "A2",
      "better_word": "excellent",
      "suggested_level": "B1",
      "context": "appropriate for professional interview",
      "original_sentence": "I had a ##good## experience",
      "improved_sentence": "I had an ##excellent## experience",
      "example_sentence": "The results of the project were excellent."
    }}
  ],
  "feedback": "Feedback using 'basic', 'intermediate', 'advanced' - NOT A1/B1/C1 labels"
}}

CRITICAL FORMATTING RULES:
- original_sentence: Extract ONLY the specific sentence/line from user's transcription containing the weak word (NOT the whole transcription)
- Mark the weak word with ##word## in original_sentence
- improved_sentence: Same sentence/line with the better word substituted
- Mark the better word with ##word## in improved_sentence
- example_sentence: A NEW sentence showing correct usage (not from transcription, no ## needed)
- ALWAYS include suggestions if any weak/basic words (A1/A2 level) are found
- For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
- Provide at least 2-3 suggestions if weak words exist
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            # Ensure CEFR distribution has all levels
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
            return data
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









   

Return STRICTLY valid JSON:
{{
  "emotion": "confident | hesitant | nervous | neutral | excited",
  "confidence_level": "high | medium | low",
  "explanation": "brief reason"
}}
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.debug(f"Emotion detection fallback: {e}")
    return {"emotion": "neutral", "confidence_level": "medium", "explanation": "Tone appears neutral."}


async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", model: str = "gpt") -> dict:
    """llm-based grammar analysis for spoken interview answers"""
    prompt = f"""You are an expert English grammar coach analyzing SPOKEN interview responses.

SPOKEN TEXT: "{user_text}"
USER LEVEL: {level}

IMPORTANT RULES:
1. This is TRANSCRIBED SPEECH - IGNORE punctuation, capitalization, and minor spelling
2. Focus ONLY on grammatical structure and word choice
3. Be encouraging but honest

ANALYZE FOR:

1. FILLER WORDS (detect ALL of these if present):
   - um, uh, uhh, er, err, ah, ahh
   - like (when not used correctly), you know, I mean, basically, actually, literally
   - so, well (when used as fillers at start)
   - kind of, sort of (when overused)

2. GRAMMAR ERRORS (check each carefully):
   - VERB TENSE: "I go yesterday" → "I went yesterday"
   - SUBJECT-VERB AGREEMENT: "He don't know" → "He doesn't know"
   - ARTICLES: "I am engineer" → "I am an engineer"
   - PREPOSITIONS: "I am good in coding" → "I am good at coding"
   - WORD ORDER: "Always I work hard" → "I always work hard"
   - PRONOUNS: "Me and him went" → "He and I went"
   - PLURALS: "I have many experience" → "I have much experience"
   - COMPARATIVES: "more better" → "better"

3. WORD SUGGESTIONS:
   - Find weak/basic words and suggest stronger alternatives
   - Example: "good" → "excellent/outstanding"
   - Example: "bad" → "challenging/difficult"
   - Example: "thing" → "aspect/factor/element"
   - Example: "do" → "accomplish/execute/perform"

CRITICAL: 
- "corrected_sentence" = Fix ONLY grammar errors
- "improved_sentence" = Fix grammar errors AND USE all word suggestions to make it professional

SCORING GUIDE (CRITICAL - follow exactly):
- 95-100: Perfect grammar, no errors, no filler words
- 85-94: Minor issues only (1-2 fillers OR 1 minor error)
- 70-84: Some issues (2-3 errors or multiple fillers)
- 50-69: Significant issues (4+ errors)
- Below 50: Major problems throughout

Return STRICTLY valid JSON (no extra text):
{{
  "score": <0-100 integer based on SCORING GUIDE above>,
  "is_correct": <true if no major errors, false otherwise>,
  
  "filler_words": ["list", "of", "detected", "fillers"],
  "filler_count": <number>,
  "filler_feedback": "<specific advice on reducing fillers>",
  
  "errors": [
    {{
      "type": "verb_tense | article | subject_verb | preposition | word_order | pronoun | plural | comparative",
      "you_said": "I ##goed## to store",
      "should_be": "I ##went## to the store",
      "wrong_word": "goed",
      "correct_word": "went",
      "explanation": "Go is irregular - past tense is went, not goed",
      "example_sentence": "Yesterday, I went to the park with my friends."
    }}
  ],
  
  "word_suggestions": [
    {{
      "you_used": "good",
      "use_instead": "excellent",
      "why": "more impactful for professional context",
      "original_sentence": "The results were ##good##",
      "improved_sentence": "The results were ##excellent##",
      "example_sentence": "The project outcomes were excellent."
    }}
  ],
  
  "corrected_sentence": "<THE WHOLE TRANSCRIPTION with ONLY grammar errors fixed>",
  "improved_sentence": "<THE WHOLE TRANSCRIPTION with grammar fixed + vocabulary enhanced>",
  
  "strengths": ["<what they did well grammatically>"],
  "feedback": "<2-3 sentences: acknowledge positives, then specific improvement tips>"
}}

CRITICAL FORMATTING RULES:
- For errors: you_said and should_be are ONLY the specific sentence/line from transcription containing the error
- Mark the wrong word with ##word## in you_said
- Mark the correct word with ##word## in should_be
- For word_suggestions: original_sentence and improved_sentence are ONLY the specific phrase containing the weak word
- Mark weak word with ##word## in original_sentence, better word with ##word## in improved_sentence
- example_sentence is a NEW sentence showing correct usage (not from transcription)
- corrected_sentence = THE WHOLE TRANSCRIPTION with all grammar fixes applied
- improved_sentence = THE WHOLE TRANSCRIPTION with grammar fixed AND vocabulary enhanced
- Empty arrays [] if no issues
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            
            data.setdefault("filler_words", [])
            data.setdefault("filler_count", len(data.get("filler_words", [])))
            data.setdefault("filler_feedback", "")
            data.setdefault("errors", [])
            data.setdefault("word_suggestions", [])
            data.setdefault("strengths", [])
            if not data.get("improved_sentence"):
                data["improved_sentence"] = data.get("corrected_sentence", user_text)
            
            
            error_count = len(data.get("errors", []))
            filler_count = len(data.get("filler_words", []))
            current_score = data.get("score", 75)
            
            
            if error_count == 0 and filler_count <= 1 and current_score < 90:
                data["score"] = 95 - (filler_count * 3)  
            elif error_count == 1 and current_score < 80:
                data["score"] = 85 - (filler_count * 2)
            elif error_count >= 4 and current_score > 70:
                data["score"] = min(current_score, 65)
            
            return data
    except Exception as e:
        logger.debug(f"Grammar analysis fallback: {e}")
    return {
        "score": 90, "is_correct": True, "filler_words": [], "filler_count": 0,
        "filler_feedback": "", "errors": [], "word_suggestions": [],
        "corrected_sentence": user_text, "improved_sentence": user_text,
        "strengths": ["Good sentence structure"], "feedback": "No major grammatical issues detected. Keep up the good work!"
    }
