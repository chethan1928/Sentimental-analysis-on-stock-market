








 async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", user_type: str = "student", model: str = "gpt") -> dict:
    """llm-based grammar analysis for spoken language with detailed suggestions"""
    
    level_context = ""
    if level == "Beginner":
        level_context = f"User is at Beginner level. Focus on basic grammar errors and simple corrections."
    elif level == "Intermediate":
        level_context = f"User is at Intermediate level. Check for intermediate grammar issues and provide detailed explanations."
    elif level == "Advanced":
        level_context = f"User is at Advanced level. Focus on subtle grammar nuances and advanced corrections."
    else:
        level_context = f"User is at Proficient level. Focus on native-like polish and professional refinement."
    
    user_type_context = ""
    if user_type == "professional":
        user_type_context = "This is a professional user. Provide business-appropriate grammar feedback."
    elif user_type == "student":
        user_type_context = "This is a student. Provide educational grammar feedback with clear explanations."
    
    prompt = f"""
    Analyze grammar in this SPOKEN text: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (adapt complexity of explanations accordingly)
    - User Type: {user_type} (make feedback relevant to their context)
    
    Based on the user's level and type, provide appropriate feedback.
    
    CRITICAL: This is transcribed speech. COMPLETELY IGNORE:
    - Punctuation errors
    - Capitalization errors
    - Spelling/typo errors
    - Vocabulary weakness (handled separately)
    
    CHECK ONLY FOR THESE GRAMMAR ERRORS:
    1. Filler words (um, uh, uhh, like, you know, I mean, basically, actually)
    2. Wrong prepositions (e.g., "good in" → "good at")
    3. Wrong verb tense (e.g., "I goed" → "I went")
    4. Subject-verb agreement (e.g., "He don't" → "He doesn't")
    5. Missing/wrong articles (e.g., "I am engineer" → "I am an engineer")
    6. Word order issues (e.g., "Always I work" → "I always work")
    7. Missing words (e.g., "I going" → "I am going")
    
    CRITICAL FORMATTING FOR EACH ERROR:
    - "you_said": Extract ONLY the specific sentence/line from transcription containing the error (NOT the whole paragraph)
    - Mark the wrong word with # like: "I #goed# to store"
    - "should_be": Show the SAME line with correction, mark correct word with # like: "I #went# to the store"
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "is_correct": true/false,
      "filler_words": ["um", "like"],
      "filler_count": 0,
      
      "you_said": "{user_text}",
      "you_should_say": "the grammatically correct version",
      
      "errors": [
        {{
          "type": "verb_tense/preposition/article/subject_verb/word_order/missing_word",
          "you_said": "I #goed# to store",
          "should_be": "I #went# to the store",
          "wrong_word": "goed",
          "correct_word": "went",
          "explanation": "Go is irregular - past tense is went, not goed"
        }}
      ],
      
      "word_suggestions": [
        {{"you_used": "good", "use_instead": "excellent", "why": "more impactful for {user_type}", "example": "The food was excellent."}}
      ],
      
      "corrected_sentence": "sentence with grammar fixed",
      "improved_sentence": "more natural version with better words",
      "feedback": "2-3 specific sentences about their grammar, tailored to {level} level and {user_type} context"
    }}
    
    RULES:
    - DO NOT include capitalization, punctuation, or spelling errors
    - DO NOT include vocabulary suggestions (handled separately)
    - Tailor feedback complexity to {level} level
    - Make suggestions relevant for {user_type}
    - For EACH error: you_said and should_be must be ONLY the specific line/sentence (NOT whole paragraph)
    - Mark wrong word with #word# in you_said, correct word with #word# in should_be
    - Empty arrays [] if no issues
    """
    try:
        raw = await call_llm(prompt, model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                if not data.get("improved_sentence"):
                    data["improved_sentence"] = data.get("corrected_sentence", user_text)
                
                
                if "errors" in data and isinstance(data["errors"], list):
                    cleaned_errors = []
                    for error in data["errors"]:
                        if isinstance(error, dict):
                            
                            error_type = error.get("type", "").lower()
                            if error_type in ["punctuation", "capitalization", "spelling", "typo"]:
                                continue
                            
                            
                            if not error.get("better_word"):
                                error.pop("better_word", None)
                                error.pop("explanation", None)
                            cleaned_errors.append(error)
                    data["errors"] = cleaned_errors
                
                return data
    except Exception as e:
        logger.debug(f"Grammar analysis fallback: {e}")
    word_count = len(user_text.split())
    return {
        "score": 70, "is_correct": True, "filler_words": [], "filler_count": 0,
        "you_said": user_text, "you_should_say": user_text, "errors": [],
        "word_suggestions": [], "corrected_sentence": user_text, "improved_sentence": user_text,
        "feedback": f"Analyzed {word_count} words. No major grammatical issues detected."
    }
    

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
    
    SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved"):
    - Set current_level = "spelling_error"
    - Set better_word = correct spelling
    
    Calculate percentage of words at each CEFR level. Percentages should sum to 100.
    
    CRITICAL FORMATTING FOR SUGGESTIONS:
    1. "original_sentence": Extract ONLY the specific sentence/phrase containing the weak word from transcription (NOT the whole paragraph)
    2. "improved_sentence": Show the SAME sentence/phrase with the better word
    3. Mark the weak word with # like #good# in original_sentence
    4. Mark the better word with # like #excellent# in improved_sentence
    
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
        {{
          "word": "good",
          "current_level": "A2",
          "better_word": "excellent",
          "suggested_level": "B1",
          "context": "appropriate for {user_type}",
          "original_sentence": "I had a #good# experience at work",
          "improved_sentence": "I had an #excellent# experience at work"
        }}
      ]
    }}
    
    IMPORTANT RULES:
    - ALWAYS include suggestions if any weak/basic words (A1/A2 level) are found
    - For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
    - original_sentence must be the EXACT line from user's text containing the word
    - Mark words with # like #word# for frontend display
    - Tailor suggestions to {level} level (don't suggest C1 words to A1 learners)
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











async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", model: str = "gpt") -> dict:
    """llm-based grammar analysis for spoken interview answers"""
    prompt = f"""You are an expert English grammar coach analyzing SPOKEN interview responses.

SPOKEN TEXT: "{user_text}"
USER LEVEL: {level}

IMPORTANT RULES:
1. This is TRANSCRIBED SPEECH - IGNORE punctuation, capitalization, and spelling errors
2. Focus ONLY on grammatical structure
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

CRITICAL FORMATTING FOR EACH ERROR:
- "you_said": Extract ONLY the specific sentence/line from transcription containing the error (NOT whole paragraph)
- Mark the wrong word with # like: "I #goed# to store"
- "should_be": Show the SAME line with correction, mark correct word with # like: "I #went# to the store"

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
      "you_said": "I #goed# to the office",
      "should_be": "I #went# to the office",
      "wrong_word": "goed",
      "correct_word": "went",
      "explanation": "<brief, friendly explanation>"
    }}
  ],
  
  "word_suggestions": [
    {{
      "weak_word": "<basic word user used>",
      "better_options": ["option1", "option2"],
      "example": "<show how to use in THEIR sentence with better word>"
    }}
  ],
  
  "corrected_sentence": "<grammatically correct version - fix errors only>",
  "improved_sentence": "<USE ALL word_suggestions to make it professional and polished>",
  
  "strengths": ["<what they did well grammatically>"],
  "feedback": "<2-3 sentences: acknowledge positives, then specific improvement tips>"
}}

RULES:
- DO NOT include capitalization, punctuation, or spelling errors
- For EACH error: you_said and should_be must be ONLY the specific line/sentence (NOT whole paragraph)
- Mark wrong word with #word# in you_said, correct word with #word# in should_be
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

SPELLING ERRORS:
If a word is MISSPELLED (e.g., "awareded", "recieved"):
- Set current_level = "spelling_error"
- Set better_word = correct spelling

Calculate percentage of words at each CEFR level. Percentages should sum to 100.

IMPORTANT: In the "feedback" field, DO NOT mention "A1", "A2", "B1", "B2", "C1", "C2" directly.
Instead use:
- A1/A2 words = "basic words" or "simple vocabulary"
- B1/B2 words = "intermediate words" or "good vocabulary"
- C1/C2 words = "advanced words" or "sophisticated vocabulary"

CRITICAL FORMATTING FOR SUGGESTIONS:
1. "original_sentence": Extract ONLY the specific sentence/phrase containing the weak word (NOT whole paragraph)
2. "improved_sentence": Show the SAME sentence/phrase with the better word
3. Mark the weak word with # like #good# in original_sentence
4. Mark the better word with # like #excellent# in improved_sentence

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
    {{
      "word": "good",
      "current_level": "A2",
      "better_word": "excellent",
      "suggested_level": "B1",
      "original_sentence": "I had a #good# experience at work",
      "improved_sentence": "I had an #excellent# experience at work"
    }}
  ],
  "feedback": "Feedback using 'basic', 'intermediate', 'advanced' - NOT A1/B1/C1 labels"
}}

IMPORTANT RULES:
- ALWAYS include suggestions if any weak/basic words (A1/A2 level) are found
- For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
- original_sentence must be the EXACT line from user's text containing the word
- Mark words with # like #word# for frontend display
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

