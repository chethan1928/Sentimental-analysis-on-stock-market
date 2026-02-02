async def transcribe_audio_file(audio_file, target_lang: str = "en") -> str:
    """Helper function to transcribe audio file using Whisper - eliminates code duplication"""
    audio_file.file.seek(0)
    logger.info(f"[transcribe_audio_file] Starting transcription, target_lang: {target_lang}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        temp_upload = tmp.name
    logger.info(f"[transcribe_audio_file] Audio saved to temp: {temp_upload}, size: {os.path.getsize(temp_upload)} bytes")
    
    audio_path = None
    try:
        
        def convert_audio():
            audio = AudioSegment.from_file(temp_upload)
            audio = audio.set_frame_rate(16000).set_channels(1)
            converted_path = temp_upload.replace('.tmp', '_converted.wav')
            audio.export(converted_path, format="wav")
            return converted_path
        
        audio_path = await asyncio.to_thread(convert_audio)
        logger.info(f"[transcribe_audio_file] Converted audio: {audio_path}, size: {os.path.getsize(audio_path)} bytes")
        os.unlink(temp_upload)  
        
        # Pass language to Whisper for proper transcription in target language
        # If target_lang is specified, use it; otherwise let Whisper auto-detect
        whisper_lang = normalize_language_code(target_lang, default=None) if target_lang != "en" else None
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, language=whisper_lang
        )
        user_text = " ".join([seg.text for seg in segments]).strip()
        detected_lang = info.language if hasattr(info, 'language') else 'unknown'
        logger.info(f"[transcribe_audio_file] Transcription result (detected: {detected_lang}): '{user_text[:100] if user_text else 'empty'}'")
        
        return user_text
    except Exception as e:
        logger.error(f"[transcribe_audio_file] Audio transcription failed: {e}")
        return ""
    finally:
        
        for path in [temp_upload, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"[transcribe_audio_file] Failed to cleanup {path}: {e}")



async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", user_type: str = "student", model: str = "gpt", target_lang: str = "en") -> dict:
    """llm-based grammar analysis for spoken language with detailed suggestions"""
    
    # Normalize target_lang for response language instruction
    response_lang_instruction = ""
    if target_lang and target_lang.lower() != "en":
        response_lang_instruction = f"\n\nIMPORTANT: Provide ALL feedback text, explanations, and suggestions in {target_lang} language. The JSON keys should remain in English, but all text values (feedback, explanations, etc.) must be in {target_lang}."
    
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
    
    CRITICAL FORMATTING FOR EACH ERROR - MUST USE # MARKERS:
    - "you_said": Copy the EXACT sentence from transcription, mark the WRONG PART with # on both sides
    - "should_be": Show the SAME sentence with corrections, mark the CORRECTED PART with # on both sides
    
    EXAMPLES OF CORRECT # MARKING:
    - Single word error: "I #goed# to store" → "I #went# to the store"
    - Word order error: "#Always I work#" → "#I always work#" (mark entire rearranged phrase)
    - Missing word: "I #going#" → "I #am going#" (mark where word was added)
    - Multiple errors: "I #goed# #yesterday store#" → "I #went# #to the store yesterday#"
    - Article missing: "I am #engineer#" → "I am #an engineer#"
    
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
          "you_said": "#Always I work# hard",
          "should_be": "#I always work# hard",
          "wrong_word": "Always I work",
          "correct_word": "I always work",
          "explanation": "In English, adverbs of frequency come after the subject"
        }}
      ],
      
      "word_suggestions": [
        {{"you_used": "good", "use_instead": "excellent", "why": "more impactful for {user_type}", "example": "The food was excellent."}}
      ],
      
      "corrected_sentence": "sentence with grammar fixed",
      "improved_sentence": "more natural version with better words",
      "feedback": "2-3 specific sentences about their grammar, tailored to {level} level and {user_type} context"
    }}{response_lang_instruction}
    
    RULES:
    - DO NOT include capitalization, punctuation, or spelling errors
    - DO NOT include vocabulary suggestions (handled separately)
    - Tailor feedback complexity to {level} level
    - Make suggestions relevant for {user_type}
    - ALWAYS use the user's ACTUAL transcription for you_said (copy from above)
    - For WORD ORDER errors: mark the ENTIRE phrase that needs rearranging with #phrase#
    - For SINGLE WORD errors: mark just the word with #word#
    - Empty arrays [] if no issues
    """
    # Retry up to 3 times if JSON parsing fails
    for attempt in range(3):
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
        except json.JSONDecodeError as e:
            logger.warning(f"[analyze_grammar_llm] JSON parse error (attempt {attempt+1}/3): {e}")
            continue  # Retry
        except Exception as e:
            logger.error(f"[analyze_grammar_llm] Error: {e}")
            break  # Don't retry on other errors
    
    word_count = len(user_text.split())
    return {
        "score": 70, "is_correct": True, "filler_words": [], "filler_count": 0,
        "you_said": user_text, "you_should_say": user_text, "errors": [],
        "word_suggestions": [], "corrected_sentence": user_text, "improved_sentence": user_text,
        "feedback": f"Analyzed {word_count} words. No major grammatical issues detected."
    }
    

async def analyze_vocab_llm(user_text: str, user_type: str = "student", level: str = "Intermediate", model: str = "gpt", target_lang: str = "en") -> dict:
    """llm-based vocabulary analysis with cefr levels and percentages"""

    # Normalize target_lang for response language instruction
    response_lang_instruction = ""
    if target_lang and target_lang.lower() != "en":
        response_lang_instruction = f"\n\nIMPORTANT: Provide ALL feedback text, context descriptions, and suggestions in {target_lang} language. The JSON keys should remain in English, but all text values (feedback, context, etc.) must be in {target_lang}."
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

    CRITICAL - YOU MUST FIND AND SUGGEST IMPROVEMENTS FOR WEAK/BASIC WORDS:
    Scan the transcription above and identify ANY of these weak words:
    - good, nice, bad, thing, things, stuff
    - do, did, does, doing, done
    - get, got, gets, getting  
    - make, made, makes, making
    - very, really, pretty, quite
    - big, small, little, a lot
    - said, told, asked
    - went, go, goes, going
    - want, wanted, need, needed
    - like, liked, think, thought

    For EACH weak word found, you MUST add a suggestion in the suggestions array.

    SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
    - Set current_level = "spelling_error"
    - Set better_word = correct spelling

    CRITICAL FORMATTING FOR SUGGESTIONS - MUST USE # MARKERS:
    - original_sentence: Copy the EXACT sentence from transcription containing the weak word, mark it with #word#
    - improved_sentence: Same sentence with better word, mark it with #better_word#
    - ALWAYS use # on both sides of the word
    - Example: original_sentence: "The food was #good#" → improved_sentence: "The food was #excellent#"

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
        {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "context": "appropriate for {user_type}", "original_sentence": "The food was #good# here", "improved_sentence": "The food was #excellent# here"}}
      ]
    }}

    IMPORTANT RULES:
    - suggestions array MUST NOT be empty if ANY weak words are found in the transcription
    - If you find "good", "nice", "thing", "get", "make", "very", etc. - add a suggestion for EACH
    - original_sentence MUST be copied from the user's ACTUAL transcription (above)
    - ALWAYS mark words with #word# (single hash on both sides)
    - For MISSPELLED words: current_level = "spelling_error"{response_lang_instruction}
    """
    
    for attempt in range(3):
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
        except json.JSONDecodeError as e:
            logger.warning(f"[analyze_vocab_llm] JSON parse error (attempt {attempt+1}/3): {e}")
            continue  
        except Exception as e:
            logger.error(f"[analyze_vocab_llm] Error: {e}")
            break  
    
    return {
        "score": 70, "overall_level": "B1", "total_words": 0,
        "cefr_distribution": {
            "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
            "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
            "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
        },
        "feedback": "", "suggestions": []
    }


async def analyze_pronunciation_llm(audio_path: str = None, spoken_text: str = None, level: str = "B1", user_type: str = "student", model: str = "gpt", target_language: str = "en") -> dict:
    """pronunciation analysis using Whisper word-level confidence to detect mispronounced words"""
    if not audio_path:
        return {
            "accuracy": 70, "transcription": spoken_text or "", 
            "whisper_detection": {"detected_language": "unknown", "language_probability": 0, "original_transcription": spoken_text or ""},
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": spoken_text.split() if spoken_text else [],
            "feedback": "No audio provided for pronunciation analysis",
            "tips": ["Record audio for pronunciation feedback"], 
            "mispronounced_count": 0, "level": level, "user_type": user_type
        }
    
    try:
        target_lang = normalize_language_code(target_language, default="en")

        # Pass language to Whisper for proper transcription in target language
        whisper_lang = target_lang if target_lang and target_lang != "en" else None
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, word_timestamps=True, language=whisper_lang
        )
        
        
        words_data = []
        transcription = ""
        for seg in segments:
            transcription += seg.text + " "
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip().lower(),
                        "confidence": w.probability,
                        "start": w.start,
                        "end": w.end
                    })
        
        transcription = transcription.strip()
        
        if not words_data:
            return {
                "accuracy": 0, "transcription": transcription, 
                "word_pronunciation_scores": [],
                "words_to_practice": [], "well_pronounced_words": [],
                "feedback": "No speech detected in audio",
                "tips": ["Speak clearly into the microphone"], 
                "mispronounced_count": 0, "level": level, "user_type": user_type
            }
        
        
        
        CONFIDENCE_THRESHOLD = 0.70
        
        mispronounced_words = []
        well_pronounced = []
        all_words_pronunciation = []  
        
        for wd in words_data:
            word = wd["word"].strip(".,!?")
            if len(word) < 2:  
                continue
            
            
            pronunciation_percentage = round(wd["confidence"] * 100, 1)
            
            
            if pronunciation_percentage >= 90:
                status = "excellent"
            elif pronunciation_percentage >= 70:
                status = "good"
            elif pronunciation_percentage >= 50:
                status = "needs_improvement"
            else:
                status = "poor"
            
            all_words_pronunciation.append({
                "word": word,
                "pronunciation_percentage": pronunciation_percentage,
                "status": status
            })
                
            if wd["confidence"] < CONFIDENCE_THRESHOLD:
                mispronounced_words.append({
                    "word": word,
                    "confidence": round(wd["confidence"] * 100, 1),
                    "issue": "unclear pronunciation" if wd["confidence"] < 0.5 else "slight pronunciation issue"
                })
            else:
                well_pronounced.append(word)
        
        
        if words_data:
            avg_confidence = sum(w["confidence"] for w in words_data) / len(words_data)
            accuracy = int(avg_confidence * 100)
        else:
            accuracy = 70
        
        
        # Add target language instruction for LLM response
        response_lang_instruction = ""
        if target_language and target_language.lower() != "en":
            response_lang_instruction = f"\n\nIMPORTANT: Provide ALL feedback text, tips, and improvement suggestions in {target_language} language. The JSON keys should remain in English, but all text values (feedback, tips, etc.) must be in {target_language}."
        
        llm_prompt = f"""You are a pronunciation coach.

USER CONTEXT:
- Level: {level}
- User Type: {user_type}

TRANSCRIPTION: "{transcription}"

PER-WORD PRONUNCIATION SCORES (confidence-based):
{all_words_pronunciation}

MISPRONOUNCED WORDS (low confidence from speech recognition):
{mispronounced_words if mispronounced_words else "None - all words were clear!"}

WELL PRONOUNCED WORDS: {well_pronounced[:10]}

OVERALL ACCURACY: {accuracy}%

For each word, analyze their pronunciation percentage and provide specific guidance.

Return STRICTLY valid JSON:
{{
    "word_analysis": [
        {{
            "word": "the word",
            "pronunciation_match": 85.5,
            "rating": "excellent/good/needs_improvement/poor",
            "phonetic_guide": "how to pronounce: ex-AM-ple",
            "improvement_tip": "specific tip if needed, or null if pronunciation is good"
        }}
    ],
    "words_to_practice": [
        {{
            "word": "the word",
            "how_to_say": "syllable breakdown with stress: ex-AM-ple",
            "tip": "specific tip to pronounce this word better"
        }}
    ],
    "well_pronounced_words": ["word1", "word2"],
    "feedback": "2-3 encouraging sentences about their pronunciation",
    "tips": ["general pronunciation tip 1", "general tip 2"]
}}{response_lang_instruction}"""

        
        llm_data = None
        for attempt in range(3):
            try:
                llm_response = await call_llm(llm_prompt, mode="strict_json", timeout=30, model=model)
                json_match = re.search(r'\{[\s\S]*\}', llm_response)
                if json_match:
                    llm_data = json.loads(json_match.group())
                    break  
                else:
                    raise ValueError("No JSON found in response")
            except json.JSONDecodeError as e:
                logger.warning(f"[analyze_pronunciation_llm] JSON parse error (attempt {attempt+1}/3): {e}")
                continue  
            except Exception as llm_error:
                logger.warning(f"[analyze_pronunciation_llm] LLM error (attempt {attempt+1}/3): {llm_error}")
                continue  
        
        
        if not llm_data:
            logger.error(f"[analyze_pronunciation_llm] All 3 retries failed, using fallback")
            llm_data = {
                "words_to_practice": [
                    {
                        "word": w["word"],
                        "how_to_say": f"Say '{w['word']}' more clearly",
                        "tip": f"Confidence was {w['confidence']}%. Speak slower and clearer."
                    } for w in mispronounced_words[:5]
                ],
                "well_pronounced_words": well_pronounced[:5],
                "feedback": f"Pronunciation accuracy: {accuracy}%. " + (
                    f"Focus on: {', '.join([w['word'] for w in mispronounced_words[:3]])}" 
                    if mispronounced_words else "Great clarity!"
                ),
                "tips": ["Speak slowly and clearly", "Stress syllables properly"]
            }
        
        
        llm_word_analysis = llm_data.get("word_analysis", [])
        
        
        word_pronunciation_scores = []
        llm_analysis_map = {w.get("word", "").lower(): w for w in llm_word_analysis}
        
        for wp in all_words_pronunciation:
            word_key = wp["word"].lower()
            llm_info = llm_analysis_map.get(word_key, {})
            word_pronunciation_scores.append({
                "word": wp["word"],
                "pronunciation_match_percentage": wp["pronunciation_percentage"],
                "status": wp["status"],
                "phonetic_guide": llm_info.get("phonetic_guide", ""),
                "improvement_tip": llm_info.get("improvement_tip", "")
            })
        
        return {
            "accuracy": accuracy, 
            "transcription": transcription, 
            "whisper_detection": {
                "detected_language": info.language if hasattr(info, 'language') else "unknown",
                "language_probability": round(info.language_probability * 100, 1) if hasattr(info, 'language_probability') else 0,
                "original_transcription": transcription
            },
            "word_pronunciation_scores": word_pronunciation_scores,
            "words_to_practice": llm_data.get("words_to_practice", []),
            "well_pronounced_words": llm_data.get("well_pronounced_words", well_pronounced),
            "feedback": llm_data.get("feedback", "Analysis complete."),
            "tips": llm_data.get("tips", []),
            "mispronounced_count": len(mispronounced_words),
            "confidence_data": [{"word": w["word"], "confidence": w["confidence"]} for w in mispronounced_words],
            "level": level, 
            "user_type": user_type
        }
        
    except Exception as e:
        logger.error(f"Pronunciation error: {e}")
        return {
            "accuracy": 70, "transcription": spoken_text or "", 
            "whisper_detection": {"detected_language": "unknown", "language_probability": 0, "original_transcription": spoken_text or ""},
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": [],
            "feedback": f"Could not analyze pronunciation: {str(e)}",
            "tips": ["Ensure clear audio recording"],
            "mispronounced_count": 0,
            "level": level, "user_type": user_type
        }

):
    """
    fluent language practice api - CONVERSATIONAL ONBOARDING
    
    flow:
    1. first call (no audio/text, no scenario/level): Bot greets and asks for scenario
    2. user provides scenario: Bot asks for level
    3. user provides level: practice begins with first question
    4. subsequent calls: normal practice with analysis
    5. termination phrase: ends session with summary
    """
    try:
        user_id = current_user.id if current_user else None
        audio_path = None
        user_text = ""

        if not session_id or session_id.strip() == "" or session_id == "string":
            session_id = str(uuid.uuid4())
        
        
        session = await db.get_user_session(session_id)
        session_exists = session is not None
        native_language = session.get("native_language", native_language) if session else native_language
        target_language = session.get("target_language", target_language) if session else target_language
        
        
        if session_exists and session.get("status") == "completed":
            return {"status": "error", "session_id": session_id, "error": "This session has ended. Please start a new conversation."}
        
        
        if not session_exists:
            
            
            if scenario:
                initial_state = "collecting_level"  
            else:
                initial_state = "welcome"  
            
            session = {
                "state": initial_state,
                "name": name, 
                "level": None,  
                "scenario": scenario,  
                "native_language": native_language, 
                "target_language": target_language,
                "user_type": user_type, 
                "chat_history": [],
                "scores": {"grammar": 0, "vocabulary": 0, "pronunciation": 0, "fluency": 0, "total_wpm": 0, "count": 0, "audio_count": 0},
                "current_question": None, "current_hint": None, "turn_number": 0,
                "last_overall_score": None, "retry_count": 0,
                "attempts": [], "turn_history": [],
                "onboarding_retry": 0
            }
            await db.create_session(
                session_id=session_id,
                session_type="fluent",
                data=session,
                user_id=user_id,
                user_name=name
            )
            
            
            if scenario:
                ask_level = f"Great! We'll practice {scenario.replace('_', ' ')}. What's your English level? Beginner, Intermediate, Advanced, or Proficient?"
                ask_level_native = await translate_text(ask_level, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": ask_level})
                await db.update_session(session_id, session)
                
                ask_level_audio = await generate_tts_url(request, ask_level, target_language, voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_level",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "scenario": scenario,
                    "message": {"target": ask_level, "native": ask_level_native},
                    "audio_url": ask_level_audio
                }

        
        
        current_state = session.get("state", "practicing")
        
        
        
        
        if current_state == "welcome" and not audio_file and not text_input:
            greeting = f"Hi {name}! I'm {BOT_NAME} 🙂 What would you like to practice today? For example: ordering food, hotel check-in, casual conversation, or anything else!"
            
            greeting_native = await translate_text(greeting, "en", native_language)
            
            session["state"] = "collecting_scenario"
            session["chat_history"].append({"role": "assistant", "content": greeting})
            await db.update_session(session_id, session)
            
            greeting_audio = await generate_tts_url(request, greeting, target_language, voice_id=voice_id)
            
            return {
                "status": "onboarding",
                "step": "collecting_scenario",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "transcription": user_text,
                "message": {"target": greeting, "native": greeting_native},
                "audio_url": greeting_audio
            }
        
        
        
        
        if current_state == "collecting_scenario":
            user_text = text_input or ""
            if audio_file:
                audio_file.file.seek(0)
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                logger.error(f"[collecting_scenario] No speech detected from audio, session_id: {session_id}")
                return {"status": "error", "session_id": session_id, "error": "No speech detected. Please tell me what you'd like to practice."}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_scenario_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("scenario"):
                scenario = extraction["scenario"]
                session["scenario"] = scenario
                session["state"] = "collecting_level"
                session["onboarding_retry"] = 0
                
                
                ask_level = f"Great choice! We'll practice {scenario.replace('_', ' ')}. What's your level? Beginner, Intermediate, Advanced, or Proficient?"
                ask_level_native = await translate_text(ask_level, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": ask_level})
                await db.update_session(session_id, session)
                
                ask_level_audio = await generate_tts_url(request, ask_level, target_language, voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_level", 
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "scenario": scenario,
                    "message": {"target": ask_level, "native": ask_level_native},
                    "audio_url": ask_level_audio
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "Could you be more specific? For example: ordering food, shopping, hotel check-in, asking directions, or casual chat?"
                retry_native = await translate_text(retry_msg, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_scenario",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "retry": True,
                    "message": {"target": retry_msg, "native": retry_native}
                }
        
        
        
        
        if current_state == "collecting_level":
            user_text = text_input or ""
            if audio_file:
                audio_file.file.seek(0)
                logger.info(f"[collecting_level] Audio file received, size hint: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
                user_text = await transcribe_audio_file(audio_file, target_language)
                logger.info(f"[collecting_level] Transcription result: '{user_text[:100] if user_text else 'empty'}'")
            
            if not user_text.strip():
                logger.warning(f"[collecting_level] No speech detected from audio")
                return {"status": "error", "session_id": session_id, "error": "No speech detected. Please tell me your level."}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_level_from_text(user_text, model=model)
            level = extraction.get("level", "B1")
            session["level"] = level
            session["state"] = "practicing"
            session["onboarding_retry"] = 0
            
            
            scenario = session.get("scenario", "casual_conversation")
            question, hint = await generate_question_llm(level, scenario, name, target_language, model=model)
            
            
            level_display = LEVEL_DISPLAY.get(level, level)
            start_msg = f"Perfect! Let's start practicing {scenario.replace('_', ' ')} at {level_display} level."
            
            start_native, q_native, h_native = await asyncio.gather(
                translate_text(start_msg, "en", native_language),
                translate_text(question, target_language, native_language),
                translate_text(hint, target_language, native_language)
            )
            
            session["current_question"] = question
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": question})
            await db.update_session(session_id, session)
            
            question_audio = await generate_tts_url(request, question, target_language, voice_id=voice_id)
            
            return {
                "status": "practice_started",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "level": level_display,
                "scenario": scenario,
                "transcription": user_text,
                "message": {"target": start_msg, "native": start_native},
                "next_question": {"target": question, "native": q_native},
                "hint": {"target": hint, "native": h_native},
                "audio_url": question_audio
            }
        
        
        
        
        if skip_retry and not audio_file and not text_input:
            
            session["waiting_retry_decision"] = False
            session["retry_count"] = 0
            session["retry_clarify_count"] = 0  
            scenario = session.get("scenario", "casual_conversation")
            session_user_type = session.get("user_type", user_type)  
            follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, session_user_type, model=model)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            
            
            await db.update_session(session_id, session)
            
            return {
                "status": "continue", "session_id": session_id,
                "target_lang": target_language, "native_lang": native_language,
                "transcription": "(skipped)",
                "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                "grammar": {"score": 0, "is_correct": True, "you_said": "", "you_should_say": "", "errors": [], "word_suggestions": [], "corrected_sentence": "", "improved_sentence": "", "feedback": "Skipped"},
                "vocabulary": {"score": 0, "overall_level": "skipped", "cefr_distribution": {}, "feedback": "Skipped", "suggestions": []},
                "pronunciation": {"accuracy": 0, "total_words": 0, "words_to_practice": [], "well_pronounced_words": [], "feedback": "Skipped", "practice_sentence": "", "tips": []},
                "fluency": {"score": 0, "wpm": 0, "speed_status": "skipped", "original_text": "", "corrected_text": "", "improved_sentence": ""},
                "personalized_feedback": {"user_type": user_type, "message": "Skipped. Let's try the next question!", "improvement_areas": [], "strengths": [], "perfect_areas": [], "perfect_feedback": {}, "quick_tip": ""},
                "overall_score": 0, "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"]
            }
        
        
        if not audio_file and not text_input and current_state == "practicing" and session.get("turn_number", 0) == 0:
            scenario = session.get("scenario", "casual_conversation")
            level = session.get("level", "B1")
            greeting = f"Hey {name}! I am {BOT_NAME}. I am your {BOT_ROLE}. Let's practice {scenario.replace('_', ' ')}!"
            question, hint = await generate_question_llm(level, scenario, name, target_language, model=model)
            
            
            greeting_native, question_native, hint_native = await asyncio.gather(
                translate_text(greeting, "en", native_language),
                translate_text(question, target_language, native_language),
                translate_text(hint, target_language, native_language)
            )
            
            session["current_question"] = question
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": question})
            
            
            await db.update_session(session_id, session)
            
            return {
                "status": "conversation_started", "session_id": session_id,
                "target_lang": target_language, "native_lang": native_language,
                "greeting": {"target": greeting, "native": greeting_native},
                "next_question": {"target": question, "native": question_native},
                "hint": {"target": hint, "native": hint_native}
            }
        
        user_text = text_input or ""
        audio_analysis = None
        audio_path = None
        is_audio_input = audio_file is not None  

        
        if audio_file:
            audio_file.file.seek(0)
            logger.info(f"[practice] Audio file received, filename: {audio_file.filename}, size hint: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                shutil.copyfileobj(audio_file.file, tmp)
                temp_upload = tmp.name
            logger.info(f"[practice] Audio saved to temp file: {temp_upload}")
            
            try:
                
                def convert_audio():
                    audio = AudioSegment.from_file(temp_upload)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    converted_path = temp_upload.replace('.tmp', '_converted.wav')
                    audio.export(converted_path, format="wav")
                    return converted_path
                
                audio_path = await asyncio.to_thread(convert_audio)
                os.unlink(temp_upload)  
            except Exception as e:
                logger.error(f"[practice] Audio conversion failed: {e}")
                audio_path = temp_upload
            finally:
                
                
                if audio_path != temp_upload and os.path.exists(temp_upload):
                    try:
                        os.unlink(temp_upload)
                    except Exception as e:
                        logger.warning(f"[practice] Failed to cleanup temp file: {e}")
            
            
            session_level_for_audio = session.get("level", "B1")
            audio_analysis = await asyncio.to_thread(analyze_speaking_advanced, audio_path, session_level_for_audio)
            if audio_analysis.get("success") and audio_analysis.get("transcription"):
                user_text = audio_analysis.get("transcription")
            elif not user_text:
                
                try:
                    user_text = await asyncio.to_thread(
                        speech_to_text, audio_path, normalize_language_code(target_language, default="en")
                    )
                except Exception as e:
                    logger.error(f"[practice] Fallback speech_to_text failed: {e}")

        
        if not user_text or not user_text.strip():
            logger.error(f"[practice] No speech detected from audio, session_id: {session_id}, audio_file: {audio_file.filename if audio_file else 'None'}")
            return {"status": "error", "session_id": session_id, "error": "No speech detected. Please try again."}
        
        user_text = user_text.strip()
        session["chat_history"].append({"role": "user", "content": user_text})
        
        
        
        
        if session.get("waiting_retry_decision"):
            user_choice = user_text.lower().strip()
            
            
            cleaned_choice = user_choice.rstrip('.,!?')
            if cleaned_choice in TERMINATION_PHRASES:
                
                session["waiting_retry_decision"] = False
                count = max(1, session["scores"]["count"])
                audio_count = session["scores"].get("audio_count", 0)
                if not audio_count and (
                    session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
                ):
                    audio_count = count
                pronunciation_avg = int(session["scores"]["pronunciation"] / audio_count) if audio_count > 0 else 0
                fluency_avg = int(session["scores"]["fluency"] / audio_count) if audio_count > 0 else 0
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": pronunciation_avg if audio_count > 0 else None,
                    "fluency": fluency_avg if audio_count > 0 else None
                }
                
                if audio_count > 0:
                    overall = int(sum(v for v in final_scores.values() if v is not None) / 4)
                else:
                    overall = int((final_scores["grammar"] + final_scores["vocabulary"]) / 2)
                average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
                
                improvement_areas = [area for area, score in final_scores.items() if score is not None and score < 70]
                strengths = [area for area, score in final_scores.items() if score is not None and score >= 80]
                
                
                final_feedback_data = {
                    "final_scores": final_scores,
                    "overall_score": overall,
                    "average_wpm": average_wpm,
                    "strengths": strengths,
                    "improvement_areas": improvement_areas,
                    "total_turns": session.get("turn_number", 0)
                }
                
                
                await db.complete_session(session_id, final_feedback=final_feedback_data, overall_score=overall)
                
                return {
                    "status": "conversation_ended",
                    "session_id": session_id,
                    "target_lang": target_language, "native_lang": native_language,
                    "final_scores": final_scores,
                    "overall_score": overall,
                    "passing_score": PASSING_SCORE,
                    "average_wpm": average_wpm,
                    "wpm_status": "slow" if average_wpm < 110 else "normal" if average_wpm <= 160 else "fast",
                    "strengths": strengths,
                    "improvement_areas": improvement_areas,
                    "total_turns": session.get("turn_number", 0),
                    "message": {"target": "Session ended. Great practice!", "native": await translate_text("Session ended. Great practice!", "en", native_language)}
                }
            
            retry_keywords = ["yes", "retry", "practice", "again", "try", "redo", "repeat", "once more", "one more"]
            skip_keywords = ["no", "skip", "next", "move", "forward", "pass", "don't want", "not now", "let's move", "move on", "go ahead"]
            
            wants_retry = any(keyword in user_choice for keyword in retry_keywords)
            wants_skip = any(keyword in user_choice for keyword in skip_keywords)
            
            if wants_retry:
                
                session["waiting_retry_decision"] = False
                session["retry_clarify_count"] = 0
                session["is_retry_attempt"] = True  
                current_q = session.get("current_question", "")
                current_h = session.get("current_hint", "")
                session["chat_history"].append({"role": "assistant", "content": current_q})
                await db.update_session(session_id, session)
                
                
                retry_msg = await generate_retry_encouragement(
                    scenario=session.get("scenario", "conversation"),
                    retry_count=session.get("retry_count", 0),
                    previous_score=session.get("last_overall_score", 50),
                    user_name=session.get("name", "there"),
                    model=model
                )
                q_native, h_native, retry_msg_native = await asyncio.gather(
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_text(retry_msg, "en", native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language, "native_lang": native_language,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "message": {"target": retry_msg, "native": retry_msg_native},
                    "turn_number": session.get("turn_number", 0)
                }
            elif wants_skip:
                
                session["waiting_retry_decision"] = False
                session["retry_clarify_count"] = 0
                session["retry_count"] = 0
                session["is_retry_attempt"] = False  
                session["last_overall_score"] = None  
                scenario = session.get("scenario", "casual_conversation")
                follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, user_type, model=model)
                session["current_question"] = follow_up
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": follow_up})
                
                await db.update_session(session_id, session)
                
                
                skip_msg = await generate_skip_message(
                    scenario=scenario,
                    user_name=session.get("name", "there"),
                    model=model
                )
                skip_msg_native, follow_up_native, hint_native = await asyncio.gather(
                    translate_text(skip_msg, "en", native_language),
                    translate_text(follow_up, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language, "native_lang": native_language,
                    "message": {"target": skip_msg, "native": skip_msg_native},
                    "next_question": {"target": follow_up, "native": follow_up_native},
                    "hint": {"target": hint, "native": hint_native},
                    "turn_number": session["turn_number"]
                }
            else:
                
                clarify_count = session.get("retry_clarify_count", 0) + 1
                session["retry_clarify_count"] = clarify_count
                
                if clarify_count >= 3:
                    
                    session["waiting_retry_decision"] = False
                    session["retry_clarify_count"] = 0
                    scenario = session.get("scenario", "casual_conversation")
                    follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, user_type, model=model)
                    session["current_question"] = follow_up
                    session["current_hint"] = hint
                    session["chat_history"].append({"role": "assistant", "content": follow_up})
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "auto_skipped",
                        "session_id": session_id,
                        "target_lang": target_language, "native_lang": native_language,
                        "message": {"target": "Moving to the next question.", "native": await translate_text("Moving to the next question.", "en", native_language)},
                        "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                        "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                        "turn_number": session["turn_number"]
                    }
                else:
                    
                    level = session.get("level", "B1")
                    scenario = session.get("scenario", "casual_conversation")
                    
                    
                    if is_audio_input:
                        grammar, vocabulary, pronunciation = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, user_type=user_type, model=model, target_lang=target_language),
                            analyze_vocab_llm(user_text, level=level, user_type=user_type, model=model, target_lang=target_language),
                            analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=level, user_type=user_type, model=model, target_language=target_language)
                        )
                        
                        try:
                            audio_for_duration = AudioSegment.from_file(audio_path)
                            audio_duration = len(audio_for_duration) / 1000
                        except Exception as e:
                            logger.error(f"[practice] Failed to get audio duration: {e}")
                            word_count = len(user_text.split())
                            audio_duration = max(1, word_count / 2.5)
                        fluency = await analyze_fluency_metrics(user_text, audio_duration)
                    else:
                        
                        grammar, vocabulary = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, user_type=user_type, model=model, target_lang=target_language),
                            analyze_vocab_llm(user_text, level=level, user_type=user_type, model=model, target_lang=target_language)
                        )
                        pronunciation = None
                        fluency = None
                    
                    
                    if is_audio_input:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": pronunciation.get("score", pronunciation.get("accuracy", 70)) if pronunciation else 0,
                            "fluency": fluency.get("score", 70) if fluency else 0
                        }
                        
                        overall_score = int(scores["pronunciation"] * 0.30 + scores["grammar"] * 0.30 + scores["vocabulary"] * 0.20 + scores["fluency"] * 0.20)
                    else:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": None,
                            "fluency": None
                        }
                        
                        overall_score = int(scores["grammar"] * 0.50 + scores["vocabulary"] * 0.50)
                    
                    personalized_feedback = await generate_personalized_feedback(
                        user_type, overall_score, scores, user_text,
                        grammar=grammar, vocabulary=vocabulary, pronunciation=pronunciation, model=model
                    )
                    
                    if clarify_count == 1:
                        clarify_msg = "I didn't quite catch that. Say 'yes' to try again, or 'no' to skip to the next question."
                    else:
                        clarify_msg = "Please say 'yes' to retry the same question, or 'no' to move on."
                    
                    await db.update_session(session_id, session)
                    
                    
                    if is_audio_input and pronunciation:
                        (grammar_t, vocab_t, pron_t, feedback_t, clarify_native) = await asyncio.gather(
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                            translate_text(clarify_msg, "en", native_language)
                        )
                    else:
                        (grammar_t, vocab_t, feedback_t, clarify_native) = await asyncio.gather(
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                            translate_text(clarify_msg, "en", native_language)
                        )
                        pron_t = None
                    
                    return {
                        "status": "clarify_retry",
                        "session_id": session_id,
                        "target_lang": target_language, "native_lang": native_language,
                        "transcription": user_text,
                        "message": {"target": clarify_msg, "native": clarify_native},
                        "grammar": grammar_t,
                        "vocabulary": vocab_t,
                        "pronunciation": pron_t,
                        "fluency": fluency,
                        "personalized_feedback": feedback_t,
                        "overall_score": overall_score,
                        "clarify_count": clarify_count,
                        "turn_number": session.get("turn_number", 0)
                    }

        
        cleaned_text = user_text.lower().strip().rstrip('.,!?')
        is_termination = cleaned_text in TERMINATION_PHRASES
        
        if is_termination:
            count = max(1, session["scores"]["count"])
            audio_count = max(1, session["scores"].get("audio_count", 0))  
            
            
            has_audio_turns = session["scores"].get("audio_count", 0) > 0  
            
            if has_audio_turns:
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": int(session["scores"]["pronunciation"] / audio_count),  
                    "fluency": int(session["scores"]["fluency"] / audio_count)  
                }
                
                overall = int(
                    final_scores["pronunciation"] * 0.30 +
                    final_scores["grammar"] * 0.30 +
                    final_scores["vocabulary"] * 0.20 +
                    final_scores["fluency"] * 0.20
                )
                average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
            else:
                
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": None,
                    "fluency": None
                }
                
                overall = int(
                    final_scores["grammar"] * 0.50 +
                    final_scores["vocabulary"] * 0.50
                )
                average_wpm = 0

            
            
            if average_wpm < 110:
                wpm_status = "Slow - Try to speak a bit faster"
            elif average_wpm > 160:
                wpm_status = "Fast - Try to slow down for clarity"
            else:
                wpm_status = "Natural - Great speaking pace!"
            
            improvement_areas = []
            strengths = []
            for area, score in final_scores.items():
                if score is None:  
                    continue
                if score >= 85:
                    strengths.append(area)
                elif score < 70:
                    improvement_areas.append(area)
            
            
            turn_history_summary = []
            wpm_per_turn = []
            vocab_overall = {
                "A1": {"count": 0, "words": []},
                "A2": {"count": 0, "words": []},
                "B1": {"count": 0, "words": []},
                "B2": {"count": 0, "words": []},
                "C1": {"count": 0, "words": []},
                "C2": {"count": 0, "words": []}
            }
            
            for i, attempt in enumerate(session.get("attempts", []), 1):
                
                pron_data = attempt.get("pronunciation") or {}
                fluency_data = attempt.get("fluency") or {}
                vocab_data = attempt.get("vocabulary") or {}
                
                # Track WPM per turn
                turn_wpm = fluency_data.get("wpm", 0) if fluency_data else 0
                wpm_per_turn.append({"turn": i, "wpm": turn_wpm})
                
                # Aggregate CEFR vocabulary words
                cefr_dist = vocab_data.get("cefr_distribution", {}) if vocab_data else {}
                for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    level_data = cefr_dist.get(level, {})
                    if isinstance(level_data, dict):
                        words = level_data.get("words", [])
                        if isinstance(words, list):
                            vocab_overall[level]["words"].extend(words)
                            vocab_overall[level]["count"] = len(set(vocab_overall[level]["words"]))
                
                turn_history_summary.append({
                    "turn": i,
                    "grammar": attempt.get("grammar", {}).get("score", 0),
                    "vocabulary": attempt.get("vocabulary", {}).get("score", 0),
                    "pronunciation": pron_data.get("accuracy", 0) if pron_data else 0,
                    "fluency": fluency_data.get("score", 0) if fluency_data else 0,
                    "wpm": turn_wpm,
                    "overall": attempt.get("overall_score", 0),
                    "transcription": attempt.get("transcription", "")[:50]
                })
            
            # Deduplicate vocab words and calculate percentages
            total_vocab_words = sum(len(set(vocab_overall[level]["words"])) for level in vocab_overall)
            for level in vocab_overall:
                vocab_overall[level]["words"] = list(set(vocab_overall[level]["words"]))
                vocab_overall[level]["count"] = len(vocab_overall[level]["words"])
                vocab_overall[level]["percentage"] = round((vocab_overall[level]["count"] / total_vocab_words * 100), 1) if total_vocab_words > 0 else 0
            
            final_feedback_prompt = f"""You are {session.get('name', 'friend')}'s warm, encouraging language coach. Generate a HIGHLY PERSONALIZED and INTERACTIVE final session summary.

SESSION DATA:
- Student Name: {session['name']}
- User Type: {session.get('user_type', 'student')}
- Level: {session.get('level', 'B1')}
- Scenario: {session.get('scenario', 'conversation')}
- Total Turns: {session.get('turn_number', 0)}
- Final Scores: Grammar={final_scores['grammar']}%, Vocabulary={final_scores['vocabulary']}%, Pronunciation={final_scores['pronunciation'] if final_scores['pronunciation'] is not None else 'N/A'}%, Fluency={final_scores['fluency'] if final_scores['fluency'] is not None else 'N/A'}%
- Overall Score: {overall}/100
- Average WPM: {average_wpm}
- Strengths: {strengths if strengths else 'Building foundation'}
- Areas to Improve: {improvement_areas if improvement_areas else 'Minor refinements'}

TURN-BY-TURN PERFORMANCE (analyze progression):
{json.dumps(turn_history_summary, indent=2)}

CONVERSATION EXCERPTS:
{session.get('chat_history', [])[-6:]}

REQUIREMENTS:
1. Be WARM and CONVERSATIONAL - address {session['name']} directly
2. Reference SPECIFIC phrases they said during the session
3. Explain WHY scores changed between turns (e.g., "Your grammar jumped from 80% to 95% in Turn 3 when you used complex sentences")
4. Give ACTIONABLE, specific tips (not generic advice)
5. Celebrate their wins enthusiastically!

Return STRICTLY valid JSON:
{{
    "detailed_feedback": {{
        "grammar": {{"score": {final_scores['grammar']}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences explaining grammar performance with specific examples from their responses, mention what they did well and one specific thing to improve", "trend": "improved/declined/stable"}},
        "vocabulary": {{"score": {final_scores['vocabulary']}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about vocabulary richness, mention specific words they used well or could upgrade", "trend": "improved/declined/stable"}},
        "pronunciation": {{"score": {final_scores['pronunciation'] if final_scores['pronunciation'] is not None else '"N/A"'}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about clarity and specific words they pronounced well or struggled with", "trend": "improved/declined/stable"}},
        "fluency": {{"score": {final_scores['fluency'] if final_scores['fluency'] is not None else '"N/A"'}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about speaking pace ({average_wpm} WPM), hesitations, and natural flow", "trend": "improved/declined/stable"}}
    }},
    "turn_comparison": {{
        "best_turn": <number>,
        "worst_turn": <number>,
        "biggest_improvement": "Specific improvement like 'Grammar jumped from X% to Y% when you started using complex sentences'",
        "area_with_most_growth": "<area name>"
    }},
    "overall_analysis": "3-4 warm, personalized sentences comparing {session['name']}'s start vs end performance. Reference specific things they said. Example: 'Hey {session['name']}! You started a bit nervous in Turn 1, but by Turn 3 you were nailing those complex sentences! Your vocabulary really shined when you described...'",
    "suggestions": ["Very specific actionable suggestion with example exercise", "Another practical tip they can do today"],
    "tip": "One fun, memorable tip personalized to their weakest area - make it encouraging!"
}}"""

            try:
                llm_final_content = await call_llm(final_feedback_prompt, mode="strict_json", timeout=30, model=model)
                json_match = re.search(r'\{[\s\S]*\}', llm_final_content)
                if json_match:
                    final_data = json.loads(json_match.group())
                    detailed_feedback = final_data.get("detailed_feedback", {})
                    turn_comparison = final_data.get("turn_comparison", {})
                    analysis_text = final_data.get("overall_analysis", "")
                    final_message = final_data.get("final_message", "")
                    suggestions = final_data.get("suggestions", [])
                    tip = final_data.get("tip", "")
                else:
                    raise ValueError("No valid JSON")
            except:
                
                first_turn = turn_history_summary[0] if turn_history_summary else {}
                last_turn = turn_history_summary[-1] if turn_history_summary else {}
                
                
                grammar_change = last_turn.get("grammar", 0) - first_turn.get("grammar", 0)
                vocab_change = last_turn.get("vocabulary", 0) - first_turn.get("vocabulary", 0)
                
                detailed_feedback = {
                    "grammar": {
                        "score": final_scores["grammar"], 
                        "status": "Excellent" if final_scores["grammar"] >= 85 else "Good" if final_scores["grammar"] >= 70 else "Needs Work", 
                        "feedback": f"Great job, {session['name']}! Your grammar averaged {final_scores['grammar']}% across {session.get('turn_number', 0)} turns. " + 
                                   (f"You improved by {grammar_change}% from start to finish!" if grammar_change > 0 else "Keep practicing sentence structure for even better results."),
                        "trend": "improved" if grammar_change > 0 else "declined" if grammar_change < 0 else "stable"
                    },
                    "vocabulary": {
                        "score": final_scores["vocabulary"], 
                        "status": "Excellent" if final_scores["vocabulary"] >= 85 else "Good" if final_scores["vocabulary"] >= 70 else "Needs Work", 
                        "feedback": f"Your vocabulary scored {final_scores['vocabulary']}%! " +
                                   ("You used rich, varied words throughout - impressive!" if final_scores["vocabulary"] >= 85 else "Try incorporating more B2/C1 level words to sound more natural."),
                        "trend": "improved" if vocab_change > 0 else "declined" if vocab_change < 0 else "stable"
                    },
                }
                
                if final_scores["pronunciation"] is not None:
                    detailed_feedback["pronunciation"] = {
                        "score": final_scores["pronunciation"], 
                        "status": "Excellent" if final_scores["pronunciation"] >= 85 else "Good" if final_scores["pronunciation"] >= 70 else "Needs Work", 
                        "feedback": f"Pronunciation at {final_scores['pronunciation']}%! " +
                                   ("Your clarity was excellent - native speakers would understand you easily!" if final_scores["pronunciation"] >= 85 else 
                                    "Focus on clearly pronouncing consonant sounds and word endings for better clarity."),
                        "trend": "stable"
                    }
                if final_scores["fluency"] is not None:
                    speed_desc = "perfect pace" if 110 <= average_wpm <= 160 else "a bit slow - try to relax and speak faster" if average_wpm < 110 else "quite fast - try pausing between ideas"
                    detailed_feedback["fluency"] = {
                        "score": final_scores["fluency"], 
                        "status": "Excellent" if final_scores["fluency"] >= 85 else "Good" if final_scores["fluency"] >= 70 else "Needs Work", 
                        "feedback": f"You spoke at {average_wpm} words per minute - {speed_desc}! " +
                                   ("Great natural rhythm!" if final_scores["fluency"] >= 85 else "Practice speaking in longer phrases without hesitation."),
                        "trend": "stable"
                    }
                
                best_turn = max(range(len(turn_history_summary)), key=lambda i: turn_history_summary[i].get("overall", 0)) + 1 if turn_history_summary else 1
                worst_turn = min(range(len(turn_history_summary)), key=lambda i: turn_history_summary[i].get("overall", 0)) + 1 if turn_history_summary else 1
                
                turn_comparison = {
                    "best_turn": best_turn, 
                    "worst_turn": worst_turn, 
                    "biggest_improvement": f"Your best performance was in Turn {best_turn} with {turn_history_summary[best_turn-1].get('overall', 0) if turn_history_summary else 0}%!" if turn_history_summary else "N/A", 
                    "area_with_most_growth": strengths[0] if strengths else "overall"
                }
                
                analysis_text = f"Hey {session['name']}! 🎉 You completed {session.get('turn_number', 0)} practice turns with an overall score of {overall}%. " + \
                               (f"Your strongest areas were {', '.join(strengths)} - amazing work! " if strengths else "You're building a solid foundation. ") + \
                               (f"For next time, let's focus on improving your {', '.join(improvement_areas)} - you've got this!" if improvement_areas else "Keep up the great momentum!")
                
                final_message = f"Awesome session, {session['name']}! You scored {overall}% - be proud of your progress!"
                
                suggestions = [
                    f"Practice {improvement_areas[0]} by recording yourself and comparing with native speakers" if improvement_areas else "Challenge yourself with longer, more complex sentences",
                    f"Try shadowing exercises - listen to English content and repeat immediately" if average_wpm < 110 else "Read aloud for 10 minutes daily to maintain your great pace"
                ]
                
                tip = f"Quick tip for {session['name']}: " + (f"Focus on {improvement_areas[0]} by practicing tongue twisters!" if improvement_areas and improvement_areas[0] == "pronunciation" else 
                      f"Boost your {improvement_areas[0] if improvement_areas else 'speaking'} by having 5-minute English conversations with yourself daily - it really works!")
            
            
            final_feedback_data = {
                "final_scores": final_scores,
                "overall_score": overall,
                "average_wpm": average_wpm,
                "wpm_per_turn": wpm_per_turn,
                "vocab_overall": vocab_overall,
                "detailed_feedback": detailed_feedback,
                "turn_comparison": turn_comparison,
                "strengths": strengths,
                "improvement_areas": improvement_areas,
                "overall_analysis": analysis_text,
                "suggestions": suggestions,
                "tip": tip,
                "turn_history": turn_history_summary
            }
            await db.complete_session(session_id, final_feedback=final_feedback_data, overall_score=overall)
            
            # Build turn_feedback for termination response (same format as /fluent_feedback)
            turn_feedback = []
            # Aggregate grammar mistakes and vocabulary suggestions from all turns
            grammar_mistakes = []
            vocab_suggestions = []
            pronunciation_issues = []
            
            for i, attempt in enumerate(session.get("attempts", []), 1):
                turn_feedback.append({
                    "turn": i,
                    "transcription": attempt.get("transcription", ""),
                    "grammar": attempt.get("grammar", {}),
                    "vocabulary": attempt.get("vocabulary", {}),
                    "pronunciation": attempt.get("pronunciation"),
                    "fluency": attempt.get("fluency"),
                    "personalized_feedback": attempt.get("personalized_feedback", {}),
                    "improvement": attempt.get("improvement"),
                    "overall_score": attempt.get("overall_score", attempt.get("average_score", 0))
                })
                
                # Collect grammar errors (wrong → correct)
                gram = attempt.get("grammar") or {}
                if isinstance(gram, dict):
                    for err in gram.get("errors", []):
                        if isinstance(err, dict):
                            grammar_mistakes.append({
                                "wrong": err.get("you_said", err.get("wrong_word", "")),
                                "correct": err.get("should_be", err.get("correct_word", ""))
                            })
                
                # Collect vocabulary suggestions (weak word → better word)
                vocab = attempt.get("vocabulary") or {}
                if isinstance(vocab, dict):
                    for sug in vocab.get("suggestions", []):
                        if isinstance(sug, dict):
                            vocab_suggestions.append({
                                "weak_word": sug.get("word", ""),
                                "better_options": sug.get("better_word", "")
                            })
                
                # Collect pronunciation issues
                pron = attempt.get("pronunciation") or {}
                if isinstance(pron, dict):
                    for word_issue in pron.get("words_to_practice", []):
                        if isinstance(word_issue, dict):
                            pronunciation_issues.append({
                                "word": word_issue.get("word", ""),
                                "issue": word_issue.get("issue", ""),
                                "how_to_say": word_issue.get("how_to_say", "")
                            })
            
            # Build summary of all mistakes
            summary = {
                "grammar": {
                    "total_errors": len(grammar_mistakes),
                    "errors": grammar_mistakes
                },
                "vocabulary": {
                    "total_suggestions": len(vocab_suggestions),
                    "suggestions": vocab_suggestions
                },
                "pronunciation": {
                    "total_issues": len(pronunciation_issues),
                    "issues": pronunciation_issues
                }
            }
            
            return {
                "status": "conversation_ended", "session_id": session_id,
                "target_lang": target_language, "native_lang": native_language,
                "final_scores": final_scores, "overall_score": overall, "passing_score": PASSING_SCORE,
                "average_wpm": average_wpm, "wpm_per_turn": wpm_per_turn, "wpm_status": wpm_status,
                "vocab_overall": vocab_overall,
                "detailed_feedback": detailed_feedback, "turn_comparison": turn_comparison,
                "strengths": strengths, "improvement_areas": improvement_areas,
                "overall_analysis": analysis_text, "suggestions": suggestions,
                "total_turns": session.get("turn_number", 0), "message": final_message, "tip": tip,
                "turn_history": turn_history_summary,
                "turn_feedback": turn_feedback,
                "summary": summary  # NEW: Aggregated mistakes from all turns
            }

        
        session_level = session.get("level", "B1")
        session_user_type = session.get("user_type", user_type)
        
        
        if is_audio_input:
            grammar, vocabulary, pronunciation = await asyncio.gather(
                analyze_grammar_llm(user_text, level=session_level, user_type=session_user_type, model=model, target_lang=target_language),
                analyze_vocab_llm(user_text, user_type=session_user_type, level=session_level, model=model, target_lang=target_language),
                analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=session_level, user_type=session_user_type, model=model, target_language=session.get("target_language", "en"))
            )
        else:
            
            grammar, vocabulary = await asyncio.gather(
                analyze_grammar_llm(user_text, level=session_level, user_type=session_user_type, model=model, target_lang=target_language),
                analyze_vocab_llm(user_text, user_type=session_user_type, level=session_level, model=model, target_lang=target_language)
            )
            pronunciation = None
        
        
        if is_audio_input:
            if audio_analysis and audio_analysis.get("success"):
                fluency_data = audio_analysis.get("fluency", {})
                fluency = {
                    "score": fluency_data.get("score", 70), "wpm": fluency_data.get("wpm", 100),
                    "speed_status": "slow" if fluency_data.get("wpm", 100) < 100 else "normal" if fluency_data.get("wpm", 100) < 160 else "fast",
                    "pause_count": fluency_data.get("pause_count", 0),
                    "hesitation_count": len(grammar.get("filler_words", []))
                }
            else:
                word_count = len(user_text.split())
                audio_duration_seconds = 5
                if audio_path:
                    try:
                        audio_for_duration = AudioSegment.from_file(audio_path)
                        audio_duration_seconds = len(audio_for_duration) / 1000
                    except:
                        audio_duration_seconds = 5
                
                fluency = await analyze_fluency_metrics(user_text, audio_duration_seconds)
            
            fluency["original_text"] = user_text
            fluency["corrected_text"] = grammar.get("corrected_sentence", user_text)
            fluency["improved_sentence"] = grammar.get("improved_sentence", user_text)
            fluency["filler_words_removed"] = grammar.get("filler_words", [])
        else:
            fluency = None  
        
        
        if is_audio_input:
            scores = {
                "grammar": grammar.get("score", 70), "vocabulary": vocabulary.get("score", 70),
                "pronunciation": pronunciation.get("accuracy", 70) if pronunciation else 0, "fluency": fluency.get("score", 70) if fluency else 0
            }
            
            overall_score = int(scores["pronunciation"] * 0.30 + scores["grammar"] * 0.30 + scores["vocabulary"] * 0.20 + scores["fluency"] * 0.20)
        else:
            scores = {
                "grammar": grammar.get("score", 70), "vocabulary": vocabulary.get("score", 70),
                "pronunciation": None, "fluency": None
            }
            
            overall_score = int(scores["grammar"] * 0.50 + scores["vocabulary"] * 0.50)
        
        
        session["scores"]["grammar"] += scores["grammar"]
        session["scores"]["vocabulary"] += scores["vocabulary"]
        if is_audio_input:
            session["scores"]["pronunciation"] += scores["pronunciation"]
            session["scores"]["fluency"] += scores["fluency"]
            session["scores"]["total_wpm"] += fluency.get("wpm", 0) if fluency else 0
            session["scores"]["audio_count"] = session["scores"].get("audio_count", 0) + 1  
        session["scores"]["count"] += 1
        session["turn_number"] += 1


        personalized_feedback = await generate_personalized_feedback(
            user_type, overall_score, scores, user_text,
            grammar=grammar, vocabulary=vocabulary, pronunciation=pronunciation, model=model
        )
        
        native_language = session.get("native_language", "hi")
        target_language = session.get("target_language", "en")
        
        
        if is_audio_input and pronunciation:
            grammar_translated, vocab_translated, pron_translated, feedback_translated = await asyncio.gather(
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
            )
        else:
            grammar_translated, vocab_translated, feedback_translated = await asyncio.gather(
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
            )
            pron_translated = None

        
        current_attempt = {
            "pronunciation": pronunciation,
            "grammar": grammar,
            "vocabulary": vocabulary,
            "fluency": fluency,
            "overall_score": overall_score,
            "average_score": overall_score,
            "transcription": user_text,
            "turn_number": session["turn_number"]
        }
        session["attempts"].append(current_attempt)
        
        
        is_retrying = session.get("retry_count", 0) > 0 or session.get("is_retry_attempt", False)
        session["is_retry_attempt"] = False  
        improvement = await compare_attempts(session["attempts"], level=session_level, user_type=session_user_type) if is_retrying else {}
        
        prev_score = session.get("last_overall_score")
        if is_retrying and prev_score is not None:
            diff = overall_score - prev_score
            improvement["overall_previous_score"] = prev_score
            improvement["overall_current_score"] = overall_score
            improvement["overall_difference"] = diff
            improvement["overall_improved"] = diff > 0
            if diff > 0:
                improvement["overall_message"] = f"You improved from {prev_score}% to {overall_score}%! (+{diff}%)"
            elif diff < 0:
                improvement["overall_message"] = f"Score changed from {prev_score}% to {overall_score}% ({diff}%)"
            else:
                improvement["overall_message"] = f"Score unchanged at {overall_score}%"
        
        session["last_overall_score"] = overall_score
        should_retry = overall_score < PASSING_SCORE and not skip_retry
        
        if should_retry:
            session["retry_count"] = session.get("retry_count", 0) + 1
        else:
            session["retry_count"] = 0
        
        if should_retry:
            current_q = session.get("current_question", "")
            current_h = session.get("current_hint", "")
            
            
            retry_prompt_en = "I see there's room for improvement. Would you like to retry? Say 'yes' to try again or 'no' to skip."
            
            q_native, h_native, retry_prompt_native = await asyncio.gather(
                translate_text(current_q, target_language, native_language),
                translate_text(current_h, target_language, native_language),
                translate_text(retry_prompt_en, "en", native_language)
            )
            
            
            session["waiting_retry_decision"] = True
            await db.update_session(session_id, session)
            
            return {
                "status": "feedback", "session_id": session_id, 
                "target_lang": target_language, "native_lang": native_language,
                "transcription": user_text,
                "grammar": grammar_translated, "vocabulary": vocab_translated, 
                "pronunciation": pron_translated, "fluency": fluency,
                "personalized_feedback": feedback_translated, "overall_score": overall_score,
                "passing_score": PASSING_SCORE, "should_retry": True,
                "retry_prompt": {"target": retry_prompt_en, "native": retry_prompt_native},
                "retry_count": session.get("retry_count", 1), "improvement": improvement, "turn_number": session["turn_number"],
                "next_question": {"target": current_q, "native": q_native},
                "hint": {"target": current_h, "native": h_native}
            }

        else:
            
            session["last_overall_score"] = None
            session["is_retry_attempt"] = False
            follow_up, hint = await generate_follow_up_llm(user_text, target_language, session["chat_history"], model=model)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            
            follow_up_native, hint_native = await asyncio.gather(
                translate_text(follow_up, target_language, native_language),
                translate_text(hint, target_language, native_language)
            )
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            follow_up_audio = await generate_tts_url(request, follow_up, target_language, voice_id=voice_id)
            
            return {
                "status": "continue", "session_id": session_id, 
                "target_lang": target_language, "native_lang": native_language,
                "transcription": user_text,
                "grammar": grammar_translated, "vocabulary": vocab_translated,
                "pronunciation": pron_translated, "fluency": fluency,
                "personalized_feedback": feedback_translated, "overall_score": overall_score,
                "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"],
                "improvement": improvement,  
                "next_question": {"target": follow_up, "native": follow_up_native},
                "hint": {"target": hint, "native": hint_native},
                "audio_url": follow_up_audio
            }
    
    except Exception as e:
        logger.exception(f"Error in practice_fluent_lang: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        
        if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


