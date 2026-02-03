def analyze_speaking_advanced(audio_path: str, target_cefr: str = "B1", transcription: str = None, target_lang: str = "en") -> dict:
    """
    Advanced speaking analysis: fluency, pronunciation, 
    grammar errors, filler words, and CEFR vocabulary assessment.
    
    Args:
        audio_path: Path to the audio file for audio-based metrics (fluency, pronunciation)
        target_cefr: Target CEFR level for vocabulary assessment
        transcription: Optional - if provided, skips Whisper transcription and uses this text.
                       Use transcribe_audio_file() from fluent_api_v2 to get transcription.
        target_lang: Target language code (e.g., "en", "hi", "es") - will translate if detected language differs
    """
    result = {
        "success": False,
        "error": None,
        "transcription": "",
        "fluency": {},
        "pronunciation": {},
        "grammar_assessment": {
            "errors": [],
            "filler_words": [],
            "corrected_text": "",
            "filler_count": 0
        },
        "vocabulary_analysis": {
            "cefr_level": "",
            "vocabulary_score": 0,
            "suggestions": []
        },
        "overall_score": 0.0,
        "llm_feedback": ""
    }
    
    try:
        # Normalize target language code
        normalized_target = target_lang.lower() if target_lang else "en"
        lang_mapping = load_language_mapping()
        if normalized_target in lang_mapping:
            normalized_target = lang_mapping[normalized_target]
        
        # Use provided transcription or do Whisper transcription as fallback
        if transcription:
            # Use the provided transcription (from transcribe_audio_file)
            result["transcription"] = transcription
        else:
            # Fallback: do transcription here with forced target language
            model = _get_whisper_model()
            # Force Whisper to transcribe in the target language (auto-detection often detects wrong language)
            segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True, language=normalized_target)
            segments = list(segments)
            transcription = " ".join([s.text.strip() for s in segments]).strip()
            logger.debug(f"analyze_speaking_advanced - Whisper transcribed in {normalized_target}: {transcription[:100] if transcription else 'empty'}")
            
            result["transcription"] = transcription
        
        if not transcription:
            result["error"] = "No speech detected"
            return _to_python_type(result)

        
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(transcription.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        
        intervals = librosa.effects.split(y, top_db=30)
        pauses = []
        if len(intervals) > 1:
            for i in range(1, len(intervals)):
                p_dur = (intervals[i][0] - intervals[i-1][1]) / sr
                if p_dur > 0.3: pauses.append(p_dur)
        
        wpm_score = 100
        if wpm < IDEAL_WPM_MIN: wpm_score = max(0, 100 - (IDEAL_WPM_MIN - wpm) * 2)
        elif wpm > IDEAL_WPM_MAX: wpm_score = max(0, 100 - (wpm - IDEAL_WPM_MAX) * 2)
        
        pause_score = max(0, 100 - len(pauses) * 5 - sum(p for p in pauses if p > 2) * 10)
        fluency_score = (wpm_score * 0.5 + pause_score * 0.5)
        
        result["fluency"] = {
            "wpm": round(wpm, 1),
            "pause_count": len(pauses),
            "score": round(fluency_score, 1)
        }
        
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
        pitch_std = np.std(pitch_vals) if pitch_vals else 0
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        clarity_score = min(100, np.mean(spectral_centroids) / 30)
        intonation_score = min(100, max(0, 85 + (pitch_std - 20) * 0.2)) if 20 < pitch_std < 100 else (60 if pitch_std < 20 else 75)
        
        pron_score = (intonation_score * 0.4 + clarity_score * 0.6)
        result["pronunciation"] = {
            "clarity": round(clarity_score, 1),
            "intonation": round(intonation_score, 1),
            "score": round(pron_score, 1)
        }

        
        prompt = f"""Analyze the following spoken transcript for a language learner targeting CEFR level {target_cefr}.
        
        TRANSCRIPT: "{transcription}"
        
        INSTRUCTIONS:
        1. Identify GRAMMAR ERRORS and provide corrections.
        2. Identify FILLER WORDS (like "um", "ah", "like", "you know", "er").
        3. Provide a FULL CORRECTED VERSION of the text.
        4. Analyze VOCABULARY:
           - Is it appropriate for {target_cefr}?
           - Suggest 3 better/more advanced words.
        5. Provide an overall linguistic feedback.

        Respond ONLY in this JSON format:
        {{
            "grammar_assessment": {{
                "errors": [
                    {{"error": "incorrect phrase", "correction": "correct phrase", "rule": "why it was wrong"}}
                ],
                "filler_words": ["um", "like"],
                "corrected_text": "Complete corrected transcription here"
            }},
            "vocabulary_analysis": {{
                "detected_cefr": "B1",
                "vocabulary_score": 85,
                "suggestions": [
                    {{"original": "good", "advanced": "exceptional", "context": "describing weather"}}
                ]
            }},
            "linguistic_score": 80,
            "feedback": "Overall assessment of language usage."
        }}
        """
        
        llm_response = call_gpt(prompt, "You are an expert language examiner. Focus on grammar, filler words, and CEFR vocabulary.")
        
        import json
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start != -1:
                parsed = json.loads(llm_response[json_start:json_end])
                
                
                result["grammar_assessment"] = parsed.get("grammar_assessment", {})
                result["grammar_assessment"]["filler_count"] = len(result["grammar_assessment"].get("filler_words", []))
                
                result["vocabulary_analysis"] = parsed.get("vocabulary_analysis", {})
                result["llm_feedback"] = parsed.get("feedback", "")
                
                
                ling_score = parsed.get("linguistic_score", 0)
                result["overall_score"] = round((fluency_score * 0.3 + pron_score * 0.3 + ling_score * 0.4), 1)
        except Exception as e:
            result["llm_feedback"] = f"AI Analysis failed to parse: {str(e)}"
            result["overall_score"] = round((fluency_score + pron_score) / 2, 1)

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        
    return _to_python_type(result)async def transcribe_audio_file(audio_file, target_lang: str = "en") -> str:
    """Transcribe audio with auto-detect; translate to target if needed."""
    try:
        audio_file.file.seek(0)
    except:
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        temp_upload = tmp.name
    
    audio_path = None
    try:
        
        def convert_audio():
            audio = AudioSegment.from_file(temp_upload)
            audio = audio.set_frame_rate(16000).set_channels(1)
            converted_path = temp_upload.replace('.tmp', '_converted.wav')
            audio.export(converted_path, format="wav")
            return converted_path
        
        audio_path = await asyncio.to_thread(convert_audio)
        os.unlink(temp_upload)  
        
        # Normalize target language code
        languages_data = load_language_mapping()
        normalized_target = languages_data.get(target_lang.lower(), target_lang.lower()) if target_lang else "en"

        # Force Whisper to transcribe in the target language (auto-detect often detects wrong language)
        logger.debug(f"Transcribing audio with forced language: {normalized_target}")
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, task="transcribe", language=normalized_target
        )
        user_text = " ".join([seg.text for seg in segments]).strip()
        logger.debug(f"Whisper transcribed in {normalized_target}: {user_text[:100] if user_text else 'empty'}")

        return user_text
    except Exception as e:
        logger.debug(f"Audio transcription failed: {e}")
        return ""
    finally:
        
        for path in [temp_upload, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
async def analyze_pronunciation_llm(audio_path: str = None, spoken_text: str = None, level: str = "B1", user_type: str = "student", model: str = "gpt", target_language: str = "en") -> dict:
    """pronunciation analysis using Whisper word-level confidence to detect mispronounced words"""
    if not audio_path:
        return {
            "accuracy": 70, "transcription": spoken_text or "", 
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": spoken_text.split() if spoken_text else [],
            "feedback": "No audio provided for pronunciation analysis",
            "tips": ["Record audio for pronunciation feedback"], 
            "mispronounced_count": 0, "level": level, "user_type": user_type
        }
    
    try:
        target_lang = target_language.lower()
        languages_data = load_language_mapping()
        normalized_target = languages_data.get(target_lang, target_lang) if target_lang in languages_data else target_lang

        async def _transcribe_pronunciation(lang_hint: str = None):
            kwargs = {"word_timestamps": True}
            if lang_hint:
                kwargs["language"] = lang_hint
            segments, info = await asyncio.to_thread(_whisper_model.transcribe, audio_path, **kwargs)
            detected = info.language if info else (lang_hint or "en")
            words = []
            text = ""
            for seg in segments:
                text += seg.text + " "
                if seg.words:
                    for w in seg.words:
                        words.append({
                            "word": w.word.strip().lower(),
                            "confidence": w.probability,
                            "start": w.start,
                            "end": w.end
                        })
            return text.strip(), words, detected

        # Always force transcription in target language (auto-detect often gets wrong language)
        transcription, words_data, detected_lang = await _transcribe_pronunciation(normalized_target)
        logger.debug(f"Pronunciation - Whisper transcribed in {normalized_target}: {transcription[:100] if transcription else 'empty'}")

        # Keep original transcription for word scores alignment
        original_transcription = transcription
        translated_transcription = None

        # Only translate if detected language doesn't match target (keep original for word scores)
        if transcription and detected_lang != normalized_target:
            try:
                translated = await asyncio.to_thread(
                    GoogleTranslator(source=detected_lang, target=normalized_target).translate,
                    transcription
                )
                if translated:
                    logger.debug(f"Pronunciation: Translated from {detected_lang} to {normalized_target}")
                    translated_transcription = translated
            except Exception as e:
                logger.debug(f"Pronunciation translation failed: {e}")

        display_transcription = translated_transcription or original_transcription
        
        if not words_data:
            return {
                "accuracy": 0, "transcription": display_transcription, 
                "word_pronunciation_scores": [],
                "words_to_practice": [], "well_pronounced_words": [],
                "feedback": "No speech detected in audio",
                "tips": ["Speak clearly into the microphone"], 
                "mispronounced_count": 0, "level": level, "user_type": user_type,
                "original_transcription": original_transcription,
                "translated_transcription": translated_transcription,
                "detected_language": detected_lang,
                "target_language": normalized_target
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
        
        # Get language-specific pronunciation rules
        lang_rules = get_language_rules(target_language)
        lang_pron_text = ""
        if lang_rules:
            pron_rules = lang_rules.get("pronunciation", {})
            lang_pron_text = f"""
LANGUAGE-SPECIFIC PRONUNCIATION RULES FOR {lang_rules.get('name', target_language).upper()}:
- Stress Pattern: {pron_rules.get('stress_pattern', '')}
- Difficult Sounds: {pron_rules.get('difficult_sounds', [])}
- Special Features: {', '.join([f'{k}: {v}' for k, v in pron_rules.items() if k not in ['stress_pattern', 'difficult_sounds']])}

Use these rules when analyzing pronunciation for this language."""
        
        llm_prompt = f"""You are a pronunciation coach.

USER CONTEXT:
- Level: {level}
- User Type: {user_type}
- Target Language: {target_language}
{lang_pron_text}

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
}}"""

        try:
            llm_response = await call_llm(llm_prompt, mode="strict_json", timeout=30, model=model, target_language=target_language)
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                llm_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON")
        except Exception as llm_error:
            logger.error(f"LLM pronunciation error: {llm_error}")
            
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
            "transcription": display_transcription,
            "original_transcription": original_transcription,
            "translated_transcription": translated_transcription,
            "detected_language": detected_lang,
            "target_language": normalized_target,
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
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": [],
            "feedback": f"Could not analyze pronunciation: {str(e)}",
            "tips": ["Ensure clear audio recording"],
            "mispronounced_count": 0,
            "level": level, "user_type": user_type
        }
