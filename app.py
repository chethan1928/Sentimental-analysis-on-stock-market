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
        
        # Auto-detect language from audio (don't force to target_lang)
        segments, _ = await asyncio.to_thread(_whisper_model.transcribe, audio_path)
        user_text = " ".join([seg.text for seg in segments]).strip()
        logger.info(f"[transcribe_audio_file] Transcription result: '{user_text[:100] if user_text else 'empty'}'")
        
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
        target_lang = normalize_language_code(target_language, default="en")

        # Auto-detect language from audio (don't force to target_lang)
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, word_timestamps=True
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
}}"""

        
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

