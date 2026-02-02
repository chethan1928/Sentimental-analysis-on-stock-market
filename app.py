@router.post("/practice_pronunciation")
async def practice_pronunciation(
    name: str = Form(default="User"),
    level: str = Form(default="B1"),
    mode: str = Form(default="normal"),
    native_language: str = Form(...),  
    target_lang: str = Form(default="en"),  
    topic: str = Form(default="daily life"),
    num_words: int = Form(default=5),
    set_number: int = Form(default=None),  
    audio_file: Optional[UploadFile] = File(default=None),
    session_id: Optional[str] = Form(default=None),
    action: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default="gpt"),
    voice_id: Optional[str] = Form(default=None),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):

    """
    pronunciation practice api - handles word and sentence practice
    
    modes:
    - normal: llm-generated lessons with full analysis
    - strict: vocab file based with 15 words, 3 sentences each
    
    flow:
    1. first call (no audio): creates session, returns first word
    2. with audio: analyzes pronunciation, returns feedback
    3. action="next": skip to next word/sentence
    4. action="end": end session early
    """
    
    try:
        
        if action == "end" and session_id:
            session = await db.get_user_session(session_id)
            if session:
                summary = await generate_session_summary(session, model=model)
                native_lang = session.get("native_language", "en")
                summary_bilingual = await make_bilingual(summary, "en", native_lang)
                msg_en = "Session ended. Great practice!"
                msg_native = await translate_text(msg_en, "en", native_lang)
                
                # Build response first, then save it
                response = {
                    "status": "complete",
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "is_session_complete": True,
                    "session_summary": summary_bilingual,
                    "message": {"target": msg_en, "native": msg_native}
                }
                
                await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                
                return response
            else:
                
                return {
                    "status": "error",
                    "session_id": session_id,
                    "error": "Session not found or already expired. Cannot end a non-existent session."
                }
        
        
        session = None
        if session_id:
            session = await db.get_user_session(session_id)
        
        
        if session and session.get("status") == "completed":
            return {"status": "error", "session_id": session_id, "error": "This session has ended. Please start a new session."}
        
        if not session:
            session_id = str(uuid.uuid4())
            
            
            if mode == "strict":
                lesson = await build_lesson_strict(target_lang, num_words, set_number=set_number, model=model)
            else:
                lesson = await generate_lesson_llm(topic, num_words, target_lang, model=model)
            
            if not lesson:
                return {
                    "status": "error",
                    "message": "failed to generate lesson. please try again."
                }
            
            
            session = {
                "user_name": name,
                "mode": mode,
                "level": level,
                "native_language": native_language,  
                "target_lang": target_lang,
                "topic": topic,
                "lesson": lesson,
                "current_word_index": 0,
                "current_phase": "word",
                "current_sentence_index": 0,
                "attempt_count": 0,
                "history": [],
                "scores": {"pronunciation": []},
                "total_words": len(lesson),
                "turn_history": []
            }
            await db.create_session(
                session_id=session_id,
                session_type="pronunciation",
                data=session,
                user_id=current_user.id if current_user else None,
                user_name=name
            )
            
            
            first_word = lesson[0]
            
            
            greeting_en = f"Hi {name}! I'm Sara. Let's practice pronunciation together. Relax and speak naturally."
            instruction_en = f"Listen carefully and repeat after me: {first_word['word']}"
            greeting_native = await translate_text(greeting_en, "en", native_language)
            instruction_native = await translate_text(instruction_en, "en", native_language)
            
            # Generate TTS audio URLs for greeting
            greeting_audio = ""
            if request:
                greeting_audio = await generate_tts_url(request, greeting_en, target_lang, voice_id=voice_id)
            
            meaning_native = first_word.get(f"meaning_{native_language}", "")
            if not meaning_native:
                meaning_native = await translate_text(first_word.get("meaning_en") or first_word["word"], "en", native_language)
            
            return {
                "status": "new_session",
                "session_id": session_id,
                "target_lang": target_lang,
                "native_lang": native_language,
                "mode": mode,
                "greeting": {"target": greeting_en, "native": greeting_native, "audio_url": greeting_audio},
                "current_word": {
                    "word": first_word["word"],
                    "meaning": {
                        "target": first_word.get("meaning_en", ""),
                        "native": meaning_native
                    },
                    "instruction": {"target": instruction_en, "native": instruction_native}
                },
                "phase": "word",
                "attempt_number": 1,
                "max_attempts": MAX_ATTEMPTS,
                "progress": {
                    "current_word_index": 1,
                    "total_words": len(lesson),
                    "completed_words": []
                }
            }
        
        
        lesson = session["lesson"]
        current_idx = session["current_word_index"]
        current_phase = session["current_phase"]
        current_word = lesson[current_idx]
        
        
        
        native_lang = session.get("native_language", "en")
        
        if action == "next":
            
            session_mode = session.get("mode", "normal")
            
            if current_phase == "word":
                
                session["current_phase"] = "sentence"
                session["current_sentence_index"] = 0
                session["attempt_count"] = 0
                
                instruction_en = "Now practice this sentence."
                instruction_native = await translate_text(instruction_en, "en", native_lang)
                
                
                if "sentences" in current_word and current_word["sentences"]:
                    sentence = current_word["sentences"][0]
                    sentence_en = safe_get_sentence_text(sentence, "en")
                    sentence_native = safe_get_sentence_text(sentence, native_lang) or await translate_text(sentence_en, "en", native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_phase",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": sentence_en, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": 1,
                        "total_sentences": len(current_word["sentences"]),
                        "instruction": {"target": instruction_en, "native": instruction_native},
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
                else:
                    
                    sentence_en = current_word.get("sentence", f"Practice saying {current_word['word']}.")
                    sentence_native = await translate_text(sentence_en, "en", native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_phase",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": sentence_en, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": 1,
                        "total_sentences": 1,
                        "instruction": {"target": instruction_en, "native": instruction_native},
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
            else:
                
                sentence_idx = session.get("current_sentence_index", 0)
                
                
                if "sentences" in current_word and sentence_idx + 1 < len(current_word["sentences"]):
                    
                    session["current_sentence_index"] = sentence_idx + 1
                    next_sentence = current_word["sentences"][session["current_sentence_index"]]
                    next_sentence_en = safe_get_sentence_text(next_sentence, "en")
                    sentence_native = safe_get_sentence_text(next_sentence, native_lang) or await translate_text(next_sentence_en, "en", native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_sentence",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": next_sentence_en, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": session["current_sentence_index"] + 1,
                        "total_sentences": len(current_word["sentences"]),
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
                
                
                session["current_word_index"] += 1
                session["current_phase"] = "word"
                session["current_sentence_index"] = 0
                session["attempt_count"] = 0
                
                if session["current_word_index"] >= len(lesson):
                    
                    summary = await generate_session_summary(session)
                    
                    summary_bilingual = await make_bilingual(summary, "en", native_lang)
                    msg_en = "Excellent work! You've completed all words."
                    msg_native = await translate_text(msg_en, "en", native_lang)
                    
                    # Build response first, then save it
                    response = {
                        "status": "complete",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "is_session_complete": True,
                        "session_summary": summary_bilingual,
                        "message": {"target": msg_en, "native": msg_native}
                    }
                    
                    await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                    
                    return response
                
                next_word = lesson[session["current_word_index"]]
                instruction_en = f"Next word: {next_word['word']}"
                instruction_native = await translate_text(instruction_en, "en", native_lang)
                
                
                meaning_native = next_word.get(f"meaning_{native_lang}", "") or await translate_text(next_word.get("meaning_en") or next_word["word"], "en", native_lang)
                
                
                await db.update_session(session_id, session)
                
                return {
                    "status": "next_word",
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "current_word": {
                        "word": next_word["word"],
                        "meaning": {
                            "target": next_word.get("meaning_en", ""),
                            "native": meaning_native
                        },
                        "instruction": {"target": instruction_en, "native": instruction_native}
                    },
                    "phase": "word",
                    "attempt_number": 1,
                    "max_attempts": MAX_ATTEMPTS,
                    "progress": {
                        "current_word_index": session["current_word_index"] + 1,
                        "total_words": len(lesson)
                    }
                }
        
        
        if not audio_file:
            msg_en = "please provide audio to continue"
            msg_native = await translate_text(msg_en, "en", native_lang)
            return {
                "status": "waiting_audio",
                "session_id": session_id,
                "target_lang": session.get("target_lang", "en"),
                "native_lang": native_lang,
                "message": {"target": msg_en, "native": msg_native},
                "current_word": {"word": current_word["word"]},
                "phase": current_phase
            }
        
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"audio_{session_id}.wav")
        
        try:
            
            content = await audio_file.read()
            original_filename = audio_file.filename or "audio.wav"
            original_ext = os.path.splitext(original_filename)[1].lower()
            temp_input_path = os.path.join(temp_dir, f"input_{session_id}{original_ext or '.wav'}")
            
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            
            if original_ext in ['.mp3', '.m4a', '.ogg', '.flac', '.aac', '.webm']:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(temp_input_path)
                    audio.export(audio_path, format="wav")
                except Exception as conv_err:
                    logger.warning(f"Audio conversion failed, using original: {conv_err}")
                    shutil.copy(temp_input_path, audio_path)
            else:
                
                shutil.copy(temp_input_path, audio_path)
            
            
            target_lang_for_audio = session.get("target_lang", "en")
            transcription = await transcribe_audio(audio_path, target_lang_for_audio)
            
            if not transcription:
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {
                    "status": "transcription_failed",
                    "session_id": session_id,
                    "message": "could not understand audio. please try again.",
                    "phase": current_phase
                }

            
            
            if current_phase == "word":
                expected = current_word["word"]
                
                
                word_analysis = await analyze_word_pronunciation(audio_path, expected, target_lang_for_audio)
                score = word_analysis["score"]
                transcription = word_analysis["transcription"] or transcription
                
                
                try:
                    from pydub import AudioSegment
                    audio_for_wpm = AudioSegment.from_file(audio_path)
                    audio_duration_seconds = len(audio_for_wpm) / 1000
                    word_count = len(transcription.split()) if transcription else 1
                    word_wpm = int((word_count / audio_duration_seconds) * 60) if audio_duration_seconds > 0 else 120
                except:
                    word_wpm = 120  
                
                # Calculate speed_status for word
                if word_wpm < 100:
                    word_speed_status = "slow"
                elif word_wpm <= 150:
                    word_speed_status = "normal"
                else:
                    word_speed_status = "fast"
                
                session["attempt_count"] += 1
                
                feedback = await generate_word_feedback(expected, transcription, score, session["attempt_count"], word_analysis, model=model)
                
                session["history"].append({
                    "phase": "word",
                    "word": {  
                        "target": current_word["word"],
                        "meaning_en": current_word.get("meaning_en", ""),
                        "meaning_native": current_word.get(f"meaning_{session.get('native_language', 'hi')}", current_word.get("meaning_native", ""))
                    },
                    "expected": expected,
                    "spoken": transcription,
                    "score": score,
                    "attempt": session["attempt_count"],
                    "confidence": word_analysis.get("confidence", 0),
                    "wpm": word_wpm,
                    "speed_status": word_speed_status,
                    "pronunciation_analysis": word_analysis,
                    # Store feedback text for /feedback endpoint
                    "feedback_message": feedback.get("message", ""),
                    "feedback_tip": feedback.get("tip", ""),
                    "feedback_status": feedback.get("status", "")
                })
                session["scores"]["pronunciation"].append(score)
                
                syllable_guide = None
                if word_analysis.get("needs_practice"):
                    
                    syllable_cache = session.get("syllable_cache", {})
                    if expected.lower() in syllable_cache:
                        syllable_guide = syllable_cache[expected.lower()]
                    else:
                        syllable_guide = await generate_syllable_guide(expected, model=model)
                        
                        if "syllable_cache" not in session:
                            session["syllable_cache"] = {}
                        session["syllable_cache"][expected.lower()] = syllable_guide
                
                
                current_attempt = session["attempt_count"]
                
                
                if feedback["next_action"] == "next_phase":
                    session["current_phase"] = "sentence"
                    session["current_sentence_index"] = 0
                    session["attempt_count"] = 0
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                
                await db.update_session(session_id, session)
                
                
                feedback_native = await translate_text(feedback["message"], "en", native_lang)
                
                # Generate TTS audio URL for feedback
                feedback_audio = ""
                if request:
                    feedback_audio = await generate_tts_url(request, feedback["message"], session.get("target_lang", "en"), voice_id=voice_id)
                
                response = {
                    "status": feedback["status"],
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "transcription": transcription,
                    "pronunciation_score": score,
                    "feedback": {"target": feedback["message"], "native": feedback_native, "audio_url": feedback_audio},
                    "current_word": {"word": current_word["word"]},
                    "phase": "word" if feedback["next_action"] == "retry" else "sentence",
                    "attempt_number": current_attempt,
                    "max_attempts": MAX_ATTEMPTS,
                    "next_action": feedback["next_action"],
                    "progress": {
                        "current_word_index": current_idx + 1,
                        "total_words": len(lesson)
                    },
                    
                    "analysis": {
                        "pronunciation": {
                            "score": score,
                            "confidence": word_analysis.get("confidence", 0),
                            "expected": expected,
                            "spoken": transcription,
                            "detected": word_analysis.get("detected", False),
                            "match_type": word_analysis.get("match_type", "unknown")
                        }
                    }
                }
                
                
                if syllable_guide:
                    response["syllable_guide"] = syllable_guide
                
                
                if feedback["next_action"] == "next_phase":
                    
                    if "sentences" in current_word and current_word["sentences"]:
                        sentence = current_word["sentences"][0]
                        sentence_en = safe_get_sentence_text(sentence, "en")
                        sentence_native = safe_get_sentence_text(sentence, native_lang) or await translate_text(sentence_en, "en", native_lang)
                        response["current_sentence"] = {
                            "text": {"target": sentence_en, "native": sentence_native}
                        }
                        response["sentence_number"] = 1
                        response["total_sentences"] = len(current_word["sentences"])
                    else:
                        
                        sentence_en = current_word.get("sentence", "")
                        sentence_native = current_word.get(f"sentence_{native_lang}", "") or await translate_text(sentence_en, "en", native_lang) if sentence_en else ""
                        response["current_sentence"] = {
                            "text": {"target": sentence_en, "native": sentence_native},
                            "example": current_word.get("example", "")
                        }
                        response["sentence_number"] = 1
                        response["total_sentences"] = 1
                
                return response
            
            
            else:
                
                sentence_idx = session.get("current_sentence_index", 0)
                if "sentences" in current_word and current_word["sentences"]:
                    sentences = current_word["sentences"]
                    current_sentence = sentences[sentence_idx]
                    expected = safe_get_sentence_text(current_sentence, "en")
                else:
                    
                    expected = current_word.get("sentence", "")
                
                
                sentence_analysis = await analyze_sentence_pronunciation(audio_path, expected, transcription, target_lang_for_audio)
                score = sentence_analysis["score"]
                
                analysis = {
                    "pronunciation": {
                        "score": score,
                        "expected": expected,
                        "spoken": transcription,
                        "mismatches": sentence_analysis.get("mismatches", []),
                        "mismatch_count": sentence_analysis.get("mismatch_count", 0),
                        "mispronounced_words": sentence_analysis.get("mispronounced_words", []),
                        "well_pronounced_words": sentence_analysis.get("well_pronounced_words", []),
                        "fluency": sentence_analysis.get("fluency", {}),
                        "accuracy_percentage": sentence_analysis.get("accuracy_percentage", 0)
                    }
                }
                
                # Get speed_status from fluency analysis
                sentence_speed_status = sentence_analysis.get("fluency", {}).get("speed_status", "normal")
                
                feedback = await generate_sentence_feedback(expected, transcription, score, analysis, model=model)
                
                session["history"].append({
                    "phase": "sentence",
                    "word": {  
                        "target": current_word["word"],
                        "meaning_en": current_word.get("meaning_en", ""),
                        "meaning_native": current_word.get(f"meaning_{session.get('native_language', 'hi')}", current_word.get("meaning_native", ""))
                    },
                    "expected": expected,
                    "spoken": transcription,
                    "score": score,
                    "mismatches": sentence_analysis.get("mismatches", []),
                    "wpm": sentence_analysis.get("fluency", {}).get("wpm", 120),
                    "speed_status": sentence_speed_status,
                    "pronunciation_analysis": sentence_analysis,
                    # Store feedback text for /feedback endpointttttt
                    "feedback_message": feedback.get("message", ""),
                    "feedback_tip": feedback.get("tip", ""),
                    "feedback_status": feedback.get("status", ""),
                    "focus_word": feedback.get("focus_word", ""),
                    "fluency_note": feedback.get("fluency_note", "")
                })
                session["scores"]["pronunciation"].append(score)
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                
                next_action = "next_word"
                is_complete = False
                
                
                if "sentences" in current_word and current_word["sentences"]:
                    sentence_idx = session.get("current_sentence_index", 0)
                    if sentence_idx + 1 < len(current_word["sentences"]):
                        session["current_sentence_index"] = sentence_idx + 1
                        next_action = "next_sentence"
                    else:
                        
                        session["current_word_index"] += 1
                        session["current_phase"] = "word"
                        session["current_sentence_index"] = 0
                        session["attempt_count"] = 0
                        
                        if session["current_word_index"] >= len(lesson):
                            is_complete = True
                            next_action = "complete"
                else:
                    
                    session["current_word_index"] += 1
                    session["current_phase"] = "word"
                    session["attempt_count"] = 0
                    
                    if session["current_word_index"] >= len(lesson):
                        is_complete = True
                        next_action = "complete"
                
                
                feedback_native = await translate_text(feedback["message"], "en", native_lang)
                
                # Generate TTS audio URL for feedback
                feedback_audio = ""
                if request:
                    feedback_audio = await generate_tts_url(request, feedback["message"], session.get("target_lang", "en"), voice_id=voice_id)
                
                response_status = "complete" if is_complete else feedback["status"]
                
                response = {
                    "status": response_status,
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "transcription": transcription,
                    "pronunciation_score": score,
                    "feedback": {"target": feedback["message"], "native": feedback_native, "audio_url": feedback_audio},
                    "analysis": analysis,
                    "current_word": {"word": current_word["word"]},
                    "phase": "sentence",
                    "next_action": next_action,
                    "is_session_complete": is_complete,
                    "progress": {
                        "current_word_index": session["current_word_index"] + 1 if not is_complete else len(lesson),
                        "total_words": len(lesson)
                    }
                }
                
                
                if is_complete:
                    summary = await generate_session_summary(session, model=model)
                    
                    summary_bilingual = await make_bilingual(summary, "en", native_lang)
                    response["session_summary"] = summary_bilingual
                    
                    # Save the complete termination response for get feedback endpoint
                    await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                    return response
                    
                elif next_action == "next_sentence" and "sentences" in current_word:
                    next_sentence = current_word["sentences"][session["current_sentence_index"]]
                    next_sentence_en = safe_get_sentence_text(next_sentence, "en")
                    sentence_native = safe_get_sentence_text(next_sentence, native_lang) or await translate_text(next_sentence_en, "en", native_lang)
                    response["current_sentence"] = {
                        "text": {"target": next_sentence_en, "native": sentence_native}
                    }
                    response["sentence_number"] = session["current_sentence_index"] + 1
                    response["total_sentences"] = len(current_word["sentences"])
                elif next_action == "next_word" and session["current_word_index"] < len(lesson):
                    next_word = lesson[session["current_word_index"]]
                    
                    meaning_native = next_word.get(f"meaning_{native_lang}", "") or await translate_text(next_word.get("meaning_en") or next_word["word"], "en", native_lang)
                    response["next_word"] = {
                        "word": next_word["word"],
                        "meaning": {
                            "target": next_word.get("meaning_en", ""),
                            "native": meaning_native
                        }
                    }
                
                
                await db.update_session(session_id, session)
                
                return response
        
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    except Exception as e:
        logger.error(f"pronunciation api error: {e}")
        return {
            "status": "error",
            "message": f"an error occurred: {str(e)}"
        }
in db.py 

async def complete_session(self, session_id: str, final_feedback: dict = None, overall_score: int = None, termination_response: dict = None) -> bool:
        """Mark session as completed and store final feedback, overall_score, and termination_response"""
        await self.init_db()
        async with async_session() as sess:

            result = await sess.execute(
                text("SELECT data FROM user_chat_sessions WHERE session_id = :sid"),
                {"sid": session_id}
            )
            row = result.fetchone()
            if row:
                data = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if isinstance(row[0], str) else {})
                data["status"] = "completed"
                data["completed_at"] = datetime.utcnow().isoformat()
                if final_feedback:
                    data["final_feedback"] = final_feedback
                if termination_response:
                    data["termination_response"] = termination_response


                if overall_score is None and final_feedback:
                    overall_score = final_feedback.get("overall_score")


                await sess.execute(
                    text("""UPDATE user_chat_sessions 
                            SET data = :data, overall_score = :score, updated_at = NOW() 
                            WHERE session_id = :sid"""),
                    {"sid": session_id, "data": json.dumps(data), "score": overall_score}
                )
                await sess.commit()
            return True
@router.get("/feedback/{session_id}")
async def get_pronunciation_feedback(session_id: str):
    """
    Get detailed per-turn feedback for a pronunciation session.
    
    Returns the same response that was returned when the session ended.
    Falls back to structured per-turn feedback if termination_response not available.
    """
    # First try to get the stored termination response
    session_data = await db.get_user_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return the stored termination response if available (same as session end response)
    if "termination_response" in session_data:
        return session_data["termination_response"]
    
    # Fall back to get_session_feedback for older sessions without termination_response
    feedback = await db.get_session_feedback(session_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Session not found")
    if feedback["session_type"] != "pronunciation":
        raise HTTPException(status_code=400, detail="Not a pronunciation session")
    return feedback
@router.get("/completed_sessions")
async def get_completed_pronunciation_sessions(current_user: User = Depends(get_current_user)):
    """
    Get only completed pronunciation sessions for the current user.
    Returns sessions where status='completed' and termination_response exists.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="pronunciation")
    
    completed_sessions = []
    for s in sessions:
        session_data = await db.get_user_session(s.get("session_id"))
        if not session_data:
            continue
        if session_data.get("status") != "completed":
            continue
        if not session_data.get("termination_response"):
            continue
        
        completed_sessions.append({
            "session_id": s.get("session_id"),
            "created_at": s.get("created_at"),
            "target_lang": session_data.get("target_lang", "en"),
            "native_lang": session_data.get("native_language", "en"),
            "mode": session_data.get("mode", "normal")
        })
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_completed": len(completed_sessions),
        "sessions": completed_sessions
    }
@router.get("/roles_with_session_ids")
async def get_roles_and_session_ids(current_user: User = Depends(get_current_user)):
    """
    Get all roles, their corresponding session IDs, and the total session counts for each role for the current user.
    """
    user_id = current_user.id if current_user else None

    # Get distinct roles
    roles = await db.get_distinct_roles_by_user(user_id, session_type="interview")

    # List to hold roles with session info
    roles_with_session_ids = []

    # Get all sessions for the user once
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")

    for role in roles:
        session_ids_for_role = []

        for s in sessions:
            session_data = await db.get_user_session(s.get("session_id"))
            if not session_data:
                continue
            if session_data.get("role") != role:
                continue
            if session_data.get("status") != "completed":
                continue
            if not session_data.get("final_feedback"):
                continue
            session_ids_for_role.append(s.get("session_id"))

        roles_with_session_ids.append({
            "role": role,
            "session_ids": session_ids_for_role,
            "total_sessions": len(session_ids_for_role)
        })

    return {
        "status": "success",
        "user_id": user_id,
        "total_roles": len(roles),
        "roles_with_session_ids": roles_with_session_ids
    }
