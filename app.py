@router.post("/practice")
async def practice_fluent_lang(
    request: Request,
    name: str = Form(...),
    native_language: str = Form(...),
    target_language: str = Form(default="en"),
    level: Optional[str] = Form(default=None),  
    scenario: Optional[str] = Form(default=None),  
    user_type: str = Form(default="student"),
    audio_file: Optional[UploadFile] = File(default=None),
    text_input: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    skip_retry: bool = Form(default=False),
    model: Optional[str] = Form(default="gpt"),
    voice_id: Optional[str] = Form(default=None, description="Custom TTS voice ID (e.g., 'en-US-JennyNeural'). If not provided, auto-selects based on language."),
    current_user: User = Depends(get_current_user),
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
            
            # Step 1: Get transcription from transcribe_audio_file (uses large-v3, auto-detect)
            audio_file.file.seek(0)
            user_text = await transcribe_audio_file(audio_file, target_language)
            
            # Step 2: Get other metrics from analyze_speaking_advanced (pass transcription to skip re-transcribing)
            session_level_for_audio = session.get("level", "B1")
            audio_analysis = await asyncio.to_thread(analyze_speaking_advanced, audio_path, session_level_for_audio, user_text)

        
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

def analyze_speaking_advanced(audio_path: str, target_cefr: str = "B1", transcription: str = None) -> dict:
    """
    Advanced speaking analysis: fluency, pronunciation, 
    grammar errors, filler words, and CEFR vocabulary assessment.
    
    Args:
        audio_path: Path to the audio file for audio-based metrics (fluency, pronunciation)
        target_cefr: Target CEFR level for vocabulary assessment
        transcription: Optional - if provided, skips Whisper transcription and uses this text.
                       Use transcribe_audio_file() from fluent_api_v2 to get transcription.
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
        # Use provided transcription or do Whisper transcription as fallback
        if transcription:
            # Use the provided transcription (from transcribe_audio_file)
            result["transcription"] = transcription
        else:
            # Fallback: do transcription here
            model = _get_whisper_model()
            segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
            segments = list(segments)
            transcription = " ".join([s.text.strip() for s in segments]).strip()
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
        
    return _to_python_type(result)
