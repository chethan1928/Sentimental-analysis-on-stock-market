async def practice_interview(
    request: Request,
    name: str = Form(...),
    native_language: str = Form(default="hi"),
    target_language: str = Form(default="en"),
    level: str = Form(default="B1"),
    audio_file: Optional[UploadFile] = File(default=None),
    text_input: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    action: Optional[str] = Form(default=None),  
    model: Optional[str] = Form(default="gpt"),  
    voice_id: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),

):
    """
    interview practice api - CONVERSATIONAL ONBOARDING
    
    flow:
    1. first call (no audio/text): Clara greets and asks for role
    2. user provides role: Clara asks for interview type
    3. user provides type: interview begins with first question
    4. subsequent calls: normal interview with analysis
    5. action="end" or termination phrase: ends session
    """
    try:
        user_text = ""
        audio_path = None

        if not session_id or session_id.strip() == "" or session_id == "string":
            session_id = str(uuid.uuid4())
        
        
        session = await db.get_user_session(session_id)
        session_exists = session is not None
        native_language = session.get("native_language", native_language) if session else native_language
        target_language = session.get("target_language", target_language) if session else target_language
        
        
        if session_exists and session.get("status") == "completed":
            return {"status": "error", "session_id": session_id, "error": "This session has ended. Please start a new session."} 
        
        if not session_exists:
            
            session = {
                "state": "welcome",  
                "name": name, 
                "scenario": None,  
                "role": None,      
                "level": level,
                "native_language": native_language, 
                "target_language": target_language,
                "chat_history": [],
                "scores": {"grammar": 0, "vocabulary": 0, "pronunciation": 0, "fluency": 0, "total_wpm": 0, "count": 0, "audio_count": 0},
                "current_question": None, "current_hint": None, "turn_number": 0,
                "last_overall_score": None, "retry_count": 0, "attempts": [],
                "turn_history": [],  
                "onboarding_retry": 0  
            }
            await db.create_session(
                session_id=session_id,
                session_type="interview",
                data=session,
                user_id=current_user.id if current_user else None,
                user_name=name
            )
        
        
        current_state = session.get("state", "interviewing") 
        
        
        if current_state == "welcome" and not audio_file and not text_input:
            greeting = f"Hi {name}! I'm {BOT_NAME}, your interview coach 🙂 So, which role are you ready for?"
            
            greeting_native = await translate_text(greeting, "en", native_language)
            
            session["state"] = "collecting_role"
            session["chat_history"].append({"role": "assistant", "content": greeting})
            await db.update_session(session_id, session)
            
            greeting_audio = await generate_tts_url(request, greeting, target_language, api_type="interview", voice_id=voice_id)
            
            return {
                "status": "onboarding",
                "step": "collecting_role",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "transcription": user_text,
                "message": {"target": greeting, "native": greeting_native},
                "audio_url": greeting_audio
            }
        
        
        if current_state == "collecting_role":
            user_text = text_input or ""
            if audio_file:
                
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                return {"status": "error", "session_id": session_id, "error": "No speech detected. Please tell me which role you're preparing for."}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_role_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("role"):
                role = extraction["role"]
                session["role"] = role
                session["state"] = "collecting_type"
                session["onboarding_retry"] = 0
                
                
                ask_type = f"Great, {role}! Is this more of an HR interview, or would you prefer something else like behavioral or technical?"
                ask_type_native = await translate_text(ask_type, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": ask_type})
                await db.update_session(session_id, session)
                
                ask_type_audio = await generate_tts_url(request, ask_type, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_type", 
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "role": role,
                    "message": {"target": ask_type, "native": ask_type_native},
                    "audio_url": ask_type_audio
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "Could you be more specific about the role? For example: Software Engineer, Marketing Manager, Business Analyst, etc."
                retry_native = await translate_text(retry_msg, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                retry_audio = await generate_tts_url(request, retry_msg, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_role",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "retry": True,
                    "message": {"target": retry_msg, "native": retry_native},
                    "audio_url": retry_audio
                }
        
        if current_state == "collecting_type":
            user_text = text_input or ""
            if audio_file:
                
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                return {"status": "error", "session_id": session_id, "error": "No speech detected. Please tell me the interview type."}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_interview_type_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("type"):
                interview_type = extraction["type"]
                session["scenario"] = interview_type
                session["state"] = "interviewing"
                session["onboarding_retry"] = 0
                
                
                role = session.get("role", "Professional")
                scenario_name = INTERVIEW_SCENARIOS.get(interview_type, interview_type.title() + " Interview")
                
                question, hint = await generate_interview_question(interview_type, role, level, name, model=model)
                
                start_msg = f"Perfect! Let's start your {scenario_name} practice for {role}."
                start_native, q_native, h_native = await asyncio.gather(
                    translate_text(start_msg, "en", native_language),
                    translate_text(question, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                session["current_question"] = question
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": question})
                await db.update_session(session_id, session)
                
                question_audio = await generate_tts_url(request, question, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "interview_started",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "role": role,
                    "scenario": interview_type,
                    "greeting": {"target": start_msg, "native": start_native},
                    "next_question": {"target": question, "native": q_native},
                    "hint": {"target": hint, "native": h_native},
                    "turn_number": 0,
                    "audio_url": question_audio
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "What type of interview would you like to practice? For example: HR, Technical, Sales, Marketing, Customer Service, or any other type?"
                retry_native = await translate_text(retry_msg, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                retry_audio = await generate_tts_url(request, retry_msg, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_type",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "retry": True,
                    "message": {"target": retry_msg, "native": retry_native},
                    "audio_url": retry_audio
                }
        
        
        role = session.get("role", "Professional")
        scenario = session.get("scenario", "general")
        
        if action == "next":
            follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, model=model)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            session["retry_count"] = 0
            session["waiting_retry_decision"] = False  
            session["retry_clarify_count"] = 0  
            
            
            await db.update_session(session_id, session)
            
            follow_up_audio = await generate_tts_url(request, follow_up, target_language, api_type="interview", voice_id=voice_id)
            
            return {
                "status": "continue", "session_id": session_id,
                "target_lang": target_language, "native_lang": native_language,
                "transcription": "(skipped)",
                "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                "grammar": {"score": 0, "is_correct": True, "errors": [], "feedback": "Skipped"},
                "vocabulary": {"score": 0, "overall_level": "skipped", "feedback": "Skipped"},
                "pronunciation": {"accuracy": 0, "word_pronunciation_scores": [], "feedback": "Skipped"},
                "fluency": {"score": 0, "wpm": 0, "speed_status": "skipped"},
                "answer_evaluation": {"clarity": "", "structure": "", "relevance": "", "improved_answer": ""},
                "personalized_feedback": {"message": "Skipped. Let's try this question!", "improvement_areas": [], "strengths": []},
                "overall_score": 0, "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"],
                "audio_url": follow_up_audio
            }
        
        if action == "end":
            return await handle_session_termination(session, session_id, model)
        
        if not audio_file and not text_input:
            
            current_q = session.get("current_question")
            current_h = session.get("current_hint", "")
            
            if current_q:
                
                q_native = await translate_text(current_q, target_language, native_language)
                h_native = await translate_text(current_h, target_language, native_language)
                
                current_q_audio = await generate_tts_url(request, current_q, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "turn_number": session.get("turn_number", 0),
                    "audio_url": current_q_audio
                }
            else:
                
                question, hint = await generate_interview_question(
                    scenario, role, session.get("level", level), name, model=model
                )
                session["current_question"] = question
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": question})
                await db.update_session(session_id, session)
                
                q_native = await translate_text(question, target_language, native_language)
                h_native = await translate_text(hint, target_language, native_language)
                
                question_audio = await generate_tts_url(request, question, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": question, "native": q_native},
                    "hint": {"target": hint, "native": h_native},
                    "turn_number": session.get("turn_number", 0),
                    "audio_url": question_audio
                }
        
        user_text = text_input or ""
        audio_path = None
        audio_duration = 5.0
        is_audio_input = audio_file is not None  
        
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                shutil.copyfileobj(audio_file.file, tmp)
                temp_upload = tmp.name
            
            try:
                
                def convert_audio():
                    audio = AudioSegment.from_file(temp_upload)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    converted_path = temp_upload.replace('.tmp', '_converted.wav')
                    audio.export(converted_path, format="wav")
                    return converted_path, len(audio) / 1000
                
                audio_path, audio_duration = await asyncio.to_thread(convert_audio)
                os.unlink(temp_upload)  
            except Exception as e:
                logger.debug(f"Audio conversion fallback: {e}")
                audio_path = temp_upload
            finally:
                
                
                if audio_path != temp_upload and os.path.exists(temp_upload):
                    try:
                        os.unlink(temp_upload)
                    except:
                        pass
        
        if is_audio_input:
            pronunciation = await analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=session.get("level", level), model=model, target_language=target_language)
            
            if pronunciation and pronunciation.get("transcription"):
                user_text = pronunciation["transcription"]
        else:
            pronunciation = None   
        
        if audio_path:
            try:
                os.unlink(audio_path)
            except Exception:
                pass
        
        if not user_text or not user_text.strip():
            return {"status": "error", "session_id": session_id, "error": "No speech detected. Please try again."}
        
        user_text = user_text.strip()
        session["chat_history"].append({"role": "user", "content": user_text})
        
        if session.get("waiting_retry_decision"):
            user_choice = user_text.lower().strip()
            
            
            cleaned_choice = user_choice.rstrip('.,!?')
            if cleaned_choice in TERMINATION_PHRASES:
                
                session["waiting_retry_decision"] = False
                return await handle_session_termination(session, session_id, model)
            
            retry_keywords = ["yes", "retry", "practice", "again", "try", "redo", "repeat", "once more", "one more"]
            skip_keywords = ["no", "skip", "next", "move", "forward", "pass", "don't want", "not now", "let's move", "move on", "go ahead"]
            
            wants_retry = any(keyword in user_choice for keyword in retry_keywords)
            wants_skip = any(keyword in user_choice for keyword in skip_keywords)
            
            if wants_retry:
                
                session["waiting_retry_decision"] = False  
                session["retry_clarify_count"] = 0  
                current_q = session.get("current_question", "")
                current_h = session.get("current_hint", "")
                session["chat_history"].append({"role": "assistant", "content": current_q})
                await db.update_session(session_id, session)
                
                retry_msg = "Let's try this again! Take your time."
                q_native, h_native, retry_msg_native = await asyncio.gather(
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_text(retry_msg, "en", native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "message": {"target": retry_msg, "native": retry_msg_native},
                    "turn_number": session.get("turn_number", 0)
                }
            elif wants_skip:
                
                session["waiting_retry_decision"] = False  
                session["retry_clarify_count"] = 0  
                follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, model=model)
                session["current_question"] = follow_up
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": follow_up})
                session["retry_count"] = 0
                
                await db.update_session(session_id, session)
                
                follow_up_native, hint_native = await asyncio.gather(
                    translate_text(follow_up, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
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
                    
                    auto_skip_msg = "I see you're having trouble deciding. Let's move on to the next question!"
                    follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, model=model)
                    session["current_question"] = follow_up
                    session["current_hint"] = hint
                    session["chat_history"].append({"role": "assistant", "content": auto_skip_msg})
                    session["chat_history"].append({"role": "assistant", "content": follow_up})
                    session["retry_count"] = 0
                    
                    await db.update_session(session_id, session)
                    
                    auto_skip_native, follow_up_native, hint_native = await asyncio.gather(
                        translate_text(auto_skip_msg, "en", native_language),
                        translate_text(follow_up, target_language, native_language),
                        translate_text(hint, target_language, native_language)
                    )
                    
                    return {
                        "status": "auto_skipped",
                        "session_id": session_id,
                        "target_lang": target_language,
                        "native_lang": native_language,
                        "message": {"target": auto_skip_msg, "native": auto_skip_native},
                        "next_question": {"target": follow_up, "native": follow_up_native},
                        "hint": {"target": hint, "native": hint_native},
                        "turn_number": session["turn_number"],
                        
                        "grammar": None,
                        "vocabulary": None,
                        "pronunciation": None,
                        "fluency": None,
                        "answer_evaluation": None,
                        "emotion": None,
                        "personalized_feedback": None,
                        "overall_score": None,
                        "improvement": None
                    }
                else:
                    
                    current_q = session.get("current_question", "")
                    current_h = session.get("current_hint", "")
                    level = session.get("level", "B1")
                    
                    
                    if is_audio_input:
                        word_count = len(user_text.split())
                        estimated_duration = max(1, word_count / 2.5)  
                        grammar, vocabulary, answer_eval, pronunciation, fluency = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, model=model),
                            analyze_vocab_llm(user_text, level=level, model=model),
                            evaluate_answer(current_q, user_text, level, model=model),
                            analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=level, model=model, target_language=target_language),
                            analyze_fluency_metrics(user_text, estimated_duration)
                        )
                    else:
                        
                        grammar, vocabulary, answer_eval = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, model=model),
                            analyze_vocab_llm(user_text, level=level, model=model),
                            evaluate_answer(current_q, user_text, level, model=model)
                        )
                        pronunciation = None
                        fluency = None
                    
                    
                    if is_audio_input:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": pronunciation.get("score", pronunciation.get("accuracy", 70)) if pronunciation else 0,
                            "fluency": fluency.get("score", 70) if fluency else 0,
                            "answer_evaluation": answer_eval.get("score", 50)
                        }
                        
                        overall_score = int(
                            scores["grammar"] * 0.25 +
                            scores["vocabulary"] * 0.25 +
                            scores["answer_evaluation"] * 0.25 +
                            scores["pronunciation"] * 0.15 +
                            scores["fluency"] * 0.10
                        )
                    else:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": None,
                            "fluency": None,
                            "answer_evaluation": answer_eval.get("score", 50)
                        }
                        
                        overall_score = int(
                            scores["grammar"] * 0.33 +
                            scores["vocabulary"] * 0.33 +
                            scores["answer_evaluation"] * 0.34
                        )

                    
                    
                    emotion = {"emotion": "neutral", "confidence_level": "medium", "explanation": ""}
                    personalized_feedback = await generate_personalized_feedback(
                        overall_score, scores, emotion, session.get("name", "User"),
                        grammar=grammar, vocabulary=vocabulary, 
                        pronunciation=pronunciation, answer_eval=answer_eval, model=model
                    )
                    
                    
                    if clarify_count == 1:
                        clarify_msg = "I heard you say something, but I'm not sure if you want to practice again or move on. Just say 'retry' or 'skip' - or you can try answering the question again!"
                    else:
                        clarify_msg = "Still not quite sure what you'd like to do. Say 'yes' to practice the same question, or 'skip' to get a new one. One more unclear response and I'll move you to the next question."
                    
                    
                    await db.update_session(session_id, session)
                    
                    
                    if is_audio_input and pronunciation and fluency:
                        (clarify_native, q_native, h_native, grammar_t, vocab_t, 
                         pron_t, fluency_t, eval_t, personal_t) = await asyncio.gather(
                            translate_text(clarify_msg, "en", native_language),
                            translate_text(current_q, target_language, native_language),
                            translate_text(current_h, target_language, native_language),
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                            translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS),
                            translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                        )
                    else:
                        (clarify_native, q_native, h_native, grammar_t, vocab_t, 
                         eval_t, personal_t) = await asyncio.gather(
                            translate_text(clarify_msg, "en", native_language),
                            translate_text(current_q, target_language, native_language),
                            translate_text(current_h, target_language, native_language),
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                        )
                        pron_t = None
                        fluency_t = None
                    
                    return {
                        "status": "clarify_retry",
                        "session_id": session_id,
                        "target_lang": target_language,
                        "native_lang": native_language,
                        "transcription": user_text,
                        "message": {"target": clarify_msg, "native": clarify_native},
                        "next_question": {"target": current_q, "native": q_native},
                        "hint": {"target": current_h, "native": h_native},
                        "grammar": grammar_t,
                        "vocabulary": vocab_t,
                        "pronunciation": pron_t,
                        "fluency": fluency_t,
                        "answer_evaluation": eval_t,
                        "emotion": emotion,
                        "personalized_feedback": personal_t,
                        "overall_score": overall_score,
                        "clarify_count": clarify_count,
                        "turn_number": session.get("turn_number", 0)
                    }

        
        
        cleaned_text = user_text.lower().strip().rstrip('.,!?')
        is_termination = cleaned_text in TERMINATION_PHRASES or action == "end"
        
        
        
        
        grammar, vocabulary, answer_eval = await asyncio.gather(
            analyze_grammar_llm(user_text, level=level, model=model),
            analyze_vocab_llm(user_text, level=level, model=model),
            evaluate_answer(session.get("current_question", ""), user_text, level, model=model)
        )
        
        
        emotion = {"emotion": "neutral", "confidence_level": "medium", "explanation": ""}
        
        
        if is_audio_input:
            word_count = len(user_text.split())
            fluency = calculate_fluency(word_count, audio_duration)
        else:
            fluency = None  

        
        
        
        
        if not is_termination and session.get("current_question"):
            relevance_check = await check_answer_relevance(session["current_question"], user_text, model=model)
            
            if not relevance_check.get("relevant", True):
                
                redirect_msg = relevance_check.get("redirect", "Let's stay on track! 😄")
                current_q = session["current_question"]
                current_h = session.get("current_hint", "")
                
                
                full_response = f"{redirect_msg}\n\n{current_q}"
                session["chat_history"].append({"role": "assistant", "content": full_response})
                await db.update_session(session_id, session)
                
                redirect_native, q_native, h_native, grammar_t, vocab_t, eval_t = await asyncio.gather(
                    translate_text(redirect_msg, "en", native_language),
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                    translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                    translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
                )
                
                
                if is_audio_input and pronunciation and fluency:
                    pron_t, fluency_t = await asyncio.gather(
                        translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                        translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS)
                    )
                    scores = {
                        "grammar": grammar.get("score", 75),
                        "vocabulary": vocabulary.get("score", 75),
                        "pronunciation": pronunciation.get("accuracy", 75),
                        "fluency": fluency.get("score", 75)
                    }
                    answer_score = answer_eval.get("score", 50)
                    overall_score = int(
                        scores["grammar"] * 0.25 +
                        scores["vocabulary"] * 0.25 +
                        answer_score * 0.25 +
                        scores["pronunciation"] * 0.15 +
                        scores["fluency"] * 0.10
                    )
                else:
                    pron_t = None
                    fluency_t = None
                    scores = {
                        "grammar": grammar.get("score", 75),
                        "vocabulary": vocabulary.get("score", 75),
                        "pronunciation": None,
                        "fluency": None
                    }
                    answer_score = answer_eval.get("score", 50)
                    
                    overall_score = int(
                        scores["grammar"] * 0.33 +
                        scores["vocabulary"] * 0.33 +
                        answer_score * 0.34
                    )

                personalized_feedback = await generate_personalized_feedback(overall_score, scores, emotion, session["name"], model=model)
                
                
                personal_t = await translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                
                return {
                    "status": "redirect",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "message": {"target": redirect_msg, "native": redirect_native},
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    
                    "grammar": grammar_t,
                    "vocabulary": vocab_t,
                    "pronunciation": pron_t,
                    "fluency": fluency_t,
                    "answer_evaluation": eval_t,
                    "emotion": emotion,
                    "personalized_feedback": personal_t,
                    "overall_score": overall_score,
                    "passing_score": PASSING_SCORE,
                    "improvement": None,  
                    "turn_number": session.get("turn_number", 0)
                }
        
        
        if is_termination:
            return await handle_session_termination(session, session_id, model)

        
        
        if is_audio_input:
            scores = {
                "grammar": grammar.get("score", 75),
                "vocabulary": vocabulary.get("score", 75),
                "pronunciation": pronunciation.get("accuracy", 75) if pronunciation else 0,
                "fluency": fluency.get("score", 75) if fluency else 0
            }
        else:
            scores = {
                "grammar": grammar.get("score", 75),
                "vocabulary": vocabulary.get("score", 75),
                "pronunciation": None,  
                "fluency": None  
            }
        
        
        answer_score = answer_eval.get("score", 50)
        if is_audio_input:
            
            overall_score = int(
                scores["grammar"] * 0.25 +
                scores["vocabulary"] * 0.25 +
                answer_score * 0.25 +
                scores["pronunciation"] * 0.15 +
                scores["fluency"] * 0.10
            )
        else:
            
            overall_score = int(
                scores["grammar"] * 0.33 +
                scores["vocabulary"] * 0.33 +
                answer_score * 0.34
            )
        
        
        session["scores"]["grammar"] += scores["grammar"]
        session["scores"]["vocabulary"] += scores["vocabulary"]
        if is_audio_input:
            session["scores"]["pronunciation"] += scores["pronunciation"]
            session["scores"]["fluency"] += scores["fluency"]
            session["scores"]["total_wpm"] += fluency.get("wpm", 100) if fluency else 100
            session["scores"]["audio_count"] = session["scores"].get("audio_count", 0) + 1  
        session["scores"]["answer"] = session["scores"].get("answer", 0) + answer_score  
        session["scores"]["count"] += 1

        session["turn_number"] += 1
        

        
        
        personalized_feedback, (follow_up_question, follow_up_hint) = await asyncio.gather(
            generate_personalized_feedback(
                overall_score, scores, emotion, session["name"],
                grammar=grammar, vocabulary=vocabulary, 
                pronunciation=pronunciation, answer_eval=answer_eval, model=model
            ),
            generate_interactive_follow_up(user_text, session["chat_history"], role, scenario, model=model)
        )
        
        
        improvement = {}
        is_retrying = session.get("retry_count", 0) > 0
        prev_overall = session.get("last_overall_score")
        
        
        if is_retrying and prev_overall is not None:
            
            current_attempt = {
                "transcription": user_text,
                "grammar": grammar,
                "vocabulary": vocabulary,
                "pronunciation": pronunciation,
                "fluency": fluency,
                "answer_evaluation": answer_eval,
                "overall_score": overall_score
            }
            session.setdefault("attempts", []).append(current_attempt)
            
            
            improvement = await compare_attempts(
                session["attempts"], 
                level="B1",  
                user_type="professional", 
                model=model
            )
        else:
            
            current_attempt = {
                "transcription": user_text,
                "grammar": grammar,
                "vocabulary": vocabulary,
                "pronunciation": pronunciation,
                "fluency": fluency,
                "answer_evaluation": answer_eval,
                "overall_score": overall_score
            }
            session.setdefault("attempts", []).append(current_attempt)
        
        
        
        session["last_scores"] = scores.copy()
        session["last_overall_score"] = overall_score
        
        
        if "turn_history" not in session:
            session["turn_history"] = []
        
        turn_data = {
            "turn_number": session["turn_number"],
            "turn": session["turn_number"],  
            "transcription": user_text,
            "question": session.get("current_question", ""),
            "scores": scores.copy(),
            "overall_score": overall_score,
            "wpm": fluency.get("wpm", 0) if fluency else 0,  
            "grammar": grammar,
            "vocabulary": vocabulary,
            "pronunciation": pronunciation,
            "fluency": fluency,
            "answer_evaluation": answer_eval,
            "emotion": emotion,
            "personalized_feedback": personalized_feedback,
            "improvement": improvement
        }
        session["turn_history"].append(turn_data)
        
        
        should_retry = (overall_score < PASSING_SCORE or action == "practice")
        
        if should_retry:
            session["retry_count"] = session.get("retry_count", 0) + 1
            session["waiting_retry_decision"] = True  
            current_q = session.get("current_question", "")
            current_h = session.get("current_hint", "")
            
            
            retry_ask = "I see your answer, but it could be stronger. Would you like to practice this question again?"
            
            
            base_tasks = [
                translate_text(retry_ask, "en", native_language),
                translate_text(current_q, target_language, native_language),
                translate_text(current_h, target_language, native_language),
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
            ]
            base_results = await asyncio.gather(*base_tasks)
            retry_ask_native, q_native, h_native, grammar_t, vocab_t, eval_t = base_results
            
            
            pron_t = await translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS) if pronunciation else None
            fluency_t = await translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS) if fluency else None
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            retry_ask_audio = await generate_tts_url(request, retry_ask, target_language, api_type="interview")
            
            return {
                "status": "feedback",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "transcription": user_text,
                "message": {"target": retry_ask, "native": retry_ask_native},
                "next_question": {"target": current_q, "native": q_native},
                "hint": {"target": current_h, "native": h_native},
                "grammar": grammar_t,
                "vocabulary": vocab_t,
                "pronunciation": pron_t,
                "fluency": fluency_t,
                "answer_evaluation": eval_t,
                "emotion": emotion,
                "personalized_feedback": personalized_feedback,
                "overall_score": overall_score,
                "passing_score": PASSING_SCORE,
                "should_retry": True,
                "retry_count": session.get("retry_count", 1),
                "improvement": improvement,
                "turn_number": session["turn_number"],
                "audio_url": retry_ask_audio
            }
        else:
            
            session["current_question"] = follow_up_question
            session["current_hint"] = follow_up_hint
            session["chat_history"].append({"role": "assistant", "content": follow_up_question})
            session["retry_count"] = 0
            
            
            base_tasks = [
                translate_text(follow_up_question, target_language, native_language),
                translate_text(follow_up_hint, target_language, native_language),
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
            ]
            base_results = await asyncio.gather(*base_tasks)
            follow_up_native, hint_native, grammar_t, vocab_t, personal_t, eval_t = base_results
            
            
            pron_t = await translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS) if pronunciation else None
            fluency_t = await translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS) if fluency else None
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            follow_up_audio = await generate_tts_url(request, follow_up_question, target_language, api_type="interview")
            
            return {
                "status": "continue", "session_id": session_id, 
                "target_lang": target_language, "native_lang": native_language,
                "transcription": user_text,
                "next_question": {"target": follow_up_question, "native": follow_up_native},
                "hint": {"target": follow_up_hint, "native": hint_native},
                "grammar": grammar_t, "vocabulary": vocab_t, "pronunciation": pron_t, "fluency": fluency_t,
                "answer_evaluation": eval_t, "emotion": emotion,
                "personalized_feedback": personal_t,
                "overall_score": overall_score, "passing_score": PASSING_SCORE,
                "improvement": improvement,  
                "should_retry": False, "turn_number": session["turn_number"],
                "audio_url": follow_up_audio
            }
    
    except Exception as e:
        logger.exception(f"Error in practice_interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
