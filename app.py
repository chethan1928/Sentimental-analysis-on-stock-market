async def handle_session_termination(session: dict, session_id: str, model: str = "gpt") -> dict:
    """
    Helper function to handle session termination - eliminates duplicate code.
    Returns the termination response with LLM-generated summary.
    """
    count = max(1, session["scores"]["count"])
    audio_count = session["scores"].get("audio_count", 0)
    if not audio_count and (
        session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
    ):
        audio_count = count
    
    
    has_audio_turns = session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
    
    if has_audio_turns:
        pronunciation_avg = int(session["scores"]["pronunciation"] / audio_count) if audio_count > 0 else 0
        fluency_avg = int(session["scores"]["fluency"] / audio_count) if audio_count > 0 else 0
        final_scores = {
            "grammar": int(session["scores"]["grammar"] / count),
            "vocabulary": int(session["scores"]["vocabulary"] / count),
            "pronunciation": pronunciation_avg,
            "fluency": fluency_avg
        }
        avg_answer_score = int(session["scores"].get("answer", 50 * count) / count)
        overall = int(
            final_scores["grammar"] * 0.25 +
            final_scores["vocabulary"] * 0.25 +
            avg_answer_score * 0.25 +
            final_scores["pronunciation"] * 0.15 +
            final_scores["fluency"] * 0.10
        )
        average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
    else:
        
        final_scores = {
            "grammar": int(session["scores"]["grammar"] / count),
            "vocabulary": int(session["scores"]["vocabulary"] / count),
            "pronunciation": None,
            "fluency": None
        }
        avg_answer_score = int(session["scores"].get("answer", 50 * count) / count)
        
        overall = int(
            final_scores["grammar"] * 0.33 +
            final_scores["vocabulary"] * 0.33 +
            avg_answer_score * 0.34
        )
        average_wpm = 0

    
    improvement_areas = [area for area, score in final_scores.items() if score is not None and score < 70]
    strengths = [area for area, score in final_scores.items() if score is not None and score >= 80]
    
    
    turn_history = session.get("turn_history", [])
    
    # Aggregate vocab CEFR words and WPM per turn
    wpm_per_turn = []
    vocab_overall = {
        "A1": {"count": 0, "words": []},
        "A2": {"count": 0, "words": []},
        "B1": {"count": 0, "words": []},
        "B2": {"count": 0, "words": []},
        "C1": {"count": 0, "words": []},
        "C2": {"count": 0, "words": []}
    }
    
    for attempt in session.get("attempts", []):
        # Track WPM per turn
        fluency_data = attempt.get("fluency") or {}
        turn_wpm = fluency_data.get("wpm", 0) if fluency_data else 0
        wpm_per_turn.append({"turn": len(wpm_per_turn) + 1, "wpm": turn_wpm})
        
        # Aggregate CEFR vocabulary words
        vocab_data = attempt.get("vocabulary") or {}
        cefr_dist = vocab_data.get("cefr_distribution", {}) if vocab_data else {}
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            level_data = cefr_dist.get(level, {})
            if isinstance(level_data, dict):
                words = level_data.get("words", [])
                if isinstance(words, list):
                    vocab_overall[level]["words"].extend(words)
                    vocab_overall[level]["count"] = len(set(vocab_overall[level]["words"]))
    
    # Deduplicate vocab words and calculate percentages
    total_vocab_words = sum(len(set(vocab_overall[level]["words"])) for level in vocab_overall)
    for level in vocab_overall:
        vocab_overall[level]["words"] = list(set(vocab_overall[level]["words"]))
        vocab_overall[level]["count"] = len(vocab_overall[level]["words"])
        vocab_overall[level]["percentage"] = round((vocab_overall[level]["count"] / total_vocab_words * 100), 1) if total_vocab_words > 0 else 0
    
    
    llm_summary = await generate_session_summary_llm(
        user_name=session["name"],
        scenario=session.get("scenario", "interview"),
        final_scores=final_scores,
        chat_history=session["chat_history"],
        total_turns=session.get("turn_number", 0),
        average_wpm=average_wpm,
        turn_history=turn_history,
        model=model
    )
    
    # Build turn_feedback for termination response (same format as /interview_feedback)
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
            "answer_evaluation": attempt.get("answer_evaluation", {}),
            "personalized_feedback": attempt.get("personalized_feedback", {}),
            "improvement": attempt.get("improvement"),
            "overall_score": attempt.get("overall_score", 0)
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

    termination_response = {
        "status": "conversation_ended", 
        "session_id": session_id,
        "target_lang": session.get("target_language", "en"),
        "native_lang": session.get("native_language", "hi"),
        "final_scores": final_scores, 
        "overall_score": overall, 
        "passing_score": PASSING_SCORE,
        "average_wpm": average_wpm,
        "wpm_per_turn": wpm_per_turn,
        "wpm_status": "slow" if average_wpm < 110 else "normal" if average_wpm <= 160 else "fast",
        "vocab_overall": vocab_overall,
        "strengths": strengths, 
        "improvement_areas": improvement_areas,
        "total_turns": session.get("turn_number", 0),
        "turn_history": turn_history,  
        "turn_feedback": turn_feedback,
        "summary": summary,
        "overall_assessment": llm_summary.get("overall_assessment", ""),
        "grammar_feedback": llm_summary.get("grammar_feedback", {}),
        "vocabulary_feedback": llm_summary.get("vocabulary_feedback", {}),
        "pronunciation_feedback": llm_summary.get("pronunciation_feedback", {}),
        "fluency_feedback": llm_summary.get("fluency_feedback", {}),
        "interview_skills": llm_summary.get("interview_skills", {}),
        "action_plan": llm_summary.get("action_plan", []),
        "encouragement": llm_summary.get("encouragement", ""),
        "next_practice_topics": llm_summary.get("next_practice_topics", [])
    }
    await db.complete_session(session_id, final_feedback=termination_response)

    return termination_response
