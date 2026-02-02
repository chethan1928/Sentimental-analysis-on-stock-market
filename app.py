@router.get("/final_feedback/{session_id}")
async def get_interview_feedback(session_id: str):
    """
    Get the exact same response as session termination.
    Returns the stored final_feedback from DB with keys in the same order as termination response.
    """
    session = await db.get_user_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Session not completed yet")
    final_feedback = session.get("final_feedback")
    if not final_feedback:
        raise HTTPException(status_code=404, detail="Final feedback not found")
    
    # Return the response with keys in the exact same order as handle_session_termination
    # This ensures the response order matches what was returned when action="end" was called
    ordered_response = {
        "status": final_feedback.get("status", "conversation_ended"),
        "session_id": final_feedback.get("session_id", session_id),
        "target_lang": final_feedback.get("target_lang", session.get("target_language", "en")),
        "native_lang": final_feedback.get("native_lang", session.get("native_language", "hi")),
        "final_scores": final_feedback.get("final_scores", {}),
        "overall_score": final_feedback.get("overall_score", 0),
        "passing_score": final_feedback.get("passing_score", PASSING_SCORE),
        "average_wpm": final_feedback.get("average_wpm", 0),
        "wpm_per_turn": final_feedback.get("wpm_per_turn", []),
        "wpm_status": final_feedback.get("wpm_status", "normal"),
        "vocab_overall": final_feedback.get("vocab_overall", {}),
        "strengths": final_feedback.get("strengths", []),
        "improvement_areas": final_feedback.get("improvement_areas", []),
        "total_turns": final_feedback.get("total_turns", 0),
        "turn_history": final_feedback.get("turn_history", []),
        "turn_feedback": final_feedback.get("turn_feedback", []),
        "summary": final_feedback.get("summary", {}),
        "overall_assessment": final_feedback.get("overall_assessment", ""),
        "grammar_feedback": final_feedback.get("grammar_feedback", {}),
        "vocabulary_feedback": final_feedback.get("vocabulary_feedback", {}),
        "pronunciation_feedback": final_feedback.get("pronunciation_feedback", {}),
        "fluency_feedback": final_feedback.get("fluency_feedback", {}),
        "interview_skills": final_feedback.get("interview_skills", {}),
        "action_plan": final_feedback.get("action_plan", []),
        "encouragement": final_feedback.get("encouragement", ""),
        "next_practice_topics": final_feedback.get("next_practice_topics", [])
    }
    
    return ordered_response
