@router.get("/final_feedback/{session_id}")
async def get_interview_feedback(session_id: str):
    """
    Get the exact same response as session termination.
    Simply returns the stored final_feedback from DB.
    """
    session = await db.get_user_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Session not completed yet")
    final_feedback = session.get("final_feedback")
    if not final_feedback:
        raise HTTPException(status_code=404, detail="Final feedback not found")
    allowed_keys = {
        "status",
        "session_id",
        "target_lang",
        "native_lang",
        "final_scores",
        "overall_score",
        "passing_score",
        "average_wpm",
        "wpm_per_turn",
        "wpm_status",
        "vocab_overall",
        "strengths",
        "improvement_areas",
        "total_turns",
        "turn_history",
        "turn_feedback",
        "summary",
        "overall_assessment",
        "grammar_feedback",
        "vocabulary_feedback",
        "pronunciation_feedback",
        "fluency_feedback",
        "interview_skills",
        "action_plan",
        "encouragement",
        "next_practice_topics",
    }
    response = {k: final_feedback[k] for k in allowed_keys if k in final_feedback}
    if "session_id" not in response:
        response["session_id"] = session_id
    if "target_lang" not in response:
        response["target_lang"] = session.get("target_language", "en")
    if "native_lang" not in response:
        response["native_lang"] = session.get("native_language", "hi")
    return response
