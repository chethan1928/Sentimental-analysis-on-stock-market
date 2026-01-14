@router.get("/feedback/{session_id}")
async def get_interview_feedback(session_id: str):
    """
    Get detailed per-turn feedback for an interview session.
    
    Returns structured grammar, vocabulary, pronunciation, fluency and answer evaluation feedback for each turn.
    """
    feedback = await db.get_session_feedback(session_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Session not found")
    if feedback["session_type"] != "interview":
        raise HTTPException(status_code=400, detail="Not an interview session")
    return feedback


@router.get("/user_sessions/{user_id}")
async def get_interview_sessions_by_user(user_id: int):
    """
    Get all Interview sessions for a specific user.
    
    Returns all session_ids, scores, and status for the user.
    """
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")
    # Add session_number for frontend display
    for idx, session in enumerate(sessions, 1):
        session["session_number"] = f"Session {idx}"
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }
	
	





@router.get("/user_sessions/{user_id}")
async def get_fluent_sessions_by_user(user_id: int):
    """
    Get all Fluent sessions for a specific user.
    
    Returns all session_ids, scores, and status for the user.
    """
    sessions = await db.get_sessions_by_user_id(user_id, session_type="fluent")
    # Add session_number for frontend display
    for idx, session in enumerate(sessions, 1):
        session["session_number"] = f"Session {idx}"
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@router.get("/user_sessions/{user_id}/ids")
async def get_fluent_session_ids(user_id: int):
    """
    Get just the session IDs for a user.
    
    Returns only session_ids list that can be used to fetch individual feedback.
    """
    sessions = await db.get_sessions_by_user_id(user_id, session_type="fluent")
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "session_ids": [s.get("session_id") for s in sessions]
    }
	
	
	
	
	
	
	
	
	
	
	        {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "context": "appropriate for {user_type}", "example_sentence": "The presentation was excellent and impressed everyone."}},
        {{"word": "awareded", "current_level": "spelling_error", "better_word": "awarded", "suggested_level": "B1", "context": "correct spelling", "example_sentence": "She was awarded the prize for her hard work."}}
