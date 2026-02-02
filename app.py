@router.get("/feedback/{session_id}")
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
    
    # Just return what was stored during termination
    return {
        "status": "conversation_ended",
        "session_id": session_id,
        "target_lang": session.get("target_language", "en"),
        "native_lang": session.get("native_language", "hi"),
        **final_feedback  # Spread all the stored data
    }
