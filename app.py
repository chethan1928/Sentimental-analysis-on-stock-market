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
    return final_feedback
