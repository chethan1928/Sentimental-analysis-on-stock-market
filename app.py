@router.get("/user_sessions")
async def get_interview_sessions_by_user(
    role: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all Interview sessions for the authenticated user.
    
    Optionally filter by role (e.g., 'software', 'marketing', 'sales').
    Returns sessions with session_ids included.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")
    
    
    if role:
        filtered_sessions = []
        for session in sessions:
            session_data = await db.get_user_session(session.get("session_id"))
            if session_data and session_data.get("role") == role:
                session["role"] = role
                filtered_sessions.append(session)
        sessions = filtered_sessions
    else:
        
        for session in sessions:
            session_data = await db.get_user_session(session.get("session_id"))
            if session_data:
                session["role"] = session_data.get("role", "unknown")
    
    for idx, session in enumerate(sessions, 1):
        session["session_number"] = f"Session {idx}"
    
    session_ids = [s.get("session_id") for s in sessions]
    
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "filter": {"role": role} if role else None,
        "session_ids": session_ids,
        "sessions": sessions
    }
