@router.get("/user_scenarios")
async def get_user_scenarios(
    current_user: User = Depends(get_current_user)
):
    """
    Get all unique scenarios the user has practiced from their session data in DB.
    Returns list of scenarios with count of sessions for each.
    """
    user_id = current_user.id
    sessions = await db.get_sessions_by_user_id(user_id, session_type="fluent")
    
    # Extract scenarios from each session
    scenario_counts = {}
    for session in sessions:
        session_data = await db.get_user_session(session.get("session_id"))
        if session_data:
            scenario = session_data.get("scenario")
            if scenario:
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    
    # Build response with scenario details
    scenarios = [
        {"scenario": sc, "practice_count": count, "display_name": sc.replace("_", " ").title()}
        for sc, count in scenario_counts.items()
    ]
    
    # Sort by practice count (most practiced first)
    scenarios.sort(key=lambda x: x["practice_count"], reverse=True)
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_unique_scenarios": len(scenarios),
        "scenarios": scenarios
    }
   

@router.get("/user_roles")
async def get_user_roles(
    current_user: User = Depends(get_current_user)
):
    """
    Get all unique job roles the user has practiced from their session data in DB.
    Returns list of roles with count of sessions for each.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")
    
    # Extract roles from each session
    role_counts = {}
    for session in sessions:
        session_data = await db.get_user_session(session.get("session_id"))
        if session_data:
            role = session_data.get("role")
            if role:
                role_counts[role] = role_counts.get(role, 0) + 1
    
    # Build response with role details
    roles = [
        {"role": r, "practice_count": count, "display_name": r.replace("_", " ").title()}
        for r, count in role_counts.items()
    ]
    
    # Sort by practice count (most practiced first)
    roles.sort(key=lambda x: x["practice_count"], reverse=True)
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_unique_roles": len(roles),
        "roles": roles
    }
