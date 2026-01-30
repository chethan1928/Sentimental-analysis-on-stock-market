    async def get_distinct_roles_by_user(self, user_id: int, session_type: str = "interview") -> list:
        """Get distinct roles from data column for a user/session_type"""
        if user_id is None:
            return []
        await self.init_db()
        async with async_session() as sess:
            result = await sess.execute(
                text("""
                    SELECT DISTINCT (data->>'role')
                    FROM user_chat_sessions
                    WHERE user_id = :uid
                      AND session_type = :stype
                      AND data->>'role' IS NOT NULL
                    ORDER BY (data->>'role')
                """),
                {"uid": user_id, "stype": session_type}
            )
            return [r[0] for r in result.fetchall() if r[0]]

    async def get_distinct_scenarios_by_user(self, user_id: int, session_type: str = "fluent") -> list:
        """Get distinct scenarios from data column for a user/session_type"""
        if user_id is None:
            return []
        await self.init_db()
        async with async_session() as sess:
            result = await sess.execute(
                text("""
                    SELECT DISTINCT (data->>'scenario')
                    FROM user_chat_sessions
                    WHERE user_id = :uid
                      AND session_type = :stype
                      AND data->>'scenario' IS NOT NULL
                    ORDER BY (data->>'scenario')
                """),
                {"uid": user_id, "stype": session_type}
            )
            return [r[0] for r in result.fetchall() if r[0]]
@router.get("/scenarios")
async def get_user_scenarios_from_db(current_user: User = Depends(get_current_user)):
    """
    Get distinct scenarios practiced by the current user from DB session data.
    """
    user_id = current_user.id if current_user else None
    scenarios = await db.get_distinct_scenarios_by_user(user_id, session_type="fluent")
    return {
        "status": "success",
        "user_id": user_id,
        "total_scenarios": len(scenarios),
        "scenarios": scenarios
    }
@router.get("/roles")
async def get_user_roles_from_db(current_user: User = Depends(get_current_user)):
    """
    Get distinct job roles practiced by the current user from DB session data.
    """
    user_id = current_user.id if current_user else None
    roles = await db.get_distinct_roles_by_user(user_id, session_type="interview")
    return {
        "status": "success",
        "user_id": user_id,
        "total_roles": len(roles),
        "roles": roles
    }

@router.get("/session_ids_by_role")
async def get_interview_session_ids_by_role(
    role: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get session IDs for interview sessions filtered by role for the current user.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")

    ids = []
    for s in sessions:
        data = await db.get_user_session(s.get("session_id"))
        if data and data.get("role") == role:
            ids.append(s.get("session_id"))

    return {"user_id": user_id, "role": role, "session_ids": ids}
@router.get("/session_ids_by_scenario")
async def get_fluent_session_ids_by_scenario(
    scenario: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get session IDs for fluent sessions filtered by scenario for the current user.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="fluent")

    ids = []
    for s in sessions:
        data = await db.get_user_session(s.get("session_id"))
        if data and data.get("scenario") == scenario:
            ids.append(s.get("session_id"))

    return {"user_id": user_id, "scenario": scenario, "session_ids": ids}
