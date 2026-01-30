@router.get("/scenarios")
async def get_user_scenarios_from_db(current_user: User = Depends(get_current_user)):
    """
    Get distinct scenarios practiced by the current user from DB session data.
    Excludes scenarios that match context_names from ChatManagement table.
    """
    user_id = current_user.id if current_user else None
    scenarios = await db.get_distinct_scenarios_by_user(user_id, session_type="fluent")
    
    # Get context_names from ChatManagement to filter out
    try:
        context_names = await db.get_chat_management_context_names()
        # Filter out scenarios that match any context_name (case-insensitive)
        context_names_lower = [cn.lower() for cn in context_names]
        filtered_scenarios = [s for s in scenarios if s.lower() not in context_names_lower]
    except Exception:
        # If ChatManagement table doesn't exist or query fails, return all scenarios
        filtered_scenarios = scenarios
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_scenarios": len(filtered_scenarios),
        "scenarios": filtered_scenarios
    }
@router.get("/scenarios")
async def get_user_scenarios_from_db(current_user: User = Depends(get_current_user)):
    """
    Get distinct scenarios practiced by the current user from DB session data.
    Excludes scenarios that match context_names from ChatManagement table.
    """
    from utils.db import SessionLocal
    from sqlalchemy import text
    import asyncio
    
    user_id = current_user.id if current_user else None
    scenarios = await db.get_distinct_scenarios_by_user(user_id, session_type="fluent")
    
    # Get context_names from ChatManagement (sync db) to filter out
    def get_context_names_sync():
        sync_db = SessionLocal()
        try:
            result = sync_db.execute(text("SELECT DISTINCT context_name FROM chat_management WHERE context_name IS NOT NULL"))
            return [r[0] for r in result.fetchall() if r[0]]
        finally:
            sync_db.close()
    
    try:
        loop = asyncio.get_event_loop()
        context_names = await loop.run_in_executor(None, get_context_names_sync)
        # Filter out scenarios that match any context_name (case-insensitive)
        context_names_lower = [cn.lower() for cn in context_names]
        filtered_scenarios = [s for s in scenarios if s.lower() not in context_names_lower]
    except Exception:
        # If ChatManagement table doesn't exist or query fails, return all scenarios
        filtered_scenarios = scenarios
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_scenarios": len(filtered_scenarios),
        "scenarios": filtered_scenarios
    }


            return sessions

    async def get_chat_management_context_names(self) -> list:
        """Get all context_names from chat_management table for filtering"""
        await self.init_db()
        async with async_session() as sess:
            result = await sess.execute(
                text("SELECT DISTINCT context_name FROM chat_management WHERE context_name IS NOT NULL")
            )
            return [r[0] for
