async def generate_question_llm(level: str, scenario: str, user_name: str, target_lang: str, model: str = "gpt") -> tuple:
    """generate question and hint"""
    
    # Add strong language instruction
    lang_instruction = ""
    if target_lang.lower() not in ["en", "english"]:
        lang_instruction = f"\n\nCRITICAL: You MUST write the ENTIRE question and hint in {target_lang} language. DO NOT use English. The user is practicing {target_lang}, so everything must be in {target_lang}."
    
    prompt = f"""You are a friendly {BOT_ROLE} helping {user_name} practice {scenario.replace('_', ' ')} conversations.

IMPORTANT RULES:
- YOU (the bot) ALWAYS ask the questions
- The USER ({user_name}) ALWAYS answers/responds
- Play the appropriate role for the scenario (staff, interviewer, shopkeeper, receptionist, etc.)
- The user plays the customer/guest/interviewee role
{lang_instruction}

SCENARIO: {scenario}
USER LEVEL: {level}

LEVEL GUIDELINES (B1 minimum - no A1/A2):
- B1: Can handle familiar situations, routine conversations, basic opinions
- B2: Independent user, can discuss abstract topics, argue a viewpoint
- C1: Fluent, can express complex ideas spontaneously, varied vocabulary
- C2: Native-like mastery, nuanced, sophisticated, precise

Generate a {level}-appropriate conversation starter for {scenario} IN {target_lang} LANGUAGE.

Return JSON:
{{"question": "[your question to the user IN {target_lang}]", "hint": "[example response the user could give at {level} level IN {target_lang}]"}}"""
    
    try:
        raw = await call_llm(prompt, model=model, target_language=target_lang)  # ✅ Pass target_language
        # rest of the code...
      async def generate_follow_up_llm(user_response: str, target_lang: str, chat_history: list, model: str = "gpt") -> tuple:
    """generate follow-up question and hint - more interactive and friendly"""
    
    recent_chat = chat_history[-6:] if chat_history else []
    chat_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_chat])
    
    # Add strong language instruction
    lang_instruction = ""
    if target_lang.lower() not in ["en", "english"]:
        lang_instruction = f"\n\nCRITICAL: You MUST write the ENTIRE question and hint in {target_lang} language. DO NOT use English."
    
    prompt = f"""You are a warm, friendly language tutor having a REAL conversation.

CONVERSATION SO FAR:
{chat_context}

User just said: "{user_response}"
{lang_instruction}

IMPORTANT RULES:
1. This is a REAL CHAT - remember everything from the conversation above
2. Your response should FLOW NATURALLY from what was discussed  
3. FIRST react to what they said (be specific - mention THEIR words!)
4. THEN ask a follow-up that RELATES to the conversation
5. Make it feel like chatting with a friend, not an interview
6. NEVER ask random unrelated questions
7. NEVER repeat questions already asked

Generate question and hint IN {target_lang} LANGUAGE.

Return JSON:
{{"question": "[your specific reaction + contextual follow-up question IN {target_lang}]", "hint": "Example: [sample answer in {target_lang}]"}}"""
    
    try:
        raw = await call_llm(prompt, model=model, target_language=target_lang)  # ✅ Pass target_language
async def generate_context_aware_follow_up(user_response: str, chat_history: list, scenario: str, user_type: str, model: str = "gpt", target_language: str = "en") -> tuple:
    """Generate context-aware follow-up question with natural transitions"""
    
    recent_chat = chat_history[-6:] if chat_history else []
    chat_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_chat])
    
    # Add strong language instruction
    lang_instruction = ""
    if target_language.lower() not in ["en", "english"]:
        lang_instruction = f"\n\nCRITICAL: You MUST write the ENTIRE question and hint in {target_language} language. DO NOT use English. The user is practicing {target_language}."
    
    prompt = f"""You are {BOT_NAME}, a warm and friendly language tutor practicing {scenario} conversations.

CONVERSATION SO FAR:
{chat_context}

The learner just said: "{user_response}"
{lang_instruction}

IMPORTANT RULES:
1. YOU (the bot) ALWAYS ask the questions
2. The USER ALWAYS answers/responds
3. This is a REAL CHAT - remember everything from the conversation
4. Start with an interactive reaction to their answer (be specific!)
5. Then ask a follow-up question they can ANSWER
6. NEVER ask them to ask YOU something
7. NEVER repeat questions already asked
8. Write EVERYTHING in {target_language} language

Return STRICTLY valid JSON:
{{"question": "[your interactive reaction + follow-up question IN {target_language}]", "hint": "[example response in {target_language}]"}}"""

    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)  # ✅ Pass target_language
