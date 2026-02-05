
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Request, Depends
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from openai import AzureOpenAI
from pydub import AudioSegment
from logger_setup import logger
from utils.tts_utils import generate_tts_url, load_language_mapping, normalize_language_code

 

# from utils.common_utils import llm_client

# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

QWEN_ENABLED = False
qwen_client = None  
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"  

_whisper_model = WhisperModel("small", compute_type="int8")

router = APIRouter()

BOT_NAME = "Clara"
PASSING_SCORE = 50
TERMINATION_PHRASES = ["exit", "stop", "end", "finish", "quit", "done", "bye", "goodbye"]

def normalize_user_type(user_type: Optional[str]) -> str:
    """Normalize user_type to 'student' or 'professional'."""
    if not user_type:
        return "student"
    value = user_type.strip().lower()
    student_values = {
        "student", "fresher", "freshers", "college", "campus", "intern",
        "internship", "new grad", "newgrad", "graduate", "entry", "entry-level"
    }
    professional_values = {
        "professional", "experienced", "exp", "pro", "working", "industry"
    }
    if value in student_values:
        return "student"
    if value in professional_values:
        return "professional"
    if any(token in value for token in ["student", "fresher", "college", "campus", "intern", "new grad", "graduate", "entry"]):
        return "student"
    return "professional"

INTERVIEW_SCENARIOS = {
    "marketing": "Marketing Executive HR Interview",
    "sales": "Sales Representative HR Interview",
    "software": "Software Engineer HR Interview (Non-Technical)",
    "business_analyst": "Business Analyst HR Interview",
    "self_intro": "Self Introduction",
    "college_interview": "College Admission Interview",
    "job_interview": "General Job Interview",
    "professor_talk": "Talk with Professor",
    "behavioral": "Behavioral Interview",
    "technical": "Technical Interview"
}


GRAMMAR_FIELDS = ["feedback", "filler_feedback", "errors", "word_suggestions", "corrected_sentence", "improved_sentence", "strengths"]
VOCAB_FIELDS = ["feedback", "suggestions", "word_levels"]
PRON_FIELDS = ["feedback", "words_to_practice"]
FLUENCY_FIELDS = ["feedback"]
EVAL_FIELDS = ["clarity", "structure", "relevance", "confidence", "issue_summary", "improved_answer"]
PERSONAL_FIELDS = ["message", "improvement_areas", "strengths"]

async def call_llm(prompt: str, mode: str = "chat", timeout: int = 30, model: str = "gpt", target_language: str = "en") -> str:
    """async llm call with proper error handling and timeout. Supports gpt (default) or qwen."""
    base_prompts = {
        "chat": "You are a kind, human-like conversational interview coach.",
        "analysis": "You are an expert language evaluator. Analyze objectively and concisely.",
        "strict_json": "You are a structured evaluator. Respond ONLY in valid JSON. No extra text."
    }
    
    lang_lower = target_language.lower() if target_language else "en"
    is_english = lang_lower in ["en", "english"]
    lang_instruction = f" IMPORTANT: Respond entirely in {target_language} language." if not is_english else ""
    system_prompts = {k: v + lang_instruction for k, v in base_prompts.items()}
    
    
    if model.lower() == "qwen" and QWEN_ENABLED and qwen_client is not None:
        try:
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    qwen_client.chat.completions.create,
                    model=QWEN_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompts.get(mode, system_prompts["chat"])},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.7 if mode == "chat" else 0.3
                ),
                timeout=timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Qwen call failed, falling back to GPT: {e}")
            
    
    
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm_client.chat.completions.create,
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompts.get(mode, system_prompts["chat"])},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7 if mode == "chat" else 0.3
            ),
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out after {timeout}s")
        return ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


async def translate_text(text: str, source: str, target: str) -> str:
    """translate text between languages"""
    if source == target or not text or not isinstance(text, str):
        return text if isinstance(text, str) else ""
    try:
        translator = GoogleTranslator(source=source, target=target)
        return await asyncio.to_thread(translator.translate, text)
    except Exception as e:
        logger.debug(f"Translation failed: {e}")
        return text


async def translate_if_needed(text: str, target_language: str) -> str:
    """Translate English fallback text into target language when needed."""
    if not isinstance(text, str):
        return text
    if not target_language or target_language.lower() in ["en", "english"]:
        return text
    return await translate_text(text, "en", target_language)


async def translate_values(value, target_language: str):
    """Translate all string values in nested structures to target language."""
    if not target_language or target_language.lower() in ["en", "english"]:
        return value
    if isinstance(value, str):
        return await translate_text(value, "en", target_language)
    if isinstance(value, list):
        return [await translate_values(v, target_language) for v in value]
    if isinstance(value, dict):
        return {k: await translate_values(v, target_language) for k, v in value.items()}
    return value


async def make_bilingual(value, source: str, target: str):
    """Convert a value to {target, native} structure with translations"""
    if source == target:
        return value  
    
    if isinstance(value, str):
        if not value.strip():
            return {"target": value, "native": value}
        native = await translate_text(value, source, target)
        return {"target": value, "native": native}
    
    elif isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                
                translated_item = {}
                for k, v in item.items():
                    translated_item[k] = await make_bilingual(v, source, target)
                result.append(translated_item)
            elif isinstance(item, str):
                native = await translate_text(item, source, target)
                result.append({"target": item, "native": native})
            else:
                result.append(item)
        return result
    
    elif isinstance(value, dict):
        
        result = {}
        for k, v in value.items():
            result[k] = await make_bilingual(v, source, target)
        return result
    
    else:
        return value


async def translate_analysis(analysis: dict, source: str, target: str, fields_to_translate: list) -> dict:
    """Translate specified fields in analysis dict to target/native format"""
    if source == target:
        return analysis  
    
    result = {}
    for key, value in analysis.items():
        if key in fields_to_translate:
            result[key] = await make_bilingual(value, source, target)
        else:
            result[key] = value  
    return result

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# language_codes_path = os.path.join(BASE_DIR, "language-codes.json")

# def load_language_mapping():
#     try:
#         if not os.path.exists(language_codes_path):
#             logger.error(f"File not found - {language_codes_path}")
            
#         with open(language_codes_path, "r") as f:
#             language_codes = json.load(f)
#             print("Language codes loaded:", language_codes)
#             return language_codes
#     except:
#         logger.error("Error loading language mapping.")
#         return {}


async def transcribe_audio_file(audio_file: UploadFile, target_lang: str = "en") -> str:
    """Transcribe audio forcing target language (no auto-detect)."""
    try:
        audio_file.file.seek(0)
    except:
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        temp_upload = tmp.name
    
    audio_path = None
    try:
        audio = AudioSegment.from_file(temp_upload)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_path = temp_upload.replace('.tmp', '_converted.wav')
        audio.export(audio_path, format="wav")

        # Normalize target language code using load_language_mapping (consistent with fluent_api_v2.py)
        languages_data = load_language_mapping()
        normalized_target = languages_data.get(target_lang.lower(), target_lang.lower()) if target_lang else "en"

        # Force Whisper to transcribe in the target language
        logger.debug(f"Transcribing audio with forced language: {normalized_target}")
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, task="transcribe", language=normalized_target
        )
        user_text = " ".join([seg.text for seg in segments]).strip()
        logger.debug(f"Whisper transcribed in {normalized_target}: {user_text[:100] if user_text else 'empty'}")

        return user_text
    except Exception as e:
        logger.debug(f"Audio transcription failed: {e}")
        return ""
    finally:
        
        if os.path.exists(temp_upload):
            try:
                os.unlink(temp_upload)
            except:
                pass
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

TYPE_KEYWORDS = {
    "hr": "hr", "human resource": "hr", "behavioral": "behavioral", "behavior": "behavioral",
    "technical": "technical", "tech": "technical", "managerial": "managerial", "management": "managerial",
    "general": "general", "normal": "general"
}


# async def extract_role_from_text(user_text: str, model: str = "gpt") -> dict:
#     """Extract job role from natural language using LLM only - accepts ANY role"""
    
#     prompt = f"""Extract the job role/position from this text: "{user_text}"

# If a job role/position is mentioned (e.g., "software engineer", "electrical engineer", "teacher", "chef", "pilot", etc.), 
# extract it EXACTLY as the user said it and capitalize properly.

# Return JSON: {{"success": true, "role": "Exact Job Title"}}
# If no job role is mentioned: {{"success": false, "role": null}}

# Return ONLY valid JSON."""
    
#     try:
#         raw = await call_llm(prompt, mode="strict_json", timeout=10, model=model)
#         json_match = re.search(r'\{[\s\S]*\}', raw)
#         if json_match:
#             result = json.loads(json_match.group())
#             if result.get("success") and result.get("role"):
#                 return result
#     except:
#         pass
    
#     return {"success": False, "role": None}


import shutil
ffmpeg = shutil.which("ffmpeg")

async def extract_role_from_text(user_text: str, model: str = "gpt") -> dict:
   """Extract job role from natural language using LLM - accepts ANY role from audio"""
   user_lower = user_text.lower().strip()
   # If text is empty or too short
   if not user_lower or len(user_lower) < 2:
       return {"success": False, "role": None}
       
   prompt = f"""You are extracting a JOB ROLE/POSITION from user's SPEECH/AUDIO transcription.
        USER SAID: "{user_text}"
        YOUR TASK: Extract the JOB ROLE or POSITION they mentioned.
        CRITICAL RULES:
        1. Extract WHATEVER job role/position the user mentioned - it can be ANY job
        2. The user's audio might have transcription errors - understand the intent
        3. Capitalize the job title properly (e.g., "Software Engineer", "Data Scientist")
        4. Accept ANY job: traditional roles, modern roles, creative roles, anything
        EXAMPLES:
        - "software engineer" → "Software Engineer"
        - "I want to practice for data scientist role" → "Data Scientist"
        - "marketing" → "Marketing Manager"
        - "teacher" → "Teacher"
        - "chef" → "Chef"
        - "machine learning engineer" → "Machine Learning Engineer"
        - "product manager" → "Product Manager"
        - "nurse" → "Nurse"
        - "electrical engineer" → "Electrical Engineer"
        - "content writer" → "Content Writer"
        - "UI UX designer" → "UI/UX Designer"
        - "devops" → "DevOps Engineer"
        - "full stack developer" → "Full Stack Developer"
        - "hr" → "HR Manager"
        - "sales" → "Sales Executive"
        - "accountant" → "Accountant"
        - ANY job mentioned → extract and capitalize properly
        Return ONLY this JSON format:
        {{"success": true, "role": "Properly Capitalized Job Title"}}
        If NO job role is mentioned at all:
        {{"success": false, "role": null}}
        DO NOT explain. Return ONLY the JSON.
    """
   try:
       raw = await call_llm(prompt, mode="strict_json", timeout=15, model=model)
       json_match = re.search(r'\{[\s\S]*\}', raw)
       if json_match:
           result = json.loads(json_match.group())
           if result.get("success") and result.get("role"):
               return result
   except Exception as e:
       logger.exception(f"LLM role extraction failed: {e}")

   # Fallback: Try to extract role from user's words directly
   # Remove common filler words and non-meaningful sounds
   filler_words = {"i", "want", "to", "practice", "for", "the", "a", "an", "role", "position", "job", "interview", "as"}
   non_meaningful_sounds = {"hmm", "hm", "um", "uh", "uhh", "er", "err", "ah", "ahh", "oh", "okay", "ok", "yes", "no", "yeah", "yep", "nope", "like", "well", "so", "just", "maybe", "hmmmm", "ummm", "uhhh"}
   words = [w for w in re.findall(r'[a-z]+', user_lower)
            if w not in filler_words and w not in non_meaningful_sounds and len(w) > 2]
   if words:
       # Capitalize each word and join as role
       role = " ".join(word.capitalize() for word in words[:4])  # Take up to 4 words
       return {"success": True, "role": role}

   # No meaningful role found - return False so user can be asked to clarify
   return {"success": False, "role": None}


async def extract_interview_type_from_text(user_text: str, model: str = "gpt") -> dict:
    """Extract interview type from natural language - accepts ANY type"""
    user_lower = user_text.lower()
    
    
    for keyword, itype in TYPE_KEYWORDS.items():
        if keyword in user_lower:
            return {"success": True, "type": itype, "confidence": "high"}
    
    
    prompt = f"""Extract the interview type from: "{user_text}"

IMPORTANT: Accept ANY type of interview the user mentions, not just predefined ones.
Examples: hr, behavioral, technical, managerial, sales, marketing, customer service, finance, product management, design, data science, etc.

If user mentions ANY interview type, extract and format it:
- Return: {{"success": true, "type": "extracted_type_in_lowercase", "confidence": "high"}}
- Example: "I want a sales interview" → {{"success": true, "type": "sales", "confidence": "high"}}
- Example: "customer service role" → {{"success": true, "type": "customer_service", "confidence": "high"}}

If the text is completely unclear or no interview type is mentioned at all:
- Return: {{"success": false, "type": "general", "confidence": "low"}}

Return ONLY valid JSON."""
    
    try:
        raw = await call_llm(prompt, mode="strict_json", timeout=10, model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            
            if result.get("type"):
                result["success"] = True
            return result
    except:
        pass
    
    return {"success": True, "type": "general", "confidence": "low"}


async def check_answer_relevance(question: str, answer: str, model: str = "gpt", target_language: str = "en") -> dict:
    """Check if answer is relevant to question, generate friendly redirect if not"""
    
    if len(answer.split()) < 5:
        return {"relevant": True}
    
    prompt = f"""You are an interview coach. Check if this answer is COMPLETELY IRRELEVANT to the question.

Question: "{question}"
Answer: "{answer}"
Target Language: "{target_language}"

IMPORTANT RULES:
1. Be VERY LENIENT - only mark as irrelevant if the answer is about a COMPLETELY DIFFERENT TOPIC
2. If the answer even SLIGHTLY relates to the question, mark it as relevant
3. Personal stories, examples, or tangential answers should be marked RELEVANT
4. Only mark irrelevant if user talks about something totally unrelated (e.g., asked about skills but talks about weather)
5. If the answer is irrelevant, write the redirect message in the Target Language specified above

If relevant (even slightly), return: {{"relevant": true}}
If COMPLETELY UNRELATED (different topic entirely), return: {{"relevant": false, "redirect": "friendly 1-line message in the target language"}}

Return ONLY valid JSON."""
    
    try:
        raw = await call_llm(prompt, mode="strict_json", timeout=10, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            
            if not result.get("relevant", True) and not result.get("redirect"):
                return {"relevant": True}
            return result
    except:
        pass
    return {"relevant": True}  


async def compare_attempts(attempts: list, level: str = "B1", user_type: str = "professional", model: str = "gpt", target_language: str = "en") -> dict:
    """
    Compare interview attempts using LLM for detailed, elaborative feedback on ALL aspects:
    grammar, vocabulary, pronunciation, fluency, and answer quality.
    """
    if len(attempts) < 2:
        summary = "This is your first attempt. Let's see how you do!"
        if target_language and target_language != "en":
            try:
                summary = await translate_text(summary, "en", target_language)
            except Exception as e:
                logger.debug(f"Compare attempts fallback translation failed: {e}")
        return {
            "overall_improvement": 0,
            "trend": "first_attempt",
            "overall_summary": summary,
            "details": {}
        }
    
    prev = attempts[-2]
    current = attempts[-1]
    
    
    prev_grammar = (prev.get("grammar") or {}).get("score", 0) or 0
    current_grammar = (current.get("grammar") or {}).get("score", 0) or 0
    
    prev_vocab = (prev.get("vocabulary") or {}).get("score", 0) or 0
    current_vocab = (current.get("vocabulary") or {}).get("score", 0) or 0
    
    prev_pron = (prev.get("pronunciation") or {}).get("accuracy", 0) or 0
    current_pron = (current.get("pronunciation") or {}).get("accuracy", 0) or 0
    
    prev_fluency = (prev.get("fluency") or {}).get("score", 0) or 0
    current_fluency = (current.get("fluency") or {}).get("score", 0) or 0
    
    prev_answer = (prev.get("answer_evaluation") or {}).get("score", 0) or 0
    current_answer = (current.get("answer_evaluation") or {}).get("score", 0) or 0
    
    prev_overall = prev.get("overall_score", 0) or 0
    current_overall = current.get("overall_score", 0) or 0
    
    
    grammar_diff = round(current_grammar - prev_grammar, 1)
    vocab_diff = round(current_vocab - prev_vocab, 1)
    pron_diff = round(current_pron - prev_pron, 1)
    fluency_diff = round(current_fluency - prev_fluency, 1)
    answer_diff = round(current_answer - prev_answer, 1)
    overall_diff = round(current_overall - prev_overall, 1)
    
    
    if overall_diff > 10:
        trend = "significantly_improved"
    elif overall_diff > 0:
        trend = "improved"
    elif overall_diff < -10:
        trend = "declined"
    elif overall_diff < 0:
        trend = "slightly_declined"
    else:
        trend = "no_change"
    
    prompt = f"""You are an expert interview coach comparing TWO attempts at the SAME question.
Respond in the target language: {target_language}.
Provide DETAILED, ELABORATIVE feedback on improvement or decline in ALL areas.

PREVIOUS ATTEMPT:
- Overall Score: {prev_overall}%
- Grammar: {prev_grammar}%
- Vocabulary: {prev_vocab}%
- Pronunciation: {prev_pron}%
- Fluency: {prev_fluency}%
- Answer Quality: {prev_answer}%
- What they said: "{prev.get('transcription', '')[:200]}"

CURRENT ATTEMPT:
- Overall Score: {current_overall}%
- Grammar: {current_grammar}% ({'+' if grammar_diff > 0 else ''}{grammar_diff}%)
- Vocabulary: {current_vocab}% ({'+' if vocab_diff > 0 else ''}{vocab_diff}%)
- Pronunciation: {current_pron}% ({'+' if pron_diff > 0 else ''}{pron_diff}%)
- Fluency: {current_fluency}% ({'+' if fluency_diff > 0 else ''}{fluency_diff}%)
- Answer Quality: {current_answer}% ({'+' if answer_diff > 0 else ''}{answer_diff}%)
- What they said: "{current.get('transcription', '')[:200]}"

USER CONTEXT:
- Level: {level}
- User Type: {user_type}

Analyze EACH category's improvement and provide detailed, professional feedback.

Return STRICTLY valid JSON:
{{
    "overall_summary": "3-4 sentences summarizing the overall improvement journey in a professional tone.",
    "grammar_analysis": {{
        "previous_score": {prev_grammar}, "current_score": {current_grammar}, "difference": {grammar_diff},
        "improved": {str(grammar_diff > 0).lower()},
        "feedback": "Specific feedback about grammar improvement."
    }},
    "vocabulary_analysis": {{
        "previous_score": {prev_vocab}, "current_score": {current_vocab}, "difference": {vocab_diff},
        "improved": {str(vocab_diff > 0).lower()},
        "feedback": "Specific feedback about vocabulary usage."
    }},
    "pronunciation_analysis": {{
        "previous_score": {prev_pron}, "current_score": {current_pron}, "difference": {pron_diff},
        "improved": {str(pron_diff > 0).lower()},
        "feedback": "Specific feedback about pronunciation."
    }},
    "fluency_analysis": {{
        "previous_score": {prev_fluency}, "current_score": {current_fluency}, "difference": {fluency_diff},
        "improved": {str(fluency_diff > 0).lower()},
        "feedback": "Specific feedback about speaking pace."
    }},
    "answer_analysis": {{
        "previous_score": {prev_answer}, "current_score": {current_answer}, "difference": {answer_diff},
        "improved": {str(answer_diff > 0).lower()},
        "feedback": "Specific feedback about answer quality, structure, and relevance."
    }},
    "biggest_improvement": "Which area improved the most",
    "area_needing_focus": "Which area still needs work",
    "encouragement": "Professional, encouraging message",
    "next_step_tip": "One specific tip for continued improvement"
}}"""

    try:
        llm_response = await call_llm(prompt, mode="strict_json", timeout=30, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            llm_data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON")
    except Exception as e:
        logger.debug(f"LLM compare_attempts fallback: {e}")
        if overall_diff > 0:
            summary = f"Great progress! Your overall score improved from {prev_overall}% to {current_overall}% (+{overall_diff}%)."
        elif overall_diff < 0:
            summary = f"Your score changed from {prev_overall}% to {current_overall}% ({overall_diff}%). Let's work on consistency."
        else:
            summary = f"Consistent performance at {current_overall}%. Try varying your approach for improvement."
        
        llm_data = {
            "overall_summary": summary,
            "grammar_analysis": {"previous_score": prev_grammar, "current_score": current_grammar, "difference": grammar_diff, "improved": grammar_diff > 0, "feedback": f"Grammar {'improved' if grammar_diff > 0 else 'needs focus'}"},
            "vocabulary_analysis": {"previous_score": prev_vocab, "current_score": current_vocab, "difference": vocab_diff, "improved": vocab_diff > 0, "feedback": f"Vocabulary {'improved' if vocab_diff > 0 else 'needs focus'}"},
            "pronunciation_analysis": {"previous_score": prev_pron, "current_score": current_pron, "difference": pron_diff, "improved": pron_diff > 0, "feedback": f"Pronunciation {'improved' if pron_diff > 0 else 'needs focus'}"},
            "fluency_analysis": {"previous_score": prev_fluency, "current_score": current_fluency, "difference": fluency_diff, "improved": fluency_diff > 0, "feedback": f"Fluency {'improved' if fluency_diff > 0 else 'needs focus'}"},
            "answer_analysis": {"previous_score": prev_answer, "current_score": current_answer, "difference": answer_diff, "improved": answer_diff > 0, "feedback": f"Answer quality {'improved' if answer_diff > 0 else 'needs focus'}"},
            "biggest_improvement": "grammar" if grammar_diff == max(grammar_diff, vocab_diff, pron_diff, fluency_diff, answer_diff) else "answer quality",
            "area_needing_focus": "grammar" if grammar_diff == min(grammar_diff, vocab_diff, pron_diff, fluency_diff, answer_diff) else "answer quality",
            "encouragement": f"Keep practicing! Your overall score {'improved' if overall_diff > 0 else 'stayed consistent'}.",
            "next_step_tip": "Focus on structuring your answers clearly."
        }
        if target_language and target_language.lower() not in ["en", "english"]:
            llm_data = await translate_values(llm_data, target_language)
    
    return {
        "previous_overall_score": prev_overall,
        "current_overall_score": current_overall,
        "overall_improvement": overall_diff,
        "trend": trend,
        "overall_summary": llm_data.get("overall_summary", ""),
        "grammar_analysis": llm_data.get("grammar_analysis", {}),
        "vocabulary_analysis": llm_data.get("vocabulary_analysis", {}),
        "pronunciation_analysis": llm_data.get("pronunciation_analysis", {}),
        "fluency_analysis": llm_data.get("fluency_analysis", {}),
        "answer_analysis": llm_data.get("answer_analysis", {}),
        "biggest_improvement": llm_data.get("biggest_improvement", ""),
        "area_needing_focus": llm_data.get("area_needing_focus", ""),
        "encouragement": llm_data.get("encouragement", ""),
        "next_step_tip": llm_data.get("next_step_tip", "")
    }


async def generate_interactive_follow_up(user_response: str, chat_history: list, role: str, scenario: str, user_type: str = "student", model: str = "gpt", target_language: str = "en") -> tuple:
    """Generate interactive follow-up question with natural transitions"""
    
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    user_type = normalize_user_type(user_type)
    
    prompt = f"""You are {BOT_NAME}, a warm and engaging interview coach conducting a {scenario} interview for a {role} position.

    Respond in the target language: {target_language}.
    Candidate type: {user_type} (student/fresher vs professional/experienced)

    The candidate just said: "{user_response}"

Recent conversation context:
{[msg.get('content', '')[:100] for msg in recent_history[-4:]]}

CRITICAL RULES for your follow-up:
1. NEVER start with generic phrases like "That's interesting", "Great answer", "I see"
2. START by referencing something SPECIFIC they said (a keyword, example, or detail)
    3. Ask a PROBING follow-up that digs deeper or explores a new angle
    4. Include ONE encouraging word naturally (e.g., "I love that you mentioned...", "It's impressive how...")
    5. Make it conversational - like a real interview, not a quiz
    6. STRICT: If student/fresher, focus on college projects, internships, coursework, campus activities; do NOT assume work experience
    7. STRICT: If professional, focus on work experience, impact, leadership, production systems, stakeholder management

    VARIETY - Use different question types:
    - "Building on what you said about X, how would you..."
- "You mentioned X - can you walk me through a specific time when..."
- "That's a thoughtful approach to X. What challenges did you face with..."
- "I'm curious about the X you mentioned. How did that experience shape..."

Return STRICTLY valid JSON:
{{"question": "Your engaging, specific follow-up (reference their answer!)", "hint": "One practical tip for answering"}}"""

    fallback_question, fallback_hint, fallback_question_alt, fallback_hint_alt = await asyncio.gather(
        translate_if_needed("Tell me more about that.", target_language),
        translate_if_needed("Share more details.", target_language),
        translate_if_needed("Tell me more about that experience.", target_language),
        translate_if_needed("Elaborate on a specific example.", target_language)
    )
    try:
        raw = await call_llm(prompt, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", fallback_question), data.get("hint", fallback_hint)
    except:
        pass
    return fallback_question_alt, fallback_hint_alt

async def generate_interview_question(scenario: str, role: str, level: str, user_name: str, user_type: str = "student", model: str = "gpt", target_language: str = "en", turn_number: int = 0) -> tuple:
    """generate interview question with hint - first question is always an opener"""
    scenario_name = INTERVIEW_SCENARIOS.get(scenario, scenario)
    user_type = normalize_user_type(user_type)
    
    # First question should always be a standard opener
    if turn_number == 0:
        prompt = f"""You are {BOT_NAME}, a warm interview coach.

Respond in the target language: {target_language}.

    Interview scenario: {scenario_name}
    Role: {role}
    Candidate: {user_name}
    Candidate type: {user_type} (student/fresher vs professional/experienced)

    This is the FIRST question of the interview. Ask a classic opening question like:
    - "Tell me about yourself"
    - "Walk me through your background"
    - "What brings you here today?"
    STRICT: If student/fresher, ask about academic background, college projects, internships, or coursework
    STRICT: If professional, ask about recent roles, impact, and relevant experience

    Make it warm and welcoming. Keep it short and natural.

Return STRICTLY valid JSON:
{{"question": "your opening question", "hint": "suggested answer approach - mention key experiences and why you're interested in this role"}}
"""
    else:
        # Later questions should be type-specific
        prompt = f"""You are {BOT_NAME}, a warm interview coach.

Respond in the target language: {target_language}.

    Interview scenario: {scenario_name}
    Role: {role}
    Level: {level}
    Candidate: {user_name}
    Candidate type: {user_type} (student/fresher vs professional/experienced)
    Question Number: {turn_number + 1}

    Ask ONE natural interview question appropriate for this {scenario_name}.
    - For HR interviews: Ask about motivation, career goals, cultural fit, salary expectations
    - For Technical interviews: Ask about technical skills, problem-solving, coding concepts relevant to {role}
    - For Behavioral interviews: Ask situational questions (STAR method) about past experiences
    - For other types: Ask relevant domain-specific questions
    STRICT: If student/fresher, focus on college projects, internships, coursework, campus leadership, hackathons; avoid years of work experience
    STRICT: If professional, focus on work experience, impact, leadership, production systems, stakeholder management

    Provide ONE short hint for the candidate.

Return STRICTLY valid JSON:
{{"question": "your interview question", "hint": "suggested answer approach"}}
"""
    if user_type == "student":
        fallback_question, fallback_hint, fallback_question_alt, fallback_hint_alt = await asyncio.gather(
            translate_if_needed("Tell me about yourself and your academic background.", target_language),
            translate_if_needed("Mention your degree, projects, internships, and why you're interested in this role.", target_language),
            translate_if_needed("Can you walk me through your academic background?", target_language),
            translate_if_needed("Share your key projects or campus experiences and what you learned.", target_language)
        )
    else:
        fallback_question, fallback_hint, fallback_question_alt, fallback_hint_alt = await asyncio.gather(
            translate_if_needed("Tell me about yourself and your recent experience.", target_language),
            translate_if_needed("Summarize your recent role, key impact, and why you're interested in this role.", target_language),
            translate_if_needed("Can you walk me through your recent work experience?", target_language),
            translate_if_needed("Share your key accomplishments and how they relate to this role.", target_language)
        )
    try:
        raw = await call_llm(prompt, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", fallback_question), data.get("hint", fallback_hint)
    except Exception as e:
        logger.debug(f"Question generation fallback: {e}")
    return fallback_question_alt, fallback_hint_alt


async def evaluate_answer(question: str, answer: str, level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
    """evaluate interview answer quality"""
    prompt = f"""Evaluate this interview answer:

Question: {question}
Answer: {answer}
Level: {level}

Respond in the target language: {target_language}.

Return STRICTLY valid JSON:
{{
  "clarity": "Clear | Somewhat Clear | Vague",
  "structure": "Well Structured | Needs Improvement | Disorganized",
  "relevance": "Relevant | Partially Relevant | Off-topic",
  "confidence": "Confident | Neutral | Hesitant",
  "issue_summary": "brief specific feedback about the answer",
  "improved_answer": "a better version of their answer",
  "score": 0-100
}}
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except Exception as e:
        logger.debug(f"Answer evaluation fallback: {e}")
    fallback = {
        "clarity": "Clear",
        "structure": "Well Structured",
        "relevance": "Relevant",
        "confidence": "Neutral",
        "issue_summary": "Good answer overall.",
        "improved_answer": answer,
        "score": 50
    }
    if target_language and target_language != "en":
        try:
            for key in ["clarity", "structure", "relevance", "confidence", "issue_summary"]:
                fallback[key] = await translate_text(fallback[key], "en", target_language)
        except Exception as e:
            logger.debug(f"Answer evaluation fallback translation failed: {e}")
    return fallback


async def detect_emotion(user_text: str, model: str = "gpt", target_language: str = "en") -> dict:
    """detect emotion from user response"""
    prompt = f"""Analyze the emotional tone of this interview answer:

Answer: "{user_text}"
Target Language: "{target_language}"

Return STRICTLY valid JSON with the explanation written in the Target Language:
{{
  "emotion": "confident | hesitant | nervous | neutral | excited",
  "confidence_level": "high | medium | low",
  "explanation": "brief reason in target language"
}}
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.debug(f"Emotion detection fallback: {e}")
    explanation = await translate_if_needed("Tone appears neutral.", target_language)
    return {"emotion": "neutral", "confidence_level": "medium", "explanation": explanation}


# async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
#     """llm-based grammar analysis for spoken interview answers"""
#     prompt = f"""You are an expert English grammar coach analyzing SPOKEN interview responses.
# 
# Respond in the target language: {target_language}.
# 
# SPOKEN TEXT: "{user_text}"
# USER LEVEL: {level}
# 
# IMPORTANT RULES:
# 1. This is TRANSCRIBED SPEECH - IGNORE punctuation, capitalization, and minor spelling
# 2. Focus ONLY on grammatical structure and word choice
# 3. Be encouraging but honest
# 
# ANALYZE FOR:
# 
# 1. FILLER WORDS (detect ALL of these if present):
#    - um, uh, uhh, er, err, ah, ahh
#    - like (when not used correctly), you know, I mean, basically, actually, literally
#    - so, well (when used as fillers at start)
#    - kind of, sort of (when overused)
# 
# 2. GRAMMAR ERRORS (check each carefully):
#    - VERB TENSE: "I go yesterday" → "I went yesterday"
#    - SUBJECT-VERB AGREEMENT: "He don't know" → "He doesn't know"
#    - ARTICLES: "I am engineer" → "I am an engineer"
#    - PREPOSITIONS: "I am good in coding" → "I am good at coding"
#    - WORD ORDER: "Always I work hard" → "I always work hard"
#    - PRONOUNS: "Me and him went" → "He and I went"
#    - PLURALS: "I have many experience" → "I have much experience"
#    - COMPARATIVES: "more better" → "better"
# 
# 3. WORD SUGGESTIONS:
#    - Find weak/basic words and suggest stronger alternatives
#    - Example: "good" → "excellent/outstanding"
#    - Example: "bad" → "challenging/difficult"
#    - Example: "thing" → "aspect/factor/element"
#    - Example: "do" → "accomplish/execute/perform"
# 
# CRITICAL: 
# - "corrected_sentence" = Fix ONLY grammar errors
# - "improved_sentence" = Fix grammar errors AND USE all word suggestions to make it professional
# 
# SCORING GUIDE (CRITICAL - follow exactly):
# - 95-100: Perfect grammar, no errors, no filler words
# - 85-94: Minor issues only (1-2 fillers OR 1 minor error)
# - 70-84: Some issues (2-3 errors or multiple fillers)
# - 50-69: Significant issues (4+ errors)
# - Below 50: Major problems throughout
# 
# Return STRICTLY valid JSON (no extra text):
# {{
#   "score": <0-100 integer based on SCORING GUIDE above>,
#   "is_correct": <true if no major errors, false otherwise>,
#   
#   "filler_words": ["list", "of", "detected", "fillers"],
#   "filler_count": <number>,
#   "filler_feedback": "<specific advice on reducing fillers>",
#   
#   "errors": [
#     {{
#       "type": "verb_tense | article | subject_verb | preposition | word_order | pronoun | plural | comparative",
#       "you_said": "<exact phrase user said>",
#       "should_be": "<corrected phrase>",
#       "better_word": "<if applicable, show better word IN CONTEXT: 'I have excellent skills' instead of just 'excellent'>",
#       "explanation": "<brief, friendly explanation>"
#     }}
#   ],
#   
#   "word_suggestions": [
#     {{
#       "weak_word": "<basic word user used>",
#       "better_options": ["option1", "option2"],
#       "example": "<show how to use in THEIR sentence with better word>"
#     }}
#   ],
#   
#   "corrected_sentence": "<grammatically correct version - fix errors only>",
#   "improved_sentence": "<USE ALL word_suggestions to make it professional and polished>",
#   
#   "strengths": ["<what they did well grammatically>"],
#   "feedback": "<2-3 sentences: acknowledge positives, then specific improvement tips>"
# }}
# """
#     try:
#         raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
#         json_match = re.search(r'\{[\s\S]*\}', raw)
#         if json_match:
#             data = json.loads(json_match.group())
#             
#             data.setdefault("filler_words", [])
#             data.setdefault("filler_count", len(data.get("filler_words", [])))
#             data.setdefault("filler_feedback", "")
#             data.setdefault("errors", [])
#             data.setdefault("word_suggestions", [])
#             data.setdefault("strengths", [])
#             if not data.get("improved_sentence"):
#                 data["improved_sentence"] = data.get("corrected_sentence", user_text)
#             
#             
#             error_count = len(data.get("errors", []))
#             filler_count = len(data.get("filler_words", []))
#             current_score = data.get("score", 75)
#             
#             
#             if error_count == 0 and filler_count <= 1 and current_score < 90:
#                 data["score"] = 95 - (filler_count * 3)  
#             elif error_count == 1 and current_score < 80:
#                 data["score"] = 85 - (filler_count * 2)
#             elif error_count >= 4 and current_score > 70:
#                 data["score"] = min(current_score, 65)
#             
#             return data
#     except Exception as e:
#         logger.debug(f"Grammar analysis fallback: {e}")
#     return {
#         "score": 90, "is_correct": True, "filler_words": [], "filler_count": 0,
#         "filler_feedback": "", "errors": [], "word_suggestions": [],
#         "corrected_sentence": user_text, "improved_sentence": user_text,
#         "strengths": ["Good sentence structure"], "feedback": "No major grammatical issues detected. Keep up the good work!"
#     }



# async def analyze_vocab_llm(user_text: str, level: str = "Intermediate", model: str = "gpt") -> dict:
#     """llm-based vocabulary analysis with cefr levels"""
#     prompt = f"""Analyze vocabulary CEFR levels for this interview answer: "{user_text}"
 
# Level: {level}
 
# CRITICAL - SPELLING ERRORS:
# If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
# - Do NOT assign it a high CEFR level like C2
# - Include it in "suggestions" with the CORRECT SPELLING as "better_word"
 
# Calculate percentage of words at each CEFR level. Percentages should sum to 100.
 
# IMPORTANT: In the "feedback" field, DO NOT mention "A1", "A2", "B1", "B2", "C1", "C2" directly.
# Instead use:
# - A1/A2 words = "basic words" or "simple vocabulary"
# - B1/B2 words = "intermediate words" or "good vocabulary"
# - C1/C2 words = "advanced words" or "sophisticated vocabulary"
 
# CRITICAL FOR SUGGESTIONS:
# - "original_sentence": Extract the EXACT phrase from the user's transcription that contains the weak word
# - "improved_sentence": Show the SAME phrase with the better word substituted
 
# Return STRICTLY valid JSON:
# {{
#   "score": 0-100,
#   "overall_level": "A1/A2/B1/B2/C1/C2",
#   "total_words": <word count>,
#   "cefr_distribution": {{
#     "A1": {{"percentage": 20, "words": ["I", "is"]}},
#     "A2": {{"percentage": 30, "words": ["work", "name"]}},
#     "B1": {{"percentage": 40, "words": ["experience"]}},
#     "B2": {{"percentage": 10, "words": ["sophisticated"]}},
#     "C1": {{"percentage": 0, "words": []}},
#     "C2": {{"percentage": 0, "words": []}}
#   }},
#   "professional_words_used": ["list", "of", "professional", "terms"],
#   "suggestions": [
#     {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "original_sentence": "<extract from user's actual text>", "improved_sentence": "<same phrase with better word>"}}
#   ],
#   "feedback": "Feedback using 'basic', 'intermediate', 'advanced' - NOT A1/B1/C1 labels"
# }}
 
# IMPORTANT: For MISSPELLED words, set current_level = "spelling_error" and better_word = correct spelling
# """
#     try:
#         raw = await call_llm(prompt, mode="strict_json", model=model)
#         json_match = re.search(r'\{[\s\S]*\}', raw)
#         if json_match:
#             return json.loads(json_match.group())
#     except Exception as e:
#         logger.debug(f"Vocabulary analysis fallback: {e}")
#     return {
#         "score": 80, "overall_level": "B1", "total_words": len(user_text.split()),
#         "cefr_distribution": {
#             "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
#             "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
#             "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
#         },
#         "professional_words_used": [], "suggestions": [],
#         "feedback": "Vocabulary analysis could not be completed."
#     }




async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
    """llm-based grammar analysis for spoken interview answers"""
    prompt = f"""You are an expert English grammar coach analyzing SPOKEN interview responses.

Respond in the target language: {target_language}.

SPOKEN TEXT: "{user_text}"
USER LEVEL: {level}

IMPORTANT RULES:
1. This is TRANSCRIBED SPEECH - IGNORE punctuation, capitalization, and minor spelling
2. Focus ONLY on grammatical structure and word choice
3. Be encouraging but honest

ANALYZE FOR:

1. FILLER WORDS (detect ALL of these if present):
   - um, uh, uhh, er, err, ah, ahh
   - like (when not used correctly), you know, I mean, basically, actually, literally
   - so, well (when used as fillers at start)
   - kind of, sort of (when overused)

2. GRAMMAR ERRORS (check each carefully):
   - VERB TENSE: "I go yesterday" → "I went yesterday"
   - SUBJECT-VERB AGREEMENT: "He don't know" → "He doesn't know"
   - ARTICLES: "I am engineer" → "I am an engineer"
   - PREPOSITIONS: "I am good in coding" → "I am good at coding"
   - WORD ORDER: "Always I work hard" → "I always work hard"
   - PRONOUNS: "Me and him went" → "He and I went"
   - PLURALS: "I have many experience" → "I have much experience"
   - COMPARATIVES: "more better" → "better"

3. WORD SUGGESTIONS:
   - Find weak/basic words and suggest stronger alternatives
   - Example: "good" → "excellent/outstanding"
   - Example: "bad" → "challenging/difficult"
   - Example: "thing" → "aspect/factor/element"
   - Example: "do" → "accomplish/execute/perform"

CRITICAL: 
- "corrected_sentence" = Fix ONLY grammar errors
- "improved_sentence" = Fix grammar errors AND USE all word suggestions to make it professional

SCORING GUIDE (CRITICAL - follow exactly):
- 95-100: Perfect grammar, no errors, no filler words
- 85-94: Minor issues only (1-2 fillers OR 1 minor error)
- 70-84: Some issues (2-3 errors or multiple fillers)
- 50-69: Significant issues (4+ errors)
- Below 50: Major problems throughout

Return STRICTLY valid JSON (no extra text):
{{
  "score": <0-100 integer based on SCORING GUIDE above>,
  "is_correct": <true if no major errors, false otherwise>,

  "filler_words": ["list", "of", "detected", "fillers"],
  "filler_count": <number>,
  "filler_feedback": "<specific advice on reducing fillers>",

  "errors": [
    {{
      "type": "verb_tense | article | subject_verb | preposition | word_order | pronoun | plural | comparative",
      "you_said": "I #goed# to store",
      "should_be": "I #went# to the store",
      "wrong_word": "goed",
      "correct_word": "went",
      "explanation": "Go is irregular - past tense is went, not goed",
      "example_sentence": "Yesterday, I went to the park with my friends."
    }}
  ],

  "word_suggestions": [
    {{
      "you_used": "good",
      "use_instead": "excellent",
      "why": "more impactful for professional context",
      "original_sentence": "The results were #good#",
      "improved_sentence": "The results were #excellent#",
      "example_sentence": "The project outcomes were excellent."
    }}
  ],

  "corrected_sentence": "<THE WHOLE TRANSCRIPTION with ONLY grammar errors fixed>",
  "improved_sentence": "<THE WHOLE TRANSCRIPTION with grammar fixed + vocabulary enhanced>",

  "strengths": ["<what they did well grammatically>"],
  "feedback": "<2-3 sentences: acknowledge positives, then specific improvement tips>"
}}

CRITICAL FORMATTING RULES:
- For errors: you_said and should_be are ONLY the specific sentence/line from transcription containing the error
- Mark the wrong word with #word# in you_said
- Mark the correct word with #word# in should_be
- For word_suggestions: original_sentence and improved_sentence are ONLY the specific phrase containing the weak word
- Mark weak word with #word# in original_sentence, better word with #word# in improved_sentence
- example_sentence is a NEW sentence showing correct usage (not from transcription)
- corrected_sentence = THE WHOLE TRANSCRIPTION with all grammar fixes applied
- improved_sentence = THE WHOLE TRANSCRIPTION with grammar fixed AND vocabulary enhanced
- Empty arrays [] if no issues
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())

            # Normalize word_suggestions keys for consistent API response
            for item in data.get("word_suggestions", []):
                if not isinstance(item, dict):
                    continue
                if not item.get("weak_word"):
                    item["weak_word"] = item.get("you_used") or item.get("word") or ""
                if not item.get("better_options"):
                    better = item.get("use_instead") or item.get("better_word")
                    item["better_options"] = [better] if better else []

            data.setdefault("filler_words", [])
            data.setdefault("filler_count", len(data.get("filler_words", [])))
            data.setdefault("filler_feedback", "")
            data.setdefault("errors", [])
            data.setdefault("word_suggestions", [])
            data.setdefault("strengths", [])
            if not data.get("improved_sentence"):
                data["improved_sentence"] = data.get("corrected_sentence", user_text)


            error_count = len(data.get("errors", []))
            filler_count = len(data.get("filler_words", []))
            current_score = data.get("score", 75)


            if error_count == 0 and filler_count <= 1 and current_score < 90:
                data["score"] = 95 - (filler_count * 3)  
            elif error_count == 1 and current_score < 80:
                data["score"] = 85 - (filler_count * 2)
            elif error_count >= 4 and current_score > 70:
                data["score"] = min(current_score, 65)

            return data
    except Exception as e:
        logger.debug(f"Grammar analysis fallback: {e}")
    fallback_strengths = ["Good sentence structure"]
    fallback_feedback = "No major grammatical issues detected. Keep up the good work!"
    if target_language and target_language.lower() not in ["en", "english"]:
        fallback_strengths = await translate_values(fallback_strengths, target_language)
        fallback_feedback = await translate_if_needed(fallback_feedback, target_language)
    return {
        "score": 90, "is_correct": True, "filler_words": [], "filler_count": 0,
        "filler_feedback": "", "errors": [], "word_suggestions": [],
        "corrected_sentence": user_text, "improved_sentence": user_text,
        "strengths": fallback_strengths, "feedback": fallback_feedback
    }


async def analyze_vocab_llm(user_text: str, level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
    """llm-based vocabulary analysis with cefr levels"""
    prompt = f"""Analyze vocabulary CEFR levels for this interview answer: "{user_text}"

Respond in the target language: {target_language}.

Level: {level}

CRITICAL - VOCABULARY SUGGESTIONS ARE MANDATORY:
You MUST find and suggest improvements for weak/basic words like:
- good → excellent/outstanding
- bad → challenging/difficult  
- thing → aspect/factor/element
- do → accomplish/execute/perform
- get → obtain/acquire/receive
- make → create/develop/establish
- very → extremely/highly/remarkably
- nice → pleasant/wonderful/delightful
- big → substantial/significant
- small → minor/minimal

SPELLING ERRORS:
If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
- Set current_level = "spelling_error"
- Set better_word = correct spelling

Calculate percentage of words at each CEFR level. Percentages should sum to 100.
Count ALL words in the text for total_words.

IMPORTANT: In the "feedback" field, DO NOT mention "A1", "A2", "B1", "B2", "C1", "C2" directly.
Instead use:
- A1/A2 words = "basic words" or "simple vocabulary"
- B1/B2 words = "intermediate words" or "good vocabulary"
- C1/C2 words = "advanced words" or "sophisticated vocabulary"

Return STRICTLY valid JSON:
{{
  "score": 0-100,
  "overall_level": "A1/A2/B1/B2/C1/C2",
  "total_words": <actual word count>,
  "cefr_distribution": {{
    "A1": {{"percentage": 20, "words": ["I", "is", "the"]}},
    "A2": {{"percentage": 30, "words": ["work", "name", "good"]}},
    "B1": {{"percentage": 40, "words": ["experience", "actually"]}},
    "B2": {{"percentage": 10, "words": ["sophisticated"]}},
    "C1": {{"percentage": 0, "words": []}},
    "C2": {{"percentage": 0, "words": []}}
  }},
  "professional_words_used": ["list", "of", "professional", "terms"],
  "suggestions": [
    {{
      "word": "good",
      "current_level": "A2",
      "better_word": "excellent",
      "suggested_level": "B1",
      "context": "appropriate for professional interview",
      "original_sentence": "I had a #good# experience",
      "improved_sentence": "I had an #excellent# experience",
      "example_sentence": "The results of the project were excellent."
    }}
  ],
  "feedback": "Feedback using 'basic', 'intermediate', 'advanced' - NOT A1/B1/C1 labels"
}}

CRITICAL FORMATTING RULES:
- original_sentence: Extract ONLY the specific sentence/line from user's transcription containing the weak word (NOT the whole transcription)
- Mark the weak word with #word# in original_sentence
- improved_sentence: Same sentence/line with the better word substituted
- Mark the better word with #word# in improved_sentence
- example_sentence: A NEW sentence showing correct usage (not from transcription, no # needed)
- ALWAYS include suggestions if any weak/basic words (A1/A2 level) are found
- For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
- Provide at least 2-3 suggestions if weak words exist
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            # Ensure CEFR distribution has all levels
            default_cefr = {
                "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
                "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
                "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
            }
            if "cefr_distribution" not in data or not isinstance(data.get("cefr_distribution"), dict):
                data["cefr_distribution"] = default_cefr
            else:
                for level_key in default_cefr:
                    if level_key not in data["cefr_distribution"]:
                        data["cefr_distribution"][level_key] = default_cefr[level_key]
            return data
    except Exception as e:
        logger.debug(f"Vocabulary analysis fallback: {e}")
    fallback_feedback = "Vocabulary analysis could not be completed."
    if target_language and target_language.lower() not in ["en", "english"]:
        fallback_feedback = await translate_if_needed(fallback_feedback, target_language)
    return {
        "score": 80, "overall_level": "B1", "total_words": len(user_text.split()),
        "cefr_distribution": {
            "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
            "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
            "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
        },
        "professional_words_used": [], "suggestions": [],
        "feedback": fallback_feedback
    }


async def analyze_pronunciation_llm(audio_path: str = None, spoken_text: str = None, level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
    """pronunciation analysis using whisper word-level confidence"""
    
    if not audio_path:
        fallback_feedback = "No audio provided for pronunciation analysis"
        fallback_tips = ["Record audio for pronunciation feedback"]
        if target_language and target_language.lower() not in ["en", "english"]:
            fallback_feedback = await translate_if_needed(fallback_feedback, target_language)
            fallback_tips = await translate_values(fallback_tips, target_language)
        return {
            "accuracy": 75, "transcription": spoken_text or "",
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": spoken_text.split() if spoken_text else [],
            "feedback": fallback_feedback,
            "tips": fallback_tips,
            "mispronounced_count": 0
        }
    
    try:
        normalized_target = normalize_language_code(target_language, default="en")

        async def _transcribe_pronunciation(lang_hint: str = None):
            kwargs = {"word_timestamps": True}
            if lang_hint:
                kwargs["language"] = lang_hint
            segments, info = await asyncio.to_thread(_whisper_model.transcribe, audio_path, **kwargs)
            detected = info.language if info else (lang_hint or "en")
            words = []
            text = ""
            for seg in segments:
                text += seg.text + " "
                if seg.words:
                    for w in seg.words:
                        words.append({
                            "word": w.word.strip().lower(),
                            "confidence": w.probability,
                            "start": w.start,
                            "end": w.end
                        })
            return text.strip(), words, detected

        transcription, words_data, detected_lang = await _transcribe_pronunciation(normalized_target)
        display_transcription = transcription

        if not words_data:
            fallback_feedback = "No speech detected in audio"
            fallback_tips = ["Speak clearly into the microphone"]
            if target_language and target_language.lower() not in ["en", "english"]:
                fallback_feedback = await translate_if_needed(fallback_feedback, target_language)
                fallback_tips = await translate_values(fallback_tips, target_language)
            return {
                "accuracy": 0, "transcription": display_transcription,
                "word_pronunciation_scores": [],
                "words_to_practice": [], "well_pronounced_words": [],
                "feedback": fallback_feedback,
                "tips": fallback_tips,
                "mispronounced_count": 0
            }
        
        CONFIDENCE_THRESHOLD = 0.70
        mispronounced_words = []
        well_pronounced = []
        word_pronunciation_scores = []
        
        for wd in words_data:
            word = wd["word"].strip(".,!?")
            if len(word) < 2:
                continue
            
            pronunciation_percentage = round(wd["confidence"] * 100, 1)
            
            if pronunciation_percentage >= 90:
                status = "excellent"
            elif pronunciation_percentage >= 70:
                status = "good"
            elif pronunciation_percentage >= 50:
                status = "needs_improvement"
            else:
                status = "poor"
            
            word_pronunciation_scores.append({
                "word": word,
                "pronunciation_match_percentage": pronunciation_percentage,
                "status": status
            })
            
            if wd["confidence"] < CONFIDENCE_THRESHOLD:
                mispronounced_words.append({
                    "word": word,
                    "confidence": pronunciation_percentage,
                    "issue": "unclear pronunciation" if wd["confidence"] < 0.5 else "slight pronunciation issue"
                })
            else:
                well_pronounced.append(word)
        
        avg_confidence = sum(w["confidence"] for w in words_data) / len(words_data) if words_data else 0.7
        accuracy = int(avg_confidence * 100)
        
        
        llm_prompt = f"""You are a pronunciation coach for interview preparation.

Respond in the target language: {normalized_target}.

TRANSCRIPTION: "{display_transcription}"
MISPRONOUNCED WORDS: {mispronounced_words if mispronounced_words else "None - all words were clear!"}
WELL PRONOUNCED: {well_pronounced[:10]}
ACCURACY: {accuracy}%

Return STRICTLY valid JSON:
{{
    "words_to_practice": [
        {{"word": "the word", "how_to_say": "syllable breakdown: ex-AM-ple", "tip": "specific tip"}}
    ],
    "feedback": "2-3 encouraging sentences about their pronunciation for interview",
    "tips": ["general pronunciation tip 1", "general tip 2"]
}}
"""
        try:
            llm_response = await call_llm(llm_prompt, mode="strict_json", timeout=30, model=model, target_language=target_language)
            llm_data = json.loads(re.search(r'\{[\s\S]*\}', llm_response).group())
        except Exception as llm_error:
            logger.debug(f"LLM pronunciation tips fallback: {llm_error}")
            llm_data = {
                "words_to_practice": [{"word": w["word"], "how_to_say": f"Say '{w['word']}' clearly", "tip": "Speak slower"} for w in mispronounced_words[:5]],
                "feedback": f"Pronunciation accuracy: {accuracy}%.",
                "tips": ["Speak slowly and clearly", "Practice word stress"]
            }
            if target_language and target_language.lower() not in ["en", "english"]:
                llm_data["feedback"] = await translate_if_needed(llm_data.get("feedback", ""), target_language)
                llm_data["tips"] = await translate_values(llm_data.get("tips", []), target_language)
                translated_words = []
                for item in llm_data.get("words_to_practice", []):
                    if isinstance(item, dict):
                        item = item.copy()
                        item["how_to_say"] = await translate_if_needed(item.get("how_to_say", ""), target_language)
                        item["tip"] = await translate_if_needed(item.get("tip", ""), target_language)
                    translated_words.append(item)
                llm_data["words_to_practice"] = translated_words
        
        default_feedback = await translate_if_needed("Analysis complete.", target_language)
        return {
            "accuracy": accuracy,
            "transcription": display_transcription,
            "word_pronunciation_scores": word_pronunciation_scores,
            "words_to_practice": llm_data.get("words_to_practice", []),
            "well_pronounced_words": well_pronounced,
            "feedback": llm_data.get("feedback", default_feedback),
            "tips": llm_data.get("tips", []),
            "mispronounced_count": len(mispronounced_words)
        }
        
    except Exception as e:
        logger.error(f"Pronunciation error: {e}")
        fallback_feedback = f"Could not analyze pronunciation: {str(e)}"
        fallback_tips = ["Ensure clear audio recording"]
        if target_language and target_language.lower() not in ["en", "english"]:
            fallback_feedback = await translate_if_needed(fallback_feedback, target_language)
            fallback_tips = await translate_values(fallback_tips, target_language)
        return {
            "accuracy": 75, "transcription": spoken_text or "",
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": [],
            "feedback": fallback_feedback,
            "tips": fallback_tips,
            "mispronounced_count": 0
        }


def calculate_fluency(word_count: int, audio_duration: float) -> dict:
    """calculate fluency metrics"""
    wpm = int((word_count / audio_duration) * 60) if audio_duration > 0 else 100
    
    if wpm < 80:
        score = max(40, 60 - (80 - wpm))
        speed_status = "too_slow"
    elif wpm < 110:
        score = 70 + (wpm - 80)
        speed_status = "slow"
    elif wpm <= 160:
        score = 90 + min(10, (wpm - 110) // 5)
        speed_status = "normal"
    elif wpm <= 180:
        score = 85
        speed_status = "fast"
    else:
        score = max(60, 85 - (wpm - 180) // 2)
        speed_status = "too_fast"
    
    return {
        "score": min(100, score),
        "wpm": wpm,
        "speed_status": speed_status,
        "audio_duration_seconds": round(audio_duration, 1)
    }


async def analyze_fluency_metrics(user_text: str, audio_duration: float) -> dict:
    """async wrapper for fluency metrics from text and duration"""
    word_count = len(re.findall(r"\b\w+\b", user_text or ""))
    return calculate_fluency(word_count, audio_duration)


async def generate_personalized_feedback(overall_score: float, scores: dict, emotion: dict, user_name: str,
                                          grammar: dict = None, vocabulary: dict = None, 
                                          pronunciation: dict = None, answer_eval: dict = None, model: str = "gpt",
                                          target_language: str = "en") -> dict:
    """Generate personalized interview feedback using LLM based on actual errors"""
    
    
    grammar_errors = [
        e for e in (grammar.get("errors", []) if grammar else [])
        if isinstance(e, dict)
    ]
    filler_words = grammar.get("filler_words", []) if grammar else []
    word_suggestions = [
        w for w in (grammar.get("word_suggestions", []) if grammar else [])
        if isinstance(w, dict)
    ]
    vocab_suggestions = [
        v for v in (vocabulary.get("suggestions", []) if vocabulary else [])
        if isinstance(v, dict)
    ]
    mispronounced = pronunciation.get("words_to_practice", []) if pronunciation else []
    answer_issues = answer_eval.get("issue_summary", "") if answer_eval else ""
    
    
    errors_context = []
    if grammar_errors:
        errors_context.append(f"Grammar errors: {[e.get('you_said', '') + ' → ' + e.get('should_be', '') for e in grammar_errors[:3]]}")
    if filler_words:
        errors_context.append(f"Filler words used: {filler_words[:5]}")
    if word_suggestions:
        errors_context.append(f"Weak words: {[w.get('weak_word', '') for w in word_suggestions[:3]]}")
    if vocab_suggestions:
        errors_context.append(f"Vocabulary improvements: {[v.get('word', '') + ' → ' + v.get('better_word', '') for v in vocab_suggestions[:3]]}")
    if mispronounced:
        errors_context.append(f"Pronunciation to practice: {[w.get('word', '') if isinstance(w, dict) else w for w in mispronounced[:3]]}")
    if answer_issues:
        errors_context.append(f"Answer feedback: {answer_issues}")
    
    
    improvement_areas = []
    strengths = []
    for area, score in scores.items():
        if score is None:  
            continue
        if score >= 75:
            strengths.append(area)
        elif score < 65:
            improvement_areas.append(area)
    
    improvement_areas_display = improvement_areas
    strengths_display = strengths
    if target_language and target_language != "en":
        try:
            if improvement_areas:
                improvement_areas_display = list(await asyncio.gather(
                    *[translate_text(a, "en", target_language) for a in improvement_areas]
                ))
            if strengths:
                strengths_display = list(await asyncio.gather(
                    *[translate_text(s, "en", target_language) for s in strengths]
                ))
        except Exception as e:
            logger.debug(f"Personalized feedback list translation failed: {e}")

    if errors_context:
        prompt = f"""You are a professional interview coach providing constructive feedback to candidate {user_name}.

Respond in the target language: {target_language}.

SCORES:
- Grammar: {scores.get('grammar', 0)}%
- Vocabulary: {scores.get('vocabulary', 0)}%
- Pronunciation: {scores.get('pronunciation', 0)}%
- Fluency: {scores.get('fluency', 0)}%
- Answer Quality: {scores.get('answer_evaluation', 0)}%
- Overall: {overall_score}%

ACTUAL ERRORS/ISSUES FOUND:
{chr(10).join(errors_context)}

EMOTION DETECTED: {emotion.get('emotion', 'neutral')}

Generate PROFESSIONAL but ENGAGING feedback. Be encouraging yet constructive.

Return STRICTLY valid JSON:
{{
    "message": "Start with a polished, professional one-liner that acknowledges their performance (like 'That was a well-structured response.' or 'Good points raised there.' or 'I can see you're putting thought into this.'). THEN 1-2 sentences of specific, constructive feedback about their ACTUAL errors. Keep it professional but warm.",
    "improvement_areas": {json.dumps(improvement_areas_display)},
    "strengths": {json.dumps(strengths_display)},
    "emotion": "{emotion.get('emotion', 'neutral')}",
    "quick_tip": "ONE specific, actionable tip - professional tone"
}}

TONE EXAMPLES for "message" based on score:
- Score >= 85: "That was an excellent response. Your articulation was clear and..."
- Score 70-84: "Good effort on that answer. I noticed some strong points, though..."
- Score 50-69: "You're on the right track. Let's work on..."
- Score < 50: "I appreciate your attempt. Here's how we can strengthen..."

RULES:
- Professional tone (like a supportive hiring manager)
- NOT overly formal or stiff - be human and warm
- Reference ACTUAL errors constructively
- Acknowledge good attempts even when score is low"""
        
        try:
            raw = await call_llm(prompt, mode="strict_json", timeout=15, model=model, target_language=target_language)
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                result = json.loads(json_match.group())
                
                result.setdefault("improvement_areas", improvement_areas_display)
                result.setdefault("strengths", strengths_display)
                result.setdefault("emotion", emotion.get("emotion", "neutral"))
                return result
        except Exception as e:
            logger.debug(f"LLM personalized feedback fallback: {e}")
    
    
    if overall_score >= 95:
        message = f"🌟 Outstanding interview performance, {user_name}! You're interview-ready!"
    elif overall_score >= 85:
        message = f"Excellent job, {user_name}! Your communication skills are impressive."
    elif overall_score >= 70:
        message = f"Good effort, {user_name}! Focus on {', '.join(improvement_areas) if improvement_areas else 'minor details'} to improve."
    else:
        message = f"Keep practicing, {user_name}! Work on: {', '.join(improvement_areas) if improvement_areas else 'overall delivery'}."
    
    if emotion.get("confidence_level") == "low" or emotion.get("emotion") == "nervous":
        message += " Remember to take a breath and project confidence."
    
    quick_tip = f"Practice your {improvement_areas[0] if improvement_areas else 'interview skills'} regularly."
    if target_language and target_language != "en":
        try:
            message = await translate_text(message, "en", target_language)
            quick_tip = await translate_text(quick_tip, "en", target_language)
        except Exception as e:
            logger.debug(f"Personalized feedback fallback translation failed: {e}")

    return {
        "message": message,
        "improvement_areas": improvement_areas_display,
        "strengths": strengths_display,
        "emotion": emotion.get("emotion", "neutral"),
        "quick_tip": quick_tip
    }


async def generate_session_summary_llm(user_name: str, scenario: str, final_scores: dict, 
                                        chat_history: list, total_turns: int, average_wpm: int, 
                                        turn_history: list = None, model: str = "gpt",
                                        target_language: str = "en") -> dict:
    """Generate elaborative LLM-based session summary with per-turn WPM analysis"""
    
    
    conversation_summary = []
    for i, msg in enumerate(chat_history[-10:]):  
        role = "Clara" if msg["role"] == "assistant" else user_name
        conversation_summary.append(f"{role}: {msg['content'][:100]}...")
    
    
    turn_wpm_summary = ""
    if turn_history:
        turn_entries = [f"Turn {t.get('turn', i+1)}: {t.get('wpm', 0)} WPM, Score: {t.get('overall_score', 0)}%" 
                       for i, t in enumerate(turn_history)]
        turn_wpm_summary = "\n".join(turn_entries)
    
    prompt = f"""You are an expert interview coach providing a detailed session summary.

Respond in the target language: {target_language}.

CANDIDATE: {user_name}
SCENARIO: {scenario}
TOTAL QUESTIONS: {total_turns}
AVERAGE SPEAKING SPEED: {average_wpm} words per minute

FINAL SCORES:
- Grammar: {final_scores.get('grammar', 0)}%
- Vocabulary: {final_scores.get('vocabulary', 0)}%  
- Pronunciation: {final_scores.get('pronunciation', 0)}%
- Fluency: {final_scores.get('fluency', 0)}%

PER-TURN PERFORMANCE:
{turn_wpm_summary if turn_wpm_summary else "No turn data available"}

RECENT CONVERSATION:
{chr(10).join(conversation_summary)}

Generate a detailed, personalized, and encouraging session summary. Analyze WPM trend across turns.

Return STRICTLY valid JSON:
{{
    "overall_assessment": "3-4 sentences summarizing the candidate's overall interview performance, mentioning WPM trends",
    "grammar_feedback": {{
        "score": {final_scores.get('grammar', 0)},
        "status": "Excellent/Good/Needs Work",
        "what_went_well": "specific positive observation",
        "improvement_tip": "specific actionable tip",
        "example": "example of correct usage or common mistake to avoid"
    }},
    "vocabulary_feedback": {{
        "score": {final_scores.get('vocabulary', 0)},
        "status": "Excellent/Good/Needs Work",
        "what_went_well": "specific positive observation",
        "improvement_tip": "specific actionable tip",
        "suggested_words": ["professional word 1", "professional word 2", "professional word 3"]
    }},
    "pronunciation_feedback": {{
        "score": {final_scores.get('pronunciation', 0)},
        "status": "Excellent/Good/Needs Work",
        "what_went_well": "specific positive observation",
        "improvement_tip": "specific actionable tip",
        "practice_words": ["word to practice 1", "word to practice 2"]
    }},
    "fluency_feedback": {{
        "score": {final_scores.get('fluency', 0)},
        "status": "Excellent/Good/Needs Work",
        "what_went_well": "specific positive observation",
        "improvement_tip": "specific actionable tip for speaking pace",
        "wpm_trend": "analysis of WPM across turns - improving/declining/stable"
    }},
    "interview_skills": {{
        "confidence": "observation about confidence level",
        "structure": "observation about answer structure",
        "relevance": "observation about answer relevance"
    }},
    "action_plan": [
        "specific action item 1 for next week",
        "specific action item 2 for next week",
        "specific action item 3 for next week"
    ],
    "encouragement": "2-3 encouraging sentences personalized for the candidate",
    "next_practice_topics": ["topic 1", "topic 2", "topic 3"]
}}
"""
    try:
        raw = await call_llm(prompt, mode="strict_json", timeout=25, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.debug(f"Session summary LLM fallback: {e}")
    
    
    fallback = {
        "overall_assessment": f"Great effort, {user_name}! You completed {total_turns} questions in your {scenario} practice.",
        "grammar_feedback": {"score": final_scores.get("grammar", 0), "status": "Good", "what_went_well": "Good sentence structure", "improvement_tip": "Practice complex sentences", "example": "Use varied sentence structures"},
        "vocabulary_feedback": {"score": final_scores.get("vocabulary", 0), "status": "Good", "what_went_well": "Used relevant terms", "improvement_tip": "Expand professional vocabulary", "suggested_words": ["synergy", "leverage", "optimize"]},
        "pronunciation_feedback": {"score": final_scores.get("pronunciation", 0), "status": "Good", "what_went_well": "Clear articulation", "improvement_tip": "Practice difficult words", "practice_words": ["particularly", "specifically"]},
        "fluency_feedback": {"score": final_scores.get("fluency", 0), "status": "Good", "what_went_well": "Consistent pace", "improvement_tip": "Maintain steady rhythm", "wpm_trend": "stable"},
        "interview_skills": {"confidence": "Showed good confidence", "structure": "Answers were organized", "relevance": "Stayed on topic"},
        "action_plan": ["Practice speaking for 10 mins daily", "Record and review your answers", "Prepare examples for common questions"],
        "encouragement": f"Keep up the great work, {user_name}! Regular practice will help you ace your interviews.",
        "next_practice_topics": ["Tell me about yourself", "Why should we hire you?", "Describe a challenge you overcame"]
    }
    if target_language and target_language.lower() not in ["en", "english"]:
        fallback = await translate_values(fallback, target_language)
    return fallback
async def handle_session_termination(session: dict, session_id: str, model: str = "gpt") -> dict:
    """
    Helper function to handle session termination - eliminates duplicate code.
    Returns the termination response with LLM-generated summary.
    """
    count = max(1, session["scores"]["count"])
    audio_count = session["scores"].get("audio_count", 0)
    if not audio_count and (
        session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
    ):
        audio_count = count
    
    
    has_audio_turns = session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
    
    if has_audio_turns:
        pronunciation_avg = int(session["scores"]["pronunciation"] / audio_count) if audio_count > 0 else 0
        fluency_avg = int(session["scores"]["fluency"] / audio_count) if audio_count > 0 else 0
        final_scores = {
            "grammar": int(session["scores"]["grammar"] / count),
            "vocabulary": int(session["scores"]["vocabulary"] / count),
            "pronunciation": pronunciation_avg,
            "fluency": fluency_avg
        }
        avg_answer_score = int(session["scores"].get("answer", 50 * count) / count)
        overall = int(
            final_scores["grammar"] * 0.25 +
            final_scores["vocabulary"] * 0.25 +
            avg_answer_score * 0.25 +
            final_scores["pronunciation"] * 0.15 +
            final_scores["fluency"] * 0.10
        )
        average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
    else:
        
        final_scores = {
            "grammar": int(session["scores"]["grammar"] / count),
            "vocabulary": int(session["scores"]["vocabulary"] / count),
            "pronunciation": None,
            "fluency": None
        }
        avg_answer_score = int(session["scores"].get("answer", 50 * count) / count)
        
        overall = int(
            final_scores["grammar"] * 0.33 +
            final_scores["vocabulary"] * 0.33 +
            avg_answer_score * 0.34
        )
        average_wpm = 0

    
    improvement_areas = [area for area, score in final_scores.items() if score is not None and score < 70]
    strengths = [area for area, score in final_scores.items() if score is not None and score >= 80]
    
    
    turn_history = session.get("turn_history", [])
    
    # Aggregate vocab CEFR words and WPM per turn
    wpm_per_turn = []
    vocab_overall = {
        "A1": {"count": 0, "words": []},
        "A2": {"count": 0, "words": []},
        "B1": {"count": 0, "words": []},
        "B2": {"count": 0, "words": []},
        "C1": {"count": 0, "words": []},
        "C2": {"count": 0, "words": []}
    }
    
    for attempt in session.get("attempts", []):
        # Track WPM per turn
        fluency_data = attempt.get("fluency") or {}
        turn_wpm = fluency_data.get("wpm", 0) if fluency_data else 0
        wpm_per_turn.append({"turn": len(wpm_per_turn) + 1, "wpm": turn_wpm})
        
        # Aggregate CEFR vocabulary words
        vocab_data = attempt.get("vocabulary") or {}
        cefr_dist = vocab_data.get("cefr_distribution", {}) if vocab_data else {}
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            level_data = cefr_dist.get(level, {})
            if isinstance(level_data, dict):
                words = level_data.get("words", [])
                if isinstance(words, list):
                    vocab_overall[level]["words"].extend(words)
                    vocab_overall[level]["count"] = len(set(vocab_overall[level]["words"]))
    
    # Deduplicate vocab words and calculate percentages
    total_vocab_words = sum(len(set(vocab_overall[level]["words"])) for level in vocab_overall)
    for level in vocab_overall:
        vocab_overall[level]["words"] = list(set(vocab_overall[level]["words"]))
        vocab_overall[level]["count"] = len(vocab_overall[level]["words"])
        vocab_overall[level]["percentage"] = round((vocab_overall[level]["count"] / total_vocab_words * 100), 1) if total_vocab_words > 0 else 0
    
    
    llm_summary = await generate_session_summary_llm(
        user_name=session["name"],
        scenario=session.get("scenario", "interview"),
        final_scores=final_scores,
        chat_history=session["chat_history"],
        total_turns=session.get("turn_number", 0),
        average_wpm=average_wpm,
        turn_history=turn_history,
        model=model,
        target_language=session.get("target_language", "en")
    )
    
    # Build turn_feedback for termination response (same format as /interview_feedback)
    turn_feedback = []
    # Aggregate grammar mistakes and vocabulary suggestions from all turns
    grammar_mistakes = []
    vocab_suggestions = []
    pronunciation_issues = []
    
    for i, attempt in enumerate(session.get("attempts", []), 1):
        turn_feedback.append({
            "turn": i,
            "transcription": attempt.get("transcription", ""),
            "grammar": attempt.get("grammar", {}),
            "vocabulary": attempt.get("vocabulary", {}),
            "pronunciation": attempt.get("pronunciation"),
            "fluency": attempt.get("fluency"),
            "answer_evaluation": attempt.get("answer_evaluation", {}),
            "personalized_feedback": attempt.get("personalized_feedback", {}),
            "improvement": attempt.get("improvement"),
            "overall_score": attempt.get("overall_score", 0)
        })
        
        # Collect grammar errors (wrong → correct)
        gram = attempt.get("grammar") or {}
        if isinstance(gram, dict):
            for err in gram.get("errors", []):
                if isinstance(err, dict):
                    grammar_mistakes.append({
                        "wrong": err.get("you_said", err.get("wrong_word", "")),
                        "correct": err.get("should_be", err.get("correct_word", ""))
                    })
        
        # Collect vocabulary suggestions (weak word → better word)
        vocab = attempt.get("vocabulary") or {}
        if isinstance(vocab, dict):
            for sug in vocab.get("suggestions", []):
                if isinstance(sug, dict):
                    better = sug.get("better_word", "")
                    if isinstance(better, list):
                        better_options = better
                    elif better:
                        better_options = [better]
                    else:
                        better_options = []
                    vocab_suggestions.append({
                        "weak_word": sug.get("word", ""),
                        "better_options": better_options
                    })
        
        # Collect pronunciation issues
        pron = attempt.get("pronunciation") or {}
        if isinstance(pron, dict):
            for word_issue in pron.get("words_to_practice", []):
                if isinstance(word_issue, dict):
                    pronunciation_issues.append({
                        "word": word_issue.get("word", ""),
                        "issue": word_issue.get("issue", ""),
                        "how_to_say": word_issue.get("how_to_say", "")
                    })
    
    # Build summary of all mistakes
    summary = {
        "grammar": {
            "total_errors": len(grammar_mistakes),
            "errors": grammar_mistakes
        },
        "vocabulary": {
            "total_suggestions": len(vocab_suggestions),
            "suggestions": vocab_suggestions
        },
        "pronunciation": {
            "total_issues": len(pronunciation_issues),
            "issues": pronunciation_issues
        }
    }

    termination_response = {
        "status": "conversation_ended", 
        "session_id": session_id,
        "target_lang": session.get("target_language", "en"),
        "native_lang": session.get("native_language", "hi"),
        "final_scores": final_scores, 
        "overall_score": overall, 
        "passing_score": PASSING_SCORE,
        "average_wpm": average_wpm,
        "wpm_per_turn": wpm_per_turn,
        "wpm_status": "slow" if average_wpm < 110 else "normal" if average_wpm <= 160 else "fast",
        "vocab_overall": vocab_overall,
        "strengths": strengths, 
        "improvement_areas": improvement_areas,
        "total_turns": session.get("turn_number", 0),
        "turn_history": turn_history,  
        "turn_feedback": turn_feedback,
        "summary": summary,
        "overall_assessment": llm_summary.get("overall_assessment", ""),
        "grammar_feedback": llm_summary.get("grammar_feedback", {}),
        "vocabulary_feedback": llm_summary.get("vocabulary_feedback", {}),
        "pronunciation_feedback": llm_summary.get("pronunciation_feedback", {}),
        "fluency_feedback": llm_summary.get("fluency_feedback", {}),
        "interview_skills": llm_summary.get("interview_skills", {}),
        "action_plan": llm_summary.get("action_plan", []),
        "encouragement": llm_summary.get("encouragement", ""),
        "next_practice_topics": llm_summary.get("next_practice_topics", [])
    }
    await db.complete_session(session_id, final_feedback=termination_response)

    return termination_response


@router.post("/practice")
async def practice_interview(
    request: Request,
    name: str = Form(...),
    native_language: str = Form(default="hi"),
    target_language: str = Form(default="en"),
    level: str = Form(default="B1"),
    user_type: Optional[str] = Form(default="student"),
    audio_file: Optional[UploadFile] = File(default=None),
    text_input: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    action: Optional[str] = Form(default=None),  
    model: Optional[str] = Form(default="gpt"),  
    voice_id: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),

):
    """
    interview practice api - CONVERSATIONAL ONBOARDING
    
    flow:
    1. first call (no audio/text): Clara greets and asks for role
    2. user provides role: Clara asks for interview type
    3. user provides type: interview begins with first question
    4. subsequent calls: normal interview with analysis
    5. action="end" or termination phrase: ends session
    """
    try:
        user_text = ""
        audio_path = None

        if not session_id or session_id.strip() == "" or session_id == "string":
            session_id = str(uuid.uuid4())
        
        
        session = await db.get_user_session(session_id)
        session_exists = session is not None
        native_language = session.get("native_language", native_language) if session else native_language
        target_language = session.get("target_language", target_language) if session else target_language
        stored_user_type = session.get("user_type") if session else None

        native_language = normalize_language_code(native_language, default="en")
        target_language = normalize_language_code(target_language, default="en")
        user_type = normalize_user_type(stored_user_type or user_type)

        if session_exists:
            needs_update = False
            if session.get("native_language") != native_language or session.get("target_language") != target_language:
                session["native_language"] = native_language
                session["target_language"] = target_language
                needs_update = True
            if not session.get("user_type"):
                session["user_type"] = user_type
                needs_update = True
            if needs_update:
                await db.update_session(session_id, session)
        
        
        if session_exists and session.get("status") == "completed":
            error_msg = await translate_text("This session has ended. Please start a new session.", "en", native_language)
            return {"status": "error", "session_id": session_id, "error": error_msg} 
        
        if not session_exists:
            
            session = {
                "state": "welcome",  
                "name": name, 
                "scenario": None,  
                "role": None,      
                "level": level,
                "user_type": user_type,
                "native_language": native_language, 
                "target_language": target_language,
                "chat_history": [],
                "scores": {"grammar": 0, "vocabulary": 0, "pronunciation": 0, "fluency": 0, "total_wpm": 0, "count": 0, "audio_count": 0},
                "current_question": None, "current_hint": None, "turn_number": 0,
                "last_overall_score": None, "retry_count": 0, "attempts": [],
                "turn_history": [],  
                "onboarding_retry": 0  
            }
            await db.create_session(
                session_id=session_id,
                session_type="interview",
                data=session,
                user_id=current_user.id if current_user else None,
                user_name=name
            )
        
        
        current_state = session.get("state", "interviewing") 
        
        
        if current_state == "welcome" and not audio_file and not text_input:
            greeting = f"Hi {name}! I'm {BOT_NAME}, your interview coach 🙂 So, which role are you ready for?"
            
            greeting_target, greeting_native = await asyncio.gather(
                translate_text(greeting, "en", target_language),
                translate_text(greeting, "en", native_language)
            )
            
            session["state"] = "collecting_role"
            session["chat_history"].append({"role": "assistant", "content": greeting})
            await db.update_session(session_id, session)
            
            greeting_audio = await generate_tts_url(request, greeting_target, target_language, api_type="interview", voice_id=voice_id)
            
            return {
                "status": "onboarding",
                "step": "collecting_role",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "transcription": user_text,
                "message": {"target": greeting_target, "native": greeting_native},
                "audio_url": greeting_audio
            }
        
        
        if current_state == "collecting_role":
            user_text = text_input or ""
            if audio_file:
                
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                error_msg = await translate_text("No speech detected. Please tell me which role you're preparing for.", "en", native_language)
                return {"status": "error", "session_id": session_id, "error": error_msg}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_role_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("role"):
                role = extraction["role"]
                session["role"] = role
                session["state"] = "collecting_type"
                session["onboarding_retry"] = 0
                
                
                ask_type = f"Great, {role}! Is this more of an HR interview, or would you prefer something else like behavioral or technical?"
                ask_type_target, ask_type_native = await asyncio.gather(
                    translate_text(ask_type, "en", target_language),
                    translate_text(ask_type, "en", native_language)
                )
                
                session["chat_history"].append({"role": "assistant", "content": ask_type})
                await db.update_session(session_id, session)
                
                ask_type_audio = await generate_tts_url(request, ask_type_target, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_type", 
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "role": role,
                    "message": {"target": ask_type_target, "native": ask_type_native},
                    "audio_url": ask_type_audio
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "Could you be more specific about the role? For example: Software Engineer, Marketing Manager, Business Analyst, etc."
                retry_target, retry_native = await asyncio.gather(
                    translate_text(retry_msg, "en", target_language),
                    translate_text(retry_msg, "en", native_language)
                )
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                retry_audio = await generate_tts_url(request, retry_target, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_role",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "retry": True,
                    "message": {"target": retry_target, "native": retry_native},
                    "audio_url": retry_audio
                }
        
        if current_state == "collecting_type":
            user_text = text_input or ""
            if audio_file:
                
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                error_msg = await translate_text("No speech detected. Please tell me the interview type.", "en", native_language)
                return {"status": "error", "session_id": session_id, "error": error_msg}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_interview_type_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("type"):
                interview_type = extraction["type"]
                session["scenario"] = interview_type
                session["state"] = "interviewing"
                session["onboarding_retry"] = 0
                
                
                role = session.get("role", "Professional")
                scenario_name = INTERVIEW_SCENARIOS.get(interview_type, interview_type.title() + " Interview")
                
                question, hint = await generate_interview_question(
                    interview_type, role, level, name, user_type=user_type, model=model, target_language=target_language, turn_number=0
                )
                
                start_msg = f"Perfect! Let's start your {scenario_name} practice for {role}."
                start_target, start_native, q_native, h_native = await asyncio.gather(
                    translate_text(start_msg, "en", target_language),
                    translate_text(start_msg, "en", native_language),
                    translate_text(question, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                session["current_question"] = question
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": question})
                await db.update_session(session_id, session)
                
                question_audio = await generate_tts_url(request, question, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "interview_started",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "role": role,
                    "scenario": interview_type,
                    "greeting": {"target": start_target, "native": start_native},
                    "next_question": {"target": question, "native": q_native},
                    "hint": {"target": hint, "native": h_native},
                    "turn_number": 0,
                    "audio_url": question_audio
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "What type of interview would you like to practice? For example: HR, Technical, Sales, Marketing, Customer Service, or any other type?"
                retry_target, retry_native = await asyncio.gather(
                    translate_text(retry_msg, "en", target_language),
                    translate_text(retry_msg, "en", native_language)
                )
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                retry_audio = await generate_tts_url(request, retry_target, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_type",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "retry": True,
                    "message": {"target": retry_target, "native": retry_native},
                    "audio_url": retry_audio
                }
        
        
        role = session.get("role", "Professional")
        scenario = session.get("scenario", "general")
        
        if action == "next":
            follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, user_type=user_type, model=model, target_language=target_language)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            session["retry_count"] = 0
            session["waiting_retry_decision"] = False  
            session["retry_clarify_count"] = 0  
            
            
            await db.update_session(session_id, session)
            
            follow_up_audio = await generate_tts_url(request, follow_up, target_language, api_type="interview", voice_id=voice_id)
            
            skipped_msg = await translate_text("Skipped", "en", target_language) if target_language != "en" else "Skipped"
            skipped_next_msg = await translate_text("Skipped. Let's try this question!", "en", target_language) if target_language != "en" else "Skipped. Let's try this question!"

            return {
                "status": "continue", "session_id": session_id,
                "target_lang": target_language, "native_lang": native_language,
                "transcription": "(skipped)",
                "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                "grammar": {"score": 0, "is_correct": True, "errors": [], "feedback": skipped_msg},
                "vocabulary": {"score": 0, "overall_level": "skipped", "feedback": skipped_msg},
                "pronunciation": {"accuracy": 0, "word_pronunciation_scores": [], "feedback": skipped_msg},
                "fluency": {"score": 0, "wpm": 0, "speed_status": "skipped"},
                "answer_evaluation": {"clarity": "", "structure": "", "relevance": "", "improved_answer": ""},
                "personalized_feedback": {"message": skipped_next_msg, "improvement_areas": [], "strengths": []},
                "overall_score": 0, "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"],
                "audio_url": follow_up_audio
            }
        
        if action == "end":
            return await handle_session_termination(session, session_id, model)
        
        if not audio_file and not text_input:
            
            current_q = session.get("current_question")
            current_h = session.get("current_hint", "")
            
            if current_q:
                
                q_native = await translate_text(current_q, target_language, native_language)
                h_native = await translate_text(current_h, target_language, native_language)
                
                current_q_audio = await generate_tts_url(request, current_q, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "turn_number": session.get("turn_number", 0),
                    "audio_url": current_q_audio
                }
            else:
                
                question, hint = await generate_interview_question(
                    scenario, role, session.get("level", level), name, user_type=user_type, model=model, target_language=target_language, turn_number=session.get("turn_number", 0)
                )
                session["current_question"] = question
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": question})
                await db.update_session(session_id, session)
                
                q_native = await translate_text(question, target_language, native_language)
                h_native = await translate_text(hint, target_language, native_language)
                
                question_audio = await generate_tts_url(request, question, target_language, api_type="interview", voice_id=voice_id)
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": question, "native": q_native},
                    "hint": {"target": hint, "native": h_native},
                    "turn_number": session.get("turn_number", 0),
                    "audio_url": question_audio
                }
        
        user_text = text_input or ""
        audio_path = None
        audio_duration = 5.0
        is_audio_input = audio_file is not None  
        
        if audio_file:
            try:
                audio_file.file.seek(0)
            except Exception:
                pass
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                shutil.copyfileobj(audio_file.file, tmp)
                temp_upload = tmp.name
            
            try:
                
                def convert_audio():
                    audio = AudioSegment.from_file(temp_upload)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    converted_path = temp_upload.replace('.tmp', '_converted.wav')
                    audio.export(converted_path, format="wav")
                    return converted_path, len(audio) / 1000
                
                audio_path, audio_duration = await asyncio.to_thread(convert_audio)
                os.unlink(temp_upload)  
            except Exception as e:
                logger.debug(f"Audio conversion fallback: {e}")
                audio_path = temp_upload
            finally:
                
                
                if audio_path != temp_upload and os.path.exists(temp_upload):
                    try:
                        os.unlink(temp_upload)
                    except:
                        pass
        
        if is_audio_input:
            pronunciation = await analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=session.get("level", level), model=model, target_language=target_language)
            
            if pronunciation and pronunciation.get("transcription"):
                user_text = pronunciation["transcription"]
        else:
            pronunciation = None   
        
        if audio_path:
            try:
                os.unlink(audio_path)
            except Exception:
                pass
        
        if not user_text or not user_text.strip():
            error_msg = await translate_text("No speech detected. Please try again.", "en", native_language)
            return {"status": "error", "session_id": session_id, "error": error_msg}
        
        user_text = user_text.strip()
        session["chat_history"].append({"role": "user", "content": user_text})
        
        if session.get("waiting_retry_decision"):
            user_choice = user_text.lower().strip()
            
            
            cleaned_choice = user_choice.rstrip('.,!?')
            if cleaned_choice in TERMINATION_PHRASES:
                
                session["waiting_retry_decision"] = False
                return await handle_session_termination(session, session_id, model)
            
            is_english = target_language.lower() in ["en", "english"]
            if is_english:
                retry_keywords = ["yes", "retry", "practice", "again", "try", "redo", "repeat", "once more", "one more"]
                skip_keywords = ["no", "skip", "next", "move", "forward", "pass", "don't want", "not now", "let's move", "move on", "go ahead"]
                wants_retry = any(keyword in user_choice for keyword in retry_keywords)
                wants_skip = any(keyword in user_choice for keyword in skip_keywords)
            else:
                cleaned_choice = re.sub(r"[\s\W_]+", "", user_choice)
                wants_retry = cleaned_choice == "1"
                wants_skip = cleaned_choice == "2"
            
            if wants_retry:
                
                session["waiting_retry_decision"] = False  
                session["retry_clarify_count"] = 0  
                current_q = session.get("current_question", "")
                current_h = session.get("current_hint", "")
                session["chat_history"].append({"role": "assistant", "content": current_q})
                await db.update_session(session_id, session)
                
                retry_msg = "Let's try this again! Take your time."
                q_native, h_native, retry_msg_target, retry_msg_native = await asyncio.gather(
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_text(retry_msg, "en", target_language),
                    translate_text(retry_msg, "en", native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "message": {"target": retry_msg_target, "native": retry_msg_native},
                    "turn_number": session.get("turn_number", 0)
                }
            elif wants_skip:
                
                session["waiting_retry_decision"] = False  
                session["retry_clarify_count"] = 0  
                follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, user_type=user_type, model=model, target_language=target_language)
                session["current_question"] = follow_up
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": follow_up})
                session["retry_count"] = 0
                
                await db.update_session(session_id, session)
                
                follow_up_native, hint_native = await asyncio.gather(
                    translate_text(follow_up, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "next_question": {"target": follow_up, "native": follow_up_native},
                    "hint": {"target": hint, "native": hint_native},
                    "turn_number": session["turn_number"]
                }
            else:
                
                clarify_count = session.get("retry_clarify_count", 0) + 1
                session["retry_clarify_count"] = clarify_count
                
                
                if clarify_count >= 3:
                    session["waiting_retry_decision"] = False
                    session["retry_clarify_count"] = 0
                    
                    auto_skip_msg = "I see you're having trouble deciding. Let's move on to the next question!"
                    follow_up, hint = await generate_interactive_follow_up("", session["chat_history"], role, scenario, user_type=user_type, model=model, target_language=target_language)
                    session["current_question"] = follow_up
                    session["current_hint"] = hint
                    session["chat_history"].append({"role": "assistant", "content": auto_skip_msg})
                    session["chat_history"].append({"role": "assistant", "content": follow_up})
                    session["retry_count"] = 0
                    
                    await db.update_session(session_id, session)
                    
                    auto_skip_target, auto_skip_native, follow_up_native, hint_native = await asyncio.gather(
                        translate_text(auto_skip_msg, "en", target_language),
                        translate_text(auto_skip_msg, "en", native_language),
                        translate_text(follow_up, target_language, native_language),
                        translate_text(hint, target_language, native_language)
                    )
                    
                    return {
                        "status": "auto_skipped",
                        "session_id": session_id,
                        "target_lang": target_language,
                        "native_lang": native_language,
                        "message": {"target": auto_skip_target, "native": auto_skip_native},
                        "next_question": {"target": follow_up, "native": follow_up_native},
                        "hint": {"target": hint, "native": hint_native},
                        "turn_number": session["turn_number"],
                        
                        "grammar": None,
                        "vocabulary": None,
                        "pronunciation": None,
                        "fluency": None,
                        "answer_evaluation": None,
                        "emotion": None,
                        "personalized_feedback": None,
                        "overall_score": None,
                        "improvement": None
                    }
                else:
                    
                    current_q = session.get("current_question", "")
                    current_h = session.get("current_hint", "")
                    level = session.get("level", "B1")
                    
                    if is_audio_input:
                        word_count = len(user_text.split())
                        estimated_duration = max(1, word_count / 2.5)  
                        grammar, vocabulary, answer_eval, pronunciation, fluency = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, model=model, target_language=target_language),
                            analyze_vocab_llm(user_text, level=level, model=model, target_language=target_language),
                            evaluate_answer(current_q, user_text, level, model=model, target_language=target_language),
                            analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=level, model=model, target_language=target_language),
                            analyze_fluency_metrics(user_text, estimated_duration)
                        )
                    else:
                        
                        grammar, vocabulary, answer_eval = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, model=model, target_language=target_language),
                            analyze_vocab_llm(user_text, level=level, model=model, target_language=target_language),
                            evaluate_answer(current_q, user_text, level, model=model, target_language=target_language)
                        )
                        pronunciation = None
                        fluency = None
                    
                    
                    if is_audio_input:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": pronunciation.get("score", pronunciation.get("accuracy", 70)) if pronunciation else 0,
                            "fluency": fluency.get("score", 70) if fluency else 0,
                            "answer_evaluation": answer_eval.get("score", 50)
                        }
                        
                        overall_score = int(
                            scores["grammar"] * 0.25 +
                            scores["vocabulary"] * 0.25 +
                            scores["answer_evaluation"] * 0.25 +
                            scores["pronunciation"] * 0.15 +
                            scores["fluency"] * 0.10
                        )
                    else:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": None,
                            "fluency": None,
                            "answer_evaluation": answer_eval.get("score", 50)
                        }
                        
                        overall_score = int(
                            scores["grammar"] * 0.33 +
                            scores["vocabulary"] * 0.33 +
                            scores["answer_evaluation"] * 0.34
                        )

                    
                    
                    emotion = {"emotion": "neutral", "confidence_level": "medium", "explanation": ""}
                    personalized_feedback = await generate_personalized_feedback(
                        overall_score, scores, emotion, session.get("name", "User"),
                        grammar=grammar, vocabulary=vocabulary, 
                        pronunciation=pronunciation, answer_eval=answer_eval, model=model,
                        target_language=target_language
                    )
                    
                    
                    if clarify_count == 1:
                is_english = target_language.lower() in ["en", "english"]
                if is_english:
                    clarify_msg = "I heard you say something, but I'm not sure if you want to practice again or move on. Just say 'retry' or 'skip' - or you can try answering the question again!"
                else:
                    clarify_msg = "I heard you say something, but I'm not sure if you want to practice again or move on. Type 1 to retry, 2 to skip."
                    else:
                        clarify_msg = "Still not quite sure what you'd like to do. Say 'yes' to practice the same question, or 'skip' to get a new one. One more unclear response and I'll move you to the next question."
                    
                    
                    await db.update_session(session_id, session)
                    
                    
                    if is_audio_input and pronunciation and fluency:
                        (clarify_target, clarify_native, q_native, h_native, grammar_t, vocab_t, 
                         pron_t, fluency_t, eval_t, personal_t) = await asyncio.gather(
                            translate_text(clarify_msg, "en", target_language),
                            translate_text(clarify_msg, "en", native_language),
                            translate_text(current_q, target_language, native_language),
                            translate_text(current_h, target_language, native_language),
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                            translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS),
                            translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                        )
                    else:
                        (clarify_target, clarify_native, q_native, h_native, grammar_t, vocab_t, 
                         eval_t, personal_t) = await asyncio.gather(
                            translate_text(clarify_msg, "en", target_language),
                            translate_text(clarify_msg, "en", native_language),
                            translate_text(current_q, target_language, native_language),
                            translate_text(current_h, target_language, native_language),
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                        )
                        pron_t = None
                        fluency_t = None
                    
                    return {
                        "status": "clarify_retry",
                        "session_id": session_id,
                        "target_lang": target_language,
                        "native_lang": native_language,
                        "transcription": user_text,
                        "message": {"target": clarify_target, "native": clarify_native},
                        "next_question": {"target": current_q, "native": q_native},
                        "hint": {"target": current_h, "native": h_native},
                        "grammar": grammar_t,
                        "vocabulary": vocab_t,
                        "pronunciation": pron_t,
                        "fluency": fluency_t,
                        "answer_evaluation": eval_t,
                        "emotion": emotion,
                        "personalized_feedback": personal_t,
                        "overall_score": overall_score,
                        "clarify_count": clarify_count,
                        "turn_number": session.get("turn_number", 0)
                    }

        
        
        cleaned_text = user_text.lower().strip().rstrip('.,!?')
        is_termination = cleaned_text in TERMINATION_PHRASES or action == "end"
        
        grammar, vocabulary, answer_eval = await asyncio.gather(
            analyze_grammar_llm(user_text, level=level, model=model, target_language=target_language),
            analyze_vocab_llm(user_text, level=level, model=model, target_language=target_language),
            evaluate_answer(session.get("current_question", ""), user_text, level, model=model, target_language=target_language)
        )
        
        
        emotion = {"emotion": "neutral", "confidence_level": "medium", "explanation": ""}
        
        
        if is_audio_input:
            word_count = len(user_text.split())
            fluency = calculate_fluency(word_count, audio_duration)
        else:
            fluency = None  

        
        
        
        
        if not is_termination and session.get("current_question"):
            relevance_check = await check_answer_relevance(session["current_question"], user_text, model=model, target_language=target_language)
            
            if not relevance_check.get("relevant", True):
                
                redirect_msg = relevance_check.get("redirect")
                if not redirect_msg:
                    redirect_msg = "Let's stay on track! ????"
                    if target_language != "en":
                        redirect_msg = await translate_text(redirect_msg, "en", target_language)
                current_q = session["current_question"]
                current_h = session.get("current_hint", "")
                
                
                full_response = f"{redirect_msg}\n\n{current_q}"
                session["chat_history"].append({"role": "assistant", "content": full_response})
                await db.update_session(session_id, session)
                
                redirect_native, q_native, h_native, grammar_t, vocab_t, eval_t = await asyncio.gather(
                    translate_text(redirect_msg, target_language, native_language),
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                    translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                    translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
                )
                
                
                if is_audio_input and pronunciation and fluency:
                    pron_t, fluency_t = await asyncio.gather(
                        translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                        translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS)
                    )
                    scores = {
                        "grammar": grammar.get("score", 75),
                        "vocabulary": vocabulary.get("score", 75),
                        "pronunciation": pronunciation.get("accuracy", 75),
                        "fluency": fluency.get("score", 75)
                    }
                    answer_score = answer_eval.get("score", 50)
                    overall_score = int(
                        scores["grammar"] * 0.25 +
                        scores["vocabulary"] * 0.25 +
                        answer_score * 0.25 +
                        scores["pronunciation"] * 0.15 +
                        scores["fluency"] * 0.10
                    )
                else:
                    pron_t = None
                    fluency_t = None
                    scores = {
                        "grammar": grammar.get("score", 75),
                        "vocabulary": vocabulary.get("score", 75),
                        "pronunciation": None,
                        "fluency": None
                    }
                    answer_score = answer_eval.get("score", 50)
                    
                    overall_score = int(
                        scores["grammar"] * 0.33 +
                        scores["vocabulary"] * 0.33 +
                        answer_score * 0.34
                    )

                personalized_feedback = await generate_personalized_feedback(overall_score, scores, emotion, session["name"], model=model, target_language=target_language)
                
                
                personal_t = await translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
                
                return {
                    "status": "redirect",
                    "session_id": session_id,
                    "target_lang": target_language,
                    "native_lang": native_language,
                    "transcription": user_text,
                    "message": {"target": redirect_msg, "native": redirect_native},
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    
                    "grammar": grammar_t,
                    "vocabulary": vocab_t,
                    "pronunciation": pron_t,
                    "fluency": fluency_t,
                    "answer_evaluation": eval_t,
                    "emotion": emotion,
                    "personalized_feedback": personal_t,
                    "overall_score": overall_score,
                    "passing_score": PASSING_SCORE,
                    "improvement": None,  
                    "turn_number": session.get("turn_number", 0)
                }
        
        
        if is_termination:
            return await handle_session_termination(session, session_id, model)

        
        
        if is_audio_input:
            scores = {
                "grammar": grammar.get("score", 75),
                "vocabulary": vocabulary.get("score", 75),
                "pronunciation": pronunciation.get("accuracy", 75) if pronunciation else 0,
                "fluency": fluency.get("score", 75) if fluency else 0
            }
        else:
            scores = {
                "grammar": grammar.get("score", 75),
                "vocabulary": vocabulary.get("score", 75),
                "pronunciation": None,  
                "fluency": None  
            }
        
        
        answer_score = answer_eval.get("score", 50)
        if is_audio_input:
            
            overall_score = int(
                scores["grammar"] * 0.25 +
                scores["vocabulary"] * 0.25 +
                answer_score * 0.25 +
                scores["pronunciation"] * 0.15 +
                scores["fluency"] * 0.10
            )
        else:
            
            overall_score = int(
                scores["grammar"] * 0.33 +
                scores["vocabulary"] * 0.33 +
                answer_score * 0.34
            )
        
        
        session["scores"]["grammar"] += scores["grammar"]
        session["scores"]["vocabulary"] += scores["vocabulary"]
        if is_audio_input:
            session["scores"]["pronunciation"] += scores["pronunciation"]
            session["scores"]["fluency"] += scores["fluency"]
            session["scores"]["total_wpm"] += fluency.get("wpm", 100) if fluency else 100
            session["scores"]["audio_count"] = session["scores"].get("audio_count", 0) + 1  
        session["scores"]["answer"] = session["scores"].get("answer", 0) + answer_score  
        session["scores"]["count"] += 1

        session["turn_number"] += 1
        

        
        
        personalized_feedback, (follow_up_question, follow_up_hint) = await asyncio.gather(
            generate_personalized_feedback(
                overall_score, scores, emotion, session["name"],
                grammar=grammar, vocabulary=vocabulary, 
                pronunciation=pronunciation, answer_eval=answer_eval, model=model,
                target_language=target_language
            ),
            generate_interactive_follow_up(user_text, session["chat_history"], role, scenario, user_type=user_type, model=model, target_language=target_language)
        )
        
        
        improvement = {}
        is_retrying = session.get("retry_count", 0) > 0
        prev_overall = session.get("last_overall_score")
        
        
        if is_retrying and prev_overall is not None:
            
            current_attempt = {
                "transcription": user_text,
                "grammar": grammar,
                "vocabulary": vocabulary,
                "pronunciation": pronunciation,
                "fluency": fluency,
                "answer_evaluation": answer_eval,
                "overall_score": overall_score
            }
            session.setdefault("attempts", []).append(current_attempt)
            
            
            improvement = await compare_attempts(
                session["attempts"], 
                level="B1",  
                user_type=user_type, 
                model=model,
                target_language=target_language
            )
        else:
            
            current_attempt = {
                "transcription": user_text,
                "grammar": grammar,
                "vocabulary": vocabulary,
                "pronunciation": pronunciation,
                "fluency": fluency,
                "answer_evaluation": answer_eval,
                "overall_score": overall_score
            }
            session.setdefault("attempts", []).append(current_attempt)
        
        
        
        session["last_scores"] = scores.copy()
        session["last_overall_score"] = overall_score
        
        
        if "turn_history" not in session:
            session["turn_history"] = []
        
        turn_data = {
            "turn_number": session["turn_number"],
            "turn": session["turn_number"],  
            "transcription": user_text,
            "question": session.get("current_question", ""),
            "scores": scores.copy(),
            "overall_score": overall_score,
            "wpm": fluency.get("wpm", 0) if fluency else 0,  
            "grammar": grammar,
            "vocabulary": vocabulary,
            "pronunciation": pronunciation,
            "fluency": fluency,
            "answer_evaluation": answer_eval,
            "emotion": emotion,
            "personalized_feedback": personalized_feedback,
            "improvement": improvement
        }
        session["turn_history"].append(turn_data)
        
        
        should_retry = (overall_score < PASSING_SCORE or action == "practice")
        
        if should_retry:
            session["retry_count"] = session.get("retry_count", 0) + 1
            session["waiting_retry_decision"] = True  
            current_q = session.get("current_question", "")
            current_h = session.get("current_hint", "")
            
            
            retry_ask = "I see your answer, but it could be stronger. Type 1 to retry, 2 to skip."
            
            
            base_tasks = [
                translate_text(retry_ask, "en", target_language),
                translate_text(retry_ask, "en", native_language),
                translate_text(current_q, target_language, native_language),
                translate_text(current_h, target_language, native_language),
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
            ]
            base_results = await asyncio.gather(*base_tasks)
            retry_ask_target, retry_ask_native, q_native, h_native, grammar_t, vocab_t, eval_t = base_results
            
            
            pron_t = await translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS) if pronunciation else None
            fluency_t = await translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS) if fluency else None
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            retry_ask_audio = await generate_tts_url(request, retry_ask_target, target_language, api_type="interview")
            
            return {
                "status": "feedback",
                "session_id": session_id,
                "target_lang": target_language,
                "native_lang": native_language,
                "transcription": user_text,
                "message": {"target": retry_ask_target, "native": retry_ask_native},
                "next_question": {"target": current_q, "native": q_native},
                "hint": {"target": current_h, "native": h_native},
                "grammar": grammar_t,
                "vocabulary": vocab_t,
                "pronunciation": pron_t,
                "fluency": fluency_t,
                "answer_evaluation": eval_t,
                "emotion": emotion,
                "personalized_feedback": personalized_feedback,
                "overall_score": overall_score,
                "passing_score": PASSING_SCORE,
                "should_retry": True,
                "retry_count": session.get("retry_count", 1),
                "improvement": improvement,
                "turn_number": session["turn_number"],
                "audio_url": retry_ask_audio
            }
        else:
            
            session["current_question"] = follow_up_question
            session["current_hint"] = follow_up_hint
            session["chat_history"].append({"role": "assistant", "content": follow_up_question})
            session["retry_count"] = 0
            
            
            base_tasks = [
                translate_text(follow_up_question, target_language, native_language),
                translate_text(follow_up_hint, target_language, native_language),
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                translate_analysis(answer_eval, target_language, native_language, EVAL_FIELDS)
            ]
            base_results = await asyncio.gather(*base_tasks)
            follow_up_native, hint_native, grammar_t, vocab_t, personal_t, eval_t = base_results
            
            
            pron_t = await translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS) if pronunciation else None
            fluency_t = await translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS) if fluency else None
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            follow_up_audio = await generate_tts_url(request, follow_up_question, target_language, api_type="interview")
            
            return {
                "status": "continue", "session_id": session_id, 
                "target_lang": target_language, "native_lang": native_language,
                "transcription": user_text,
                "next_question": {"target": follow_up_question, "native": follow_up_native},
                "hint": {"target": follow_up_hint, "native": hint_native},
                "grammar": grammar_t, "vocabulary": vocab_t, "pronunciation": pron_t, "fluency": fluency_t,
                "answer_evaluation": eval_t, "emotion": emotion,
                "personalized_feedback": personal_t,
                "overall_score": overall_score, "passing_score": PASSING_SCORE,
                "improvement": improvement,  
                "should_retry": False, "turn_number": session["turn_number"],
                "audio_url": follow_up_audio
            }
    
    except Exception as e:
        logger.exception(f"Error in practice_interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# COMMENTED OUT - Admin/debug endpoint, use /roles_with_session_ids for user sessions
# @router.get("/sessions")
# async def list_sessions():
#     """list active interview sessions from database"""
#     sessions_list = await db.list_sessions(session_type="interview")
#     return {"active_sessions": len(sessions_list), "sessions": sessions_list}


@router.get("/sessions/{session_id}")
async def get_session_data(session_id: str):
    """get complete session history including all responses, feedback, and analysis"""
    session_data = await db.get_user_session(session_id)
    if session_data:
        
        count = max(1, session_data.get("scores", {}).get("count", 1))
        raw_scores = session_data.get("scores", {})
        audio_count = raw_scores.get("audio_count", 0)
        if not audio_count and (raw_scores.get("pronunciation", 0) > 0 or raw_scores.get("fluency", 0) > 0):
            audio_count = count
        
        average_scores = {
            "grammar": int(raw_scores.get("grammar", 0) / count),
            "vocabulary": int(raw_scores.get("vocabulary", 0) / count),
            "pronunciation": int(raw_scores.get("pronunciation", 0) / audio_count) if audio_count > 0 else None,
            "fluency": int(raw_scores.get("fluency", 0) / audio_count) if audio_count > 0 else None,
        }
        
        if audio_count > 0:
            overall_average = int(sum(v for v in average_scores.values() if v is not None) / 4)
        else:
            overall_average = int((average_scores["grammar"] + average_scores["vocabulary"]) / 2)
        average_wpm = int(raw_scores.get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
        
        
        session_status = session_data.get("status", "active")
        is_completed = session_status == "completed"
        
        response = {
            "status": "success",
            "session_id": session_id,
            "session_status": session_status,  
            "can_continue": not is_completed,  
            "user_name": session_data.get("name", ""),
            "scenario": session_data.get("scenario", ""),
            "role": session_data.get("role", ""),
            "level": session_data.get("level", ""),
            "current_state": session_data.get("state", "interviewing"),
            "turns_completed": session_data.get("turn_number", 0),
            "average_scores": average_scores,
            "overall_score": overall_average,
            "average_wpm": average_wpm,
            "last_score": session_data.get("last_overall_score"),
            "last_scores": session_data.get("last_scores", {}),

            "chat_history": session_data.get("chat_history", []),
            
            "turn_history": session_data.get("turn_history", []),
        }
        
        
        if is_completed and session_data.get("final_feedback"):
            response["final_feedback"] = session_data["final_feedback"]
        
        return response
    
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/scenarios")
async def get_scenarios():
    """get available interview scenarios"""
    return {"scenarios": INTERVIEW_SCENARIOS}


# COMMENTED OUT - Use /final_feedback/sessions/{session_id} instead (returns exact same data)
# @router.get("/final_feedback/{session_id}")
# async def get_interview_feedback(session_id: str):
#     """
#     Get the exact same response as session termination.
#     Returns the stored final_feedback from DB with keys in the same order as termination response.
#     """
#     session = await db.get_user_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
#     if session.get("status") != "completed":
#         raise HTTPException(status_code=400, detail="Session not completed yet")
#     final_feedback = session.get("final_feedback")
#     if not final_feedback:
#         raise HTTPException(status_code=404, detail="Final feedback not found")
#     
#     ordered_response = {
#         "status": final_feedback.get("status", "conversation_ended"),
#         "session_id": final_feedback.get("session_id", session_id),
#         "target_lang": final_feedback.get("target_lang", session.get("target_language", "en")),
#         "native_lang": final_feedback.get("native_lang", session.get("native_language", "hi")),
#         "final_scores": final_feedback.get("final_scores", {}),
#         "overall_score": final_feedback.get("overall_score", 0),
#         "passing_score": final_feedback.get("passing_score", PASSING_SCORE),
#         "average_wpm": final_feedback.get("average_wpm", 0),
#         "wpm_per_turn": final_feedback.get("wpm_per_turn", []),
#         "wpm_status": final_feedback.get("wpm_status", "normal"),
#         "vocab_overall": final_feedback.get("vocab_overall", {}),
#         "strengths": final_feedback.get("strengths", []),
#         "improvement_areas": final_feedback.get("improvement_areas", []),
#         "total_turns": final_feedback.get("total_turns", 0),
#         "turn_history": final_feedback.get("turn_history", []),
#         "turn_feedback": final_feedback.get("turn_feedback", []),
#         "summary": final_feedback.get("summary", {}),
#         "overall_assessment": final_feedback.get("overall_assessment", ""),
#         "grammar_feedback": final_feedback.get("grammar_feedback", {}),
#         "vocabulary_feedback": final_feedback.get("vocabulary_feedback", {}),
#         "pronunciation_feedback": final_feedback.get("pronunciation_feedback", {}),
#         "fluency_feedback": final_feedback.get("fluency_feedback", {}),
#         "interview_skills": final_feedback.get("interview_skills", {}),
#         "action_plan": final_feedback.get("action_plan", []),
#         "encouragement": final_feedback.get("encouragement", ""),
#         "next_practice_topics": final_feedback.get("next_practice_topics", [])
#     }
#     
#     return ordered_response
# COMMENTED OUT - Use /roles_with_session_ids instead
# @router.get("/user_sessions")
# async def get_interview_sessions_by_user(
#     role: Optional[str] = None,
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get all Interview sessions for the authenticated user.
#     
#     Optionally filter by role (e.g., 'software', 'marketing', 'sales').
#     Returns sessions with session_ids included.
#     """
#     user_id = current_user.id
#     sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")
#     
#     
#     if role:
#         filtered_sessions = []
#         for session in sessions:
#             session_data = await db.get_user_session(session.get("session_id"))
#             if session_data and session_data.get("role") == role:
#                 session["role"] = role
#                 filtered_sessions.append(session)
#         sessions = filtered_sessions
#     else:
#         
#         for session in sessions:
#             session_data = await db.get_user_session(session.get("session_id"))
#             if session_data:
#                 session["role"] = session_data.get("role", "unknown")
#     
#     for idx, session in enumerate(sessions, 1):
#         session["session_number"] = f"Session {idx}"
#     
#     session_ids = [s.get("session_id") for s in sessions]
#     
#     return {
#         "user_id": user_id,
#         "total_sessions": len(sessions),
#         "filter": {"role": role} if role else None,
#         "session_ids": session_ids,
#         "sessions": sessions
#     }


@router.get("/completed_sessions")
async def get_completed_interview_sessions(current_user: User = Depends(get_current_user)):
    """
    Get only completed interview sessions for the authenticated user.
    Returns session_ids and session metadata for completed sessions.
    """
    user_id = current_user.id
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")
    completed_sessions = []
    for s in sessions:
        session_data = await db.get_user_session(s.get("session_id"))
        if not session_data:
            continue
        if session_data.get("status") != "completed":
            continue
        if not session_data.get("final_feedback"):
            continue
        completed_sessions.append({
            "session_id": s.get("session_id"),
            "created_at": s.get("created_at"),
            "role": session_data.get("role", ""),
            "scenario": session_data.get("scenario", ""),
            "target_lang": session_data.get("target_language", "en"),
            "native_lang": session_data.get("native_language", "hi")
        })
    return {
        "status": "success",
        "total_sessions": len(completed_sessions),
        "session_ids": [s.get("session_id") for s in completed_sessions],
        "sessions": completed_sessions
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

@router.get("/roles_with_session_ids")
async def get_roles_and_session_ids(current_user: User = Depends(get_current_user)):
    """
    Get all roles, their corresponding session IDs, and the total session counts for each role for the current user.
    """
    user_id = current_user.id if current_user else None

    # Get distinct roles
    roles = await db.get_distinct_roles_by_user(user_id, session_type="interview")

    # List to hold roles with session info
    roles_with_session_ids = []

    # Get all sessions for the user once
    sessions = await db.get_sessions_by_user_id(user_id, session_type="interview")

    for role in roles:
        session_ids_for_role = []

        for s in sessions:
            session_data = await db.get_user_session(s.get("session_id"))
            if session_data and session_data.get("role") == role and session_data.get("status") == "completed":
                session_ids_for_role.append(s.get("session_id"))

        if session_ids_for_role:
            roles_with_session_ids.append({
                "role": role,
                "session_ids": session_ids_for_role,
                "total_sessions": len(session_ids_for_role)
            })

    return {
        "status": "success",
        "user_id": user_id,
        "total_roles": len(roles_with_session_ids),
        "roles_with_session_ids": roles_with_session_ids
    }




@router.get("/final_feedback/sessions/{session_id}")
async def get_interview_feedback_sessions(session_id: str):
    """
    Get the exact same response as session termination.
    Simply returns the stored final_feedback from DB - exactly as it was when session ended.
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
