 



QWEN_ENABLED = False
qwen_client = None  
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"  


_whisper_model = WhisperModel("small", device="cpu", compute_type="int8")


 
PASSING_SCORE = 50
MAX_ATTEMPTS = 3  
SENTENCES_PER_WORD_NORMAL = 3   
SENTENCES_PER_WORD_STRICT = 5   
DEFAULT_NUM_WORDS = 5           


# import asyncio
# from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_NAME = "Qwen/Qwen3-0.6B"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype="auto",
#     device_map="auto"
# )
# import asyncio
# import torch

# async def call_llm(prompt: str, mode: str = "chat", timeout: int = 30) -> str:
#     system_prompts = {
#         "chat": f"You are {BOT_NAME}, a warm and helpful FAQ practice coach.",
#         "analysis": "You are an expert language and interview evaluator. Analyze objectively.",
#         "strict_json": "You are a structured evaluator. Respond ONLY in valid JSON. No extra text."
#     }

#     def run():
#         messages = [
#             {"role": "system", "content": system_prompts.get(mode, system_prompts["chat"])},
#             {"role": "user", "content": prompt}
#         ]

#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = tokenizer(text, return_tensors="pt").to(model.device)

#         with torch.no_grad():
#             output = model.generate(
#                 **inputs,
#                 max_new_tokens=1000,
#                 temperature=0.7 if mode == "chat" else 0.3
#             )

#         # Decode only new tokens
#         return tokenizer.decode(
#             output[0][inputs.input_ids.shape[1]:],
#             skip_special_tokens=True
#         ).strip()

#     try:
#         return await asyncio.wait_for(asyncio.to_thread(run), timeout)
#     except Exception as e:
#         logger.error(f"LLM call failed: {e}")
#         return ""

# ===============================

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
        "audio_duration_seconds": round(audio_duration, 1),
        "feedback": f"Your speaking speed is {speed_status.replace('_', ' ')} ({wpm} WPM)."
    }


async def analyze_fluency_metrics(user_text: str, audio_duration: float) -> dict:
    """async wrapper for fluency metrics from text and duration"""
    
    word_count = len(re.findall(r"\b\w+\b", user_text or ""))
    return calculate_fluency(word_count, audio_duration)






def call_gpt_sync(prompt: str, system_prompt: str = None, model: str = "gpt") -> str:
    """sync gpt call - will be run in thread pool. Supports gpt (default) or qwen."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": "You are a friendly language tutor. Return only valid JSON when asked."})
    messages.append({"role": "user", "content": prompt})
    
    
    if model.lower() == "qwen" and QWEN_ENABLED and qwen_client is not None:
        try:
            response = qwen_client.chat.completions.create(
                model=QWEN_MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Qwen call failed, falling back to GPT: {e}")
            
    
    
    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


async def call_llm(prompt: str, system_prompt: str = None, timeout: int = 30, model: str = "gpt") -> str:
    """async llm call with timeout. Supports gpt (default) or qwen."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(call_gpt_sync, prompt, system_prompt, model),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError as e:
        logger.error(f"llm timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"llm error: {e}")
        raise






def safe_json_loads(text: str) -> dict:
    """extract and parse json from llm response"""
    if not text:
        raise ValueError("empty response")
    
    
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    
    
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    
    raise ValueError(f"no json found in: {text[:100]}")


def extract_json_array(text: str) -> list:
    """extract json array from llm response"""
    if not text:
        return []
    
    text = text.strip().replace("```json", "").replace("```", "").strip()
    
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    
    return []






async def transcribe_audio(audio_path: str, target_lang: str = "en") -> str:
    """transcribe audio using whisper (pre-loaded model)"""
    try:
        # Normalize language code (e.g., "hindi" -> "hi")
        target_lang = target_lang.lower().strip()
        languages_data = load_language_mapping()
        if target_lang in languages_data:
            target_lang = languages_data.get(target_lang, "en")
        
        segments, info = await asyncio.to_thread(_whisper_model.transcribe, audio_path, language=target_lang)
        text = " ".join([seg.text for seg in segments])
        return text.lower().strip()
    except Exception as e:
        logger.error(f"transcription error: {e}")
        return ""


async def transcribe_audio_with_words(audio_path: str, target_lang: str = "en") -> dict:
    """transcribe with word-level timestamps and confidence (pre-loaded model)"""
    try:
        # Normalize language code (e.g., "hindi" -> "hi")
        target_lang = target_lang.lower().strip()
        languages_data = load_language_mapping()
        if target_lang in languages_data:
            target_lang = languages_data.get(target_lang, "en")
        
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, 
            audio_path,
            language=target_lang,
            word_timestamps=True
        )
        
        words = []
        full_text = []
        
        for segment in segments:
            full_text.append(segment.text)
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
        
        return {
            "text": " ".join(full_text).lower().strip(),
            "words": words,
            "duration": info.duration if info else 0
        }
    except Exception as e:
        logger.error(f"transcription with words error: {e}")
        return {"text": "", "words": [], "duration": 0}






def pronunciation_score_wer(expected: str, spoken: str) -> int:
    """WER-based pronunciation score for SENTENCES - compares full text match"""
    try:
        error = wer(expected.lower().strip(), spoken.lower().strip())
        return max(0, 100 - int(error * 100))
    except Exception as e:
        logger.error(f"WER scoring error: {e}")
        return 0


def pronunciation_score_word(expected_word: str, transcription: str) -> dict:
    """
    Word-based pronunciation score - STRICT matching.
    Returns 0 for total mismatch, score only if word is detected or very close.
    Note: All return values are native Python types for JSON serialization.
    """
    expected_lower = expected_word.lower().strip()
    transcription_lower = transcription.lower().strip()
    
    # Exact match - word is in transcription
    if expected_lower in transcription_lower:
        return {"score": 100, "detected": True, "match_type": "exact"}
    
    # Check each spoken word for very close match only
    transcription_words = transcription_lower.split()
    for spoken_word in transcription_words:
        # Only allow if VERY similar (>80% character match) - stricter threshold
        if len(spoken_word) > 0 and len(expected_lower) > 0:
            common = set(spoken_word) & set(expected_lower)
            similarity = len(common) / max(len(set(spoken_word)), len(set(expected_lower)))
            # Also check length is similar (prevents "cat" matching "beautiful")
            length_ratio = min(len(spoken_word), len(expected_lower)) / max(len(spoken_word), len(expected_lower))
            # Use bool() to ensure native Python boolean (not numpy.bool_)
            if bool(similarity >= 0.8) and bool(length_ratio >= 0.7):
                return {"score": int(similarity * 85), "detected": True, "match_type": "partial"}
    
    # Total mismatch - return 0
    return {"score": 0, "detected": False, "match_type": "none"}


async def analyze_word_pronunciation(audio_path: str, expected_word: str, target_lang: str = "en") -> dict:
    """
    Analyze single word pronunciation using Whisper confidence.
    Returns confidence score and syllable guide if needed.
    """
    try:
        
        transcription_data = await transcribe_audio_with_words(audio_path, target_lang)
        
        words = transcription_data.get("words", [])
        full_transcription = transcription_data.get("text", "").strip()
        
        
        word_match = pronunciation_score_word(expected_word, full_transcription)
        
        
        word_confidence = 0.0
        matched_word = None
        for wd in words:
            if expected_word.lower() in wd["word"].lower() or wd["word"].lower() in expected_word.lower():
                word_confidence = wd.get("probability", 0.7)
                matched_word = wd["word"]
                break
        
        
        if not matched_word and words:
            word_confidence = sum(w.get("probability", 0.7) for w in words) / len(words)
        
        
        confidence_score = int(word_confidence * 100)
        
        
        if word_match["detected"]:
            final_score = max(confidence_score, word_match["score"])
        else:
            final_score = 0  # Wrong word = zero score
        
        return {
            "score": int(final_score),
            "confidence": float(round(word_confidence * 100, 1)),
            "transcription": full_transcription,
            "expected": expected_word,
            "detected": bool(word_match["detected"]),
            "match_type": word_match["match_type"],
            "words_data": words,
            "needs_practice": bool(final_score < PASSING_SCORE or word_confidence < 0.70)
        }
        
    except Exception as e:
        logger.error(f"Word pronunciation analysis error: {e}")
        return {
            "score": 0,
            "confidence": 0,
            "transcription": "",
            "expected": expected_word,
            "detected": False,
            "match_type": "error",
            "words_data": [],
            "needs_practice": True
        }


async def generate_syllable_guide(word: str, model: str = "gpt") -> dict:
    """Generate detailed pronunciation guide with phoneme-level breakdown"""
    prompt = f"""You are an expert pronunciation coach. Provide a DETAILED pronunciation guide for the word: "{word}"

Return STRICTLY valid JSON:
{{
    "word": "{word}",
    "syllables": "break-down with hy-phens (e.g., break-fast)",
    "syllable_count": <number of syllables>,
    "phonetic_ipa": "IPA transcription (e.g., /ˈbrekfəst/)",
    "phonetic_simple": "simple phonetic guide (e.g., BREK-fuhst)",
    "stress_pattern": "which syllable to stress (e.g., FIRST syllable)",
    
    "phoneme_breakdown": [
        {{
            "syllable": "break",
            "sounds": [
                {{"letter": "b", "sound": "/b/", "how_to_say": "Press lips together, release with voice"}},
                {{"letter": "r", "sound": "/r/", "how_to_say": "Curl tongue back slightly"}},
                {{"letter": "ea", "sound": "/e/", "how_to_say": "Short 'e' as in 'bed'"}},
                {{"letter": "k", "sound": "/k/", "how_to_say": "Back of tongue touches soft palate"}}
            ]
        }},
        {{
            "syllable": "fast",
            "sounds": [
                {{"letter": "f", "sound": "/f/", "how_to_say": "Upper teeth on lower lip, blow air"}},
                {{"letter": "a", "sound": "/ə/", "how_to_say": "Unstressed 'uh' sound"}},
                {{"letter": "st", "sound": "/st/", "how_to_say": "'s' followed by 't'"}}
            ]
        }}
    ],
    
    "mouth_position": "Describe how to position mouth/tongue for key sounds",
    "common_mistakes": ["list", "of", "common", "mistakes"],
    "practice_tip": "One specific tip for practicing this word",
    "similar_words": ["words", "with", "similar", "sounds"]
}}"""
    
    try:
        raw = await call_llm(prompt, timeout=15, model=model)
        data = safe_json_loads(raw)
        
        data.setdefault("word", word)
        data.setdefault("syllables", word)
        data.setdefault("phoneme_breakdown", [])
        return data
    except Exception as e:
        logger.error(f"Syllable guide generation error: {e}")
        return {
            "word": word,
            "syllables": word,
            "syllable_count": 1,
            "phonetic_ipa": f"/{word}/",
            "phonetic_simple": word.upper(),
            "stress_pattern": "first syllable",
            "phoneme_breakdown": [],
            "mouth_position": "Speak slowly and clearly",
            "common_mistakes": [],
            "practice_tip": "Break the word into smaller parts",
            "similar_words": []
        }


async def analyze_sentence_pronunciation(audio_path: str, expected_sentence: str, spoken_text: str, target_lang: str = "en") -> dict:
    """
    Analyze sentence pronunciation using WER and word-level analysis.
    Returns WER score, mismatches, and improvement suggestions.
    """
    try:
        
        transcription_data = await transcribe_audio_with_words(audio_path, target_lang)
        words_data = transcription_data.get("words", [])
        duration = transcription_data.get("duration", 5)
        
        
        wer_score = pronunciation_score_wer(expected_sentence, spoken_text)
        
        
        expected_words = expected_sentence.lower().strip().split()
        spoken_words = spoken_text.lower().strip().split()
        
        mismatches = []
        mispronounced = []
        well_pronounced = []
        
        
        for i, expected_word in enumerate(expected_words):
            if i < len(spoken_words):
                spoken_word = spoken_words[i]
                if expected_word != spoken_word:
                    mismatches.append({
                        "position": i + 1,
                        "expected": expected_word,
                        "spoken": spoken_word,
                        "issue": "word mismatch"
                    })
            else:
                mismatches.append({
                    "position": i + 1,
                    "expected": expected_word,
                    "spoken": "(missing)",
                    "issue": "word missing"
                })
        
        
        if len(spoken_words) > len(expected_words):
            for i in range(len(expected_words), len(spoken_words)):
                mismatches.append({
                    "position": i + 1,
                    "expected": "(none)",
                    "spoken": spoken_words[i],
                    "issue": "extra word"
                })
        
        
        for wd in words_data:
            word = wd.get("word", "").strip().lower()
            confidence = wd.get("probability", 1.0)
            
            if confidence < 0.7:
                mispronounced.append({
                    "word": word,
                    "confidence": round(confidence * 100, 1),
                    "issue": "unclear pronunciation" if confidence < 0.5 else "needs improvement"
                })
            else:
                well_pronounced.append(word)
        
        
        fluency = await analyze_fluency_metrics(spoken_text, duration)
        
        return {
            "score": wer_score,
            "transcription": spoken_text,
            "expected": expected_sentence,
            "mismatches": mismatches,
            "mismatch_count": len(mismatches),
            "mispronounced_words": mispronounced,
            "well_pronounced_words": well_pronounced[:10],
            "fluency": fluency,
            "accuracy_percentage": wer_score
        }
        
    except Exception as e:
        logger.error(f"Sentence pronunciation analysis error: {e}")
        return {
            "score": 0,
            "transcription": spoken_text,
            "expected": expected_sentence,
            "mismatches": [],
            "mismatch_count": 0,
            "mispronounced_words": [],
            "well_pronounced_words": [],
            "fluency": {"wpm": 0, "speed_status": "unknown", "duration": 0},
            "accuracy_percentage": 0
        }






async def translate_text(text: str, source: str, target: str) -> str:
    """translate text between languages - handles full names like 'hindi' to 'hi'"""
    if not text or not isinstance(text, str):
        return text if isinstance(text, str) else ""
    
    # Convert full language names to codes using JSON mapping
    source = source.lower().strip()
    target = target.lower().strip()
    
    languages_data = load_language_mapping()
    if source in languages_data:
        source = languages_data.get(source, source)
    if target in languages_data:
        target = languages_data.get(target, target)

    if source == target:
        return text
        
    try:
        translated = await asyncio.to_thread(
            GoogleTranslator(source=source, target=target).translate,
            text
        )
        return translated
    except Exception as e:
        logger.error(f"translation error: {e}")
        return text







async def generate_lesson_llm(topic: str, num_words: int, target_lang: str, model: str = "gpt") -> list:
    """Generate lesson using LLM for normal mode - creates 3 sentences per word"""
    
    
    lang_names = {
        "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
        "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
        "pt": "Portuguese", "ru": "Russian", "it": "Italian", "en": "English"
    }
    target_lang_name = lang_names.get(target_lang, target_lang)
    
    prompt = f"""You are a language tutor. Create exactly {num_words} {target_lang_name} vocabulary items for a lesson on '{topic}'.

IMPORTANT RULES:
- The 'word' must be in {target_lang_name}
- 'meaning_en' must be the DEFINITION IN ENGLISH that explains what the word means (example: for "breakfast", meaning_en should be "the first meal of the day, eaten in the morning")
- NEVER use the word itself as meaning_en - always explain what it means!
- 'meaning_{target_lang}' must be the meaning/definition in {target_lang_name}
- All sentences must be in {target_lang_name} with English translations

For EACH word, provide:
- word: the {target_lang_name} vocabulary word
- meaning_en: English definition (NOT the word itself, but a clear explanation of what it means)
- meaning_{target_lang}: meaning/definition in {target_lang_name}
- sentences: array of {SENTENCES_PER_WORD_NORMAL} sentences containing the word

Each sentence must have:
- {target_lang}: {target_lang_name} sentence (MUST contain the word)
- en: English translation

Return ONLY valid JSON array:
[{{"word": "<target word>", "meaning_en": "English meaning", "meaning_{target_lang}": "<target meaning>", "sentences": [{{"{target_lang}": "<target sentence>", "en": "<English translation>"}}]}}]

NO markdown. NO code fences.
"""
    
    try:
        raw = await call_llm(prompt, model=model)
        lesson = extract_json_array(raw)
        
        if not lesson:
            logger.warning("lesson generation returned empty, using fallback")
            return await generate_fallback_lesson(topic, num_words, target_lang)
        
        
        valid_items = []
        
        for item in lesson:
            word = item.get("word", "").lower()
            
            
            if "sentences" not in item or not item["sentences"]:
                
                if "sentence" in item:
                    item["sentences"] = [
                        {"en": item["sentence"], target_lang: item.get(f"sentence_{target_lang}", item["sentence"])}
                    ]
                else:
                    item["sentences"] = []
            
            
            valid_sentences = []
            for sent in item.get("sentences", []):
                if isinstance(sent, str):
                    sent = {"en": sent}
                en_sent = sent.get("en", "")
                target_sent = sent.get(target_lang, "")
                if not target_sent and en_sent:
                    target_sent = en_sent if target_lang == "en" else await translate_text(en_sent, "en", target_lang)
                    sent[target_lang] = target_sent
                if not en_sent and target_sent:
                    en_sent = target_sent if target_lang == "en" else await translate_text(target_sent, target_lang, "en")
                    sent["en"] = en_sent
                if word and target_sent and word in target_sent.lower():
                    valid_sentences.append(sent)
                else:
                    if target_sent:
                        sent[target_lang] = f"{target_sent.rstrip('.')}. {word}"
                    else:
                        sent[target_lang] = word
                    if not sent.get("en"):
                        sent["en"] = f"I use the word {word}."
                    valid_sentences.append(sent)
            
            
            while len(valid_sentences) < SENTENCES_PER_WORD_NORMAL:
                en_fallback = f"I use the word {word} often."
                target_fallback = en_fallback if target_lang == "en" else await translate_text(en_fallback, "en", target_lang)
                valid_sentences.append({
                    "en": en_fallback,
                    target_lang: target_fallback
                })
            
            item["sentences"] = valid_sentences[:SENTENCES_PER_WORD_NORMAL]
            item["meaning_en"] = item.get("meaning_en") or f"a word meaning {word}"
            if not item.get(f"meaning_{target_lang}"):
                if target_lang == "en":
                    item[f"meaning_{target_lang}"] = item["meaning_en"]
                else:
                    item[f"meaning_{target_lang}"] = await translate_text(item["meaning_en"], "en", target_lang)
            
            valid_items.append(item)
        
        return valid_items if valid_items else await generate_fallback_lesson(topic, num_words, target_lang)
        
    except Exception as e:
        logger.error(f"lesson generation error: {e}")
        return await generate_fallback_lesson(topic, num_words, target_lang)


async def generate_fallback_lesson(topic: str, num_words: int, target_lang: str) -> list:
    """Fallback lesson when LLM fails - creates 3 sentences per word"""
    basic_words = ["hello", "good", "learn", "speak", "practice", "today", "happy", "work", "friend", "time"][:num_words]
    lesson = []
    for word in basic_words:
        word_target = word if target_lang == "en" else await translate_text(word, "en", target_lang)
        meaning_en = "a common word"
        meaning_target = meaning_en if target_lang == "en" else await translate_text(meaning_en, "en", target_lang)
        sentence_templates = [
            f"I want to say {word_target}.",
            f"Let me practice {word_target} with you.",
            f"We should use {word_target} together."
        ]
        sentences = []
        for en_sent in sentence_templates:
            target_sent = en_sent if target_lang == "en" else await translate_text(en_sent, "en", target_lang)
            sentences.append({"en": en_sent, target_lang: target_sent})
        lesson.append({
            "word": word_target,
            "meaning_en": meaning_en,
            f"meaning_{target_lang}": meaning_target,
            "sentences": sentences
        })
    return lesson
async def make_bilingual(value, source: str, target: str):
    """Convert value to {target, native} structure"""
    if source == target:
        return value
    if isinstance(value, str):
        if not value.strip():
            return {"target": value, "native": value}
        native = await translate_text(value, source, target)
        return {"target": value, "native": native}
    elif isinstance(value, list):
        return await asyncio.gather(*[make_bilingual(item, source, target) for item in value])
    elif isinstance(value, dict):
        keys = list(value.keys())
        translated_values = await asyncio.gather(*[make_bilingual(value[k], source, target) for k in keys])
        return dict(zip(keys, translated_values))
    return value







async def load_vocab_file(set_number: int = None) -> list:
    """load vocabulary from database, optionally filtered by set"""
    return await db.get_pronunciation_vocab(set_number=set_number)



def sample_vocab(vocab: list, k: int) -> list:
    """randomly sample k words from vocab"""
    if len(vocab) < k:
        return vocab.copy()
    return random.sample(vocab, k)


async def generate_sentences_llm(word: str, target_lang: str, num_sentences: int = 3, model: str = "gpt") -> list:
    """generate sentences for a word using llm"""
    lang_names = {
        "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
        "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
        "pt": "Portuguese", "ru": "Russian", "it": "Italian", "en": "English"
    }
    target_lang_name = lang_names.get(target_lang, target_lang)
    prompt = f"""
Generate exactly {num_sentences} short spoken {target_lang_name} sentences using the word "{word}".
The {target_lang_name} sentence MUST contain the word exactly.
Provide an English translation for each sentence.

Return ONLY strict JSON in this format:
{{
  "sentences": [
    {{"{target_lang}": "...", "en": "..."}},
    {{"{target_lang}": "...", "en": "..."}},
    {{"{target_lang}": "...", "en": "..."}}
  ]
}}
"""
    
    retries = 2
    for attempt in range(retries + 1):
        try:
            raw = await call_llm(prompt, model=model)
            parsed = safe_json_loads(raw)
            
            if "sentences" in parsed and len(parsed["sentences"]) >= num_sentences:
                return parsed["sentences"][:num_sentences]
            
        except Exception as e:
            logger.error(f"sentence generation error for '{word}' (attempt {attempt+1}): {e}")
    
    
    fallback_en = [
        f"I use the word {word}.",
        f"This is an example with {word}.",
        f"{word} is easy to remember."
    ]
    sentences = []
    for en_sent in fallback_en:
        target_sent = en_sent if target_lang == "en" else await translate_text(en_sent, "en", target_lang)
        sentences.append({"en": en_sent, target_lang: target_sent})
    return sentences


def safe_get_sentence_text(sentence, key: str, fallback: str = "") -> str:
    """Safely get text from a sentence that could be a string or dict.
    This fixes the 'str' object has no attribute 'get' error."""
    if isinstance(sentence, str):
        return sentence
    elif isinstance(sentence, dict):
        return sentence.get(key, "") or sentence.get("en", "") or fallback
    return fallback


async def build_lesson_strict(target_lang: str, num_words: int = DEFAULT_NUM_WORDS, set_number: int = None, model: str = "gpt") -> list:
    """Build lesson from vocab file for strict mode - uses num_words from input, filtered by set if specified"""
    vocab = await load_vocab_file(set_number=set_number)
    
    if not vocab:
        error_msg = f"No vocabulary found in database for set {set_number}." if set_number else "No vocabulary found in database. Please run seed_tables.py first."
        raise HTTPException(status_code=500, detail=error_msg)

    selected = sample_vocab(vocab, num_words)
    
    async def normalize_sentence(s, target_lang):
        """Convert sentence to dict format if it's a string and ensure target translation."""
        if isinstance(s, str):
            target_text = s if target_lang == "en" else await translate_text(s, "en", target_lang)
            return {"en": s, target_lang: target_text}
        elif isinstance(s, dict):
            if target_lang not in s and s.get("en"):
                target_text = s["en"] if target_lang == "en" else await translate_text(s["en"], "en", target_lang)
                s = {**s, target_lang: target_text}
            return s
        return {"en": "", target_lang: ""}
    
    lesson = []
    for item in selected:
        word_source = item["word"]
        word_target = word_source if target_lang == "en" else await translate_text(word_source, "en", target_lang)
        meaning_en = item.get("meaning_en", "")
        meaning_target = item.get(f"meaning_{target_lang}")
        if not meaning_target:
            meaning_target = meaning_en if target_lang == "en" else await translate_text(meaning_en, "en", target_lang)
        
        existing_sentences_raw = item.get("sentences", [])
        
        # Normalize sentences to dict format
        existing_sentences = []
        for s in existing_sentences_raw:
            existing_sentences.append(await normalize_sentence(s, target_lang))
        
        
        if len(existing_sentences) >= SENTENCES_PER_WORD_STRICT:
            sentences = existing_sentences[:SENTENCES_PER_WORD_STRICT]
        elif existing_sentences:
            
            needed = SENTENCES_PER_WORD_STRICT - len(existing_sentences)
            extra = await generate_sentences_llm(word_target, target_lang, needed, model=model)
            sentences = existing_sentences + extra
        else:
            
            sentences = await generate_sentences_llm(word_target, target_lang, SENTENCES_PER_WORD_STRICT, model=model)
        
        lesson.append({
            "word": word_target,
            "meaning_en": meaning_en,
            f"meaning_{target_lang}": meaning_target,
            "sentences": sentences[:SENTENCES_PER_WORD_STRICT]
        })
    
    return lesson




async def analyze_pronunciation_detailed(audio_path: str, expected_text: str, spoken_text: str, level: str = "Intermediate", target_lang: str = "en") -> dict:
    """detailed pronunciation analysis combining wer and word confidence"""
    
    
    transcription_data = await transcribe_audio_with_words(audio_path, target_lang)
    
    
    base_score = pronunciation_score_wer(expected_text, spoken_text)
    
    
    mispronounced = []
    if transcription_data.get("words"):
        for word_data in transcription_data["words"]:
            if word_data.get("probability", 1.0) < 0.7:
                mispronounced.append({
                    "word": word_data["word"],
                    "confidence": round(word_data["probability"] * 100, 1),
                    "suggestion": "speak more clearly"
                })
    
    
    duration = transcription_data.get("duration", 5)
    word_count = len(spoken_text.split())
    wpm = int((word_count / duration) * 60) if duration > 0 else 0
    
    return {
        "score": base_score,
        "mispronounced_words": mispronounced,
        "fluency": {
            "wpm": wpm,
            "speed_status": "slow" if wpm < 100 else "normal" if wpm < 160 else "fast",
            "duration": round(duration, 2)
        },
        "transcription": spoken_text
    }






async def generate_word_feedback(expected: str, spoken: str, score: int, attempt: int, word_analysis: dict = None, model: str = "gpt") -> dict:
    """Generate LLM-based feedback for word practice with specific tips"""
    
    if score >= PASSING_SCORE:
        return {
            "status": "success",
            "message": f"Awesome! 🎉 You nailed '{expected}'! That sounded really clear. Now let's try it in a sentence!",
            "next_action": "next_phase"
        }
    elif attempt >= MAX_ATTEMPTS:
        return {
            "status": "max_attempts",
            "message": f"Hey, don't worry! '{expected}' is tricky. You gave it {MAX_ATTEMPTS} good tries - let's move on to sentences and come back to it!",
            "next_action": "next_phase"
        }
    else:
        
        confidence = word_analysis.get("confidence", 0) if word_analysis else 0
        detected = word_analysis.get("detected", False) if word_analysis else False
        
        prompt = f"""You are a super friendly pronunciation buddy (like a supportive friend, NOT a teacher). Give casual, encouraging feedback.

Word to pronounce: "{expected}"
User said: "{spoken}"
Score: {score}%
Word detected: {detected}
Confidence: {confidence}%
Attempt: {attempt} of {MAX_ATTEMPTS}

Generate a response that feels like a friend helping out:
1. START with a warm one-liner reaction (like "Almost there!" or "You're so close!" or "Nice try!")
2. Then 1 sentence of encouragement + ONE specific tip for "{expected}"
3. Mention it's attempt {attempt} of {MAX_ATTEMPTS} naturally

Return JSON: {{"message": "your friendly feedback here", "tip": "quick pronunciation tip"}}"""

        try:
            raw = await call_llm(prompt, timeout=10, model=model)
            data = safe_json_loads(raw)
            message = data.get("message", f"Attempt {attempt}/{MAX_ATTEMPTS}. Try saying '{expected}' more clearly.")
            tip = data.get("tip", "Speak slowly and clearly")
        except:
            message = f"Attempt {attempt}/{MAX_ATTEMPTS}. Focus on pronouncing '{expected}' clearly."
            tip = "Try breaking the word into syllables"
        
        return {
            "status": "retry",
            "message": message,
            "tip": tip,
            "next_action": "retry"
        }


async def generate_sentence_feedback(expected: str, spoken: str, score: int, analysis: dict, model: str = "gpt") -> dict:
    """Generate LLM-based feedback for sentence practice with mismatch details"""
    pron_analysis = analysis.get("pronunciation", {})
    mismatches = pron_analysis.get("mismatches", [])
    mispronounced = pron_analysis.get("mispronounced_words", [])
    fluency = pron_analysis.get("fluency", {})
    wpm = fluency.get("wpm", 0)
    speed_status = fluency.get("speed_status", "normal")
    
    if score >= PASSING_SCORE:
        
        prompt = f"""You are a super friendly pronunciation buddy (like a supportive friend). Give casual, celebratory feedback!

Expected: "{expected}"
User said: "{spoken}"
Score: {score}%
Speaking speed: {wpm} WPM ({speed_status})

Generate a warm, friendly response:
1. START with an excited one-liner ("Nice! 🔥", "You're on fire!", "That was great!")
2. Then 1 sentence mentioning their good pronunciation
3. Optionally mention their speaking pace

Return JSON: {{"message": "your excited feedback", "fluency_note": "speed observation"}}"""

        try:
            raw = await call_llm(prompt, timeout=10, model=model)
            data = safe_json_loads(raw)
            message = data.get("message", "Great pronunciation! That sounded natural.")
            fluency_note = data.get("fluency_note", f"Your pace was {speed_status}.")
        except:
            message = "That sounded great! Well done."
            fluency_note = f"Speaking speed: {speed_status}"
        
        return {
            "status": "success",
            "message": message,
            "fluency_note": fluency_note,
            "pronunciation_feedback": pron_analysis,
            "next_action": "next_sentence"
        }
    else:
        
        mismatch_info = ", ".join([f"'{m['expected']}' vs '{m['spoken']}'" for m in mismatches[:3]]) if mismatches else "some words unclear"
        low_conf_words = [w["word"] for w in mispronounced[:3]] if mispronounced else []
        
        prompt = f"""You are a super friendly pronunciation buddy (like a supportive friend). Give casual, motivating feedback.

Expected: "{expected}"
User said: "{spoken}"
Score: {score}%
Mismatched words: {mismatch_info}
Low confidence words: {low_conf_words}
Speaking speed: {wpm} WPM ({speed_status})

Generate a response like a friend would:
1. START with a warm one-liner ("You're getting there!", "Almost!", "Good try!")
2. Then 1 sentence focusing on ONE word they should practice
3. Keep it encouraging and casual!

Return JSON: {{"message": "your friendly feedback", "focus_word": "word to practice", "tip": "quick tip"}}"""

        try:
            raw = await call_llm(prompt, timeout=10, model=model)
            data = safe_json_loads(raw)
            message = data.get("message", "Good effort! Focus on speaking more clearly.")
            focus_word = data.get("focus_word", "")
            tip = data.get("tip", "Speak slowly and clearly")
        except:
            message = "Good effort! Try to match the sentence more closely."
            focus_word = mismatches[0]["expected"] if mismatches else ""
            tip = "Focus on each word clearly"
        
        return {
            "status": "needs_improvement",
            "message": message,
            "focus_word": focus_word,
            "tip": tip,
            "pronunciation_feedback": pron_analysis,
            "next_action": "next_sentence"
        }


async def generate_session_summary(session: dict, model: str = "gpt") -> dict:
    """Generate comprehensive end of session summary with per-turn WPM analysis - aligned with interview/fluent APIs"""
    history = session.get("history", [])
    
    total_attempts = len(history)
    successful_attempts = sum(1 for h in history if h.get("score", 0) >= PASSING_SCORE)
    
    avg_score = sum(h.get("score", 0) for h in history) / total_attempts if total_attempts > 0 else 0
    
    
    turn_history = []
    total_wpm = 0
    for i, h in enumerate(history, 1):
        wpm = h.get("wpm", 0)
        total_wpm += wpm
        # Extract word properly - could be string or dict
        word_data = h.get("word", "")
        if isinstance(word_data, dict):
            word_str = word_data.get("target", word_data.get("word", ""))
        else:
            word_str = str(word_data)
        
        turn_history.append({
            "turn": i,
            "word": word_str,
            "score": h.get("score", 0),
            "wpm": wpm,
            "passed": h.get("score", 0) >= PASSING_SCORE
        })
    
    average_wpm = int(total_wpm / total_attempts) if total_attempts > 0 else 0
    wpm_status = "slow" if average_wpm < 100 else "normal" if average_wpm <= 150 else "fast"
    
    
    strengths = []
    improvement_areas = []
    if avg_score >= 80:
        strengths.append("pronunciation accuracy")
    elif avg_score < 60:
        improvement_areas.append("pronunciation accuracy")
    
    if average_wpm >= 100 and average_wpm <= 150:
        strengths.append("speaking pace")
    elif average_wpm < 100:
        improvement_areas.append("speaking pace - try to speak faster")
    elif average_wpm > 150:
        improvement_areas.append("speaking pace - slow down for clarity")
    
    # Helper to extract word string from history item
    def get_word_str(h):
        word_data = h.get("word", "")
        if isinstance(word_data, dict):
            return word_data.get("target", word_data.get("word", ""))
        return str(word_data)
    
    difficult_words = [get_word_str(h) for h in history if h.get("score", 0) < 70][:5]
    well_pronounced = [get_word_str(h) for h in history if h.get("score", 0) >= 85][:5]
    
    
    prompt = f"""You are an expert pronunciation coach providing a detailed session summary.

SESSION DATA:
- Student: {session.get('user_name', 'User')}
- Level: {session.get('level', 'B1')}
- Total Words Practiced: {session.get('total_words', 0)}
- Success Rate: {successful_attempts}/{total_attempts} ({round(successful_attempts/total_attempts*100, 1) if total_attempts > 0 else 0}%)
- Average Score: {round(avg_score, 1)}%
- Average WPM: {average_wpm}

PER-TURN PERFORMANCE:
{json.dumps(turn_history, indent=2)}

DIFFICULT WORDS: {difficult_words}
WELL PRONOUNCED: {well_pronounced}

Generate a detailed, personalized, and encouraging session summary analyzing WPM trends.

Return STRICTLY valid JSON:
{{
    "overall_assessment": "3-4 sentences summarizing their pronunciation practice, mentioning WPM trends",
    "pronunciation_feedback": {{
        "score": {round(avg_score, 1)},
        "status": "Excellent/Good/Needs Work",
        "what_went_well": "specific positive observation",
        "improvement_tip": "specific actionable tip",
        "practice_words": {json.dumps(difficult_words[:3])}
    }},
    "fluency_feedback": {{
        "score": {average_wpm},
        "wpm_status": "{wpm_status}",
        "trend": "analysis of WPM across turns - improving/declining/stable",
        "tip": "specific tip for speaking pace"
    }},
    "action_plan": [
        "specific action item 1",
        "specific action item 2",
        "specific action item 3"
    ],
    "encouragement": "2-3 encouraging sentences personalized for the student",
    "next_practice_words": ["word1", "word2", "word3"]
}}
"""
    
    try:
        raw = await call_llm(prompt, model=model, timeout=20)
        data = safe_json_loads(raw)
        
        return {
            "overall_score": int(avg_score),  
            "total_words": session.get("total_words", 0),
            "average_score": round(avg_score, 1),
            "successful_attempts": successful_attempts,
            "total_attempts": total_attempts,
            "average_wpm": average_wpm,
            "wpm_status": wpm_status,
            "turn_history": turn_history,
            "strengths": strengths,
            "improvement_areas": improvement_areas,
            "difficult_words": difficult_words,
            "well_pronounced": well_pronounced,
            
            "overall_assessment": data.get("overall_assessment", f"Great practice session with {total_attempts} words!"),
            "pronunciation_feedback": data.get("pronunciation_feedback", {"score": round(avg_score, 1), "status": "Good"}),
            "fluency_feedback": data.get("fluency_feedback", {"score": average_wpm, "wpm_status": wpm_status}),
            "action_plan": data.get("action_plan", ["Keep practicing daily"]),
            "encouragement": data.get("encouragement", "Keep up the great work!"),
            "next_practice_words": data.get("next_practice_words", difficult_words[:3])
        }
    except Exception as e:
        logger.error(f"summary generation error: {e}")
        return {
            "overall_score": int(avg_score),  
            "total_words": session.get("total_words", 0),
            "average_score": round(avg_score, 1),
            "successful_attempts": successful_attempts,
            "total_attempts": total_attempts,
            "average_wpm": average_wpm,
            "wpm_status": wpm_status,
            "turn_history": turn_history,
            "strengths": strengths,
            "improvement_areas": improvement_areas,
            "difficult_words": difficult_words,
            "well_pronounced": well_pronounced,
            "overall_assessment": "Great practice session! Keep up the good work.",
            "pronunciation_feedback": {"score": round(avg_score, 1), "status": "Good", "improvement_tip": "Practice difficult words slowly"},
            "fluency_feedback": {"score": average_wpm, "wpm_status": wpm_status, "trend": "stable"},
            "action_plan": ["Practice daily for best results", "Focus on difficult words"],
            "encouragement": "You're making progress! Keep practicing.",
            "next_practice_words": difficult_words[:3]
        }






@router.post("/practice_pronunciation")
async def practice_pronunciation(
    name: str = Form(default="User"),
    level: str = Form(default="B1"),
    mode: str = Form(default="normal"),
    native_language: str = Form(...),  
    target_lang: str = Form(default="en"),  
    topic: str = Form(default="daily life"),
    num_words: int = Form(default=5),
    set_number: int = Form(default=None),  
    audio_file: Optional[UploadFile] = File(default=None),
    session_id: Optional[str] = Form(default=None),
    action: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default="gpt"),
    voice_id: Optional[str] = Form(default=None),
    request: Request = None,
    current_user: User = Depends(get_current_user),
):

    """
    pronunciation practice api - handles word and sentence practice
    
    modes:
    - normal: llm-generated lessons with full analysis
    - strict: vocab file based with 15 words, 3 sentences each
    
    flow:
    1. first call (no audio): creates session, returns first word
    2. with audio: analyzes pronunciation, returns feedback
    3. action="next": skip to next word/sentence
    4. action="end": end session early
    """
    
    try:
        
        if action == "end" and session_id:
            session = await db.get_user_session(session_id)
            if session:
                summary = await generate_session_summary(session, model=model)
                native_lang = session.get("native_language", "en")
                summary_bilingual = await make_bilingual(summary, "en", native_lang)
                msg_en = "Session ended. Great practice!"
                msg_target = msg_en if session.get("target_lang", "en") == "en" else await translate_text(msg_en, "en", session.get("target_lang", "en"))
                msg_native = msg_target if native_lang == session.get("target_lang", "en") else await translate_text(msg_en, "en", native_lang)
                
                # Build response first, then save it
                response = {
                    "status": "complete",
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "is_session_complete": True,
                    "session_summary": summary_bilingual,
                    "message": {"target": msg_target, "native": msg_native}
                }
                
                await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                
                return response
            else:
                
                return {
                    "status": "error",
                    "session_id": session_id,
                    "error": "Session not found or already expired. Cannot end a non-existent session."
                }
        
        
        session = None
        if session_id:
            session = await db.get_user_session(session_id)
        
        
        if session and session.get("status") == "completed":
            return {"status": "error", "session_id": session_id, "error": "This session has ended. Please start a new session."}
        
        if not session:
            session_id = str(uuid.uuid4())
            
            
            if mode == "strict":
                lesson = await build_lesson_strict(target_lang, num_words, set_number=set_number, model=model)
            else:
                lesson = await generate_lesson_llm(topic, num_words, target_lang, model=model)
            
            if not lesson:
                return {
                    "status": "error",
                    "message": "failed to generate lesson. please try again."
                }
            
            
            session = {
                "user_name": name,
                "mode": mode,
                "level": level,
                "native_language": native_language,  
                "target_lang": target_lang,
                "topic": topic,
                "lesson": lesson,
                "current_word_index": 0,
                "current_phase": "word",
                "current_sentence_index": 0,
                "attempt_count": 0,
                "history": [],
                "scores": {"pronunciation": []},
                "total_words": len(lesson),
                "turn_history": []
            }
            await db.create_session(
                session_id=session_id,
                session_type="pronunciation",
                data=session,
                user_id=current_user.id if current_user else None,
                user_name=name
            )
            
            
            first_word = lesson[0]
            
            
            greeting_en = f"Hi {name}! I'm Sara. Let's practice pronunciation together. Relax and speak naturally."
            instruction_en = f"Listen carefully and repeat after me: {first_word['word']}"
            greeting_target = greeting_en if target_lang == "en" else await translate_text(greeting_en, "en", target_lang)
            instruction_target = instruction_en if target_lang == "en" else await translate_text(instruction_en, "en", target_lang)
            greeting_native = greeting_target if native_language == target_lang else await translate_text(greeting_en, "en", native_language)
            instruction_native = instruction_target if native_language == target_lang else await translate_text(instruction_en, "en", native_language)
            
            # Generate TTS audio URLs for greeting
            greeting_audio = ""
            if request:
                greeting_audio = await generate_tts_url(request, greeting_target, target_lang, voice_id=voice_id)
            
            meaning_target = first_word.get(f"meaning_{target_lang}", first_word.get("meaning_en", ""))
            meaning_native = first_word.get(f"meaning_{native_language}", "")
            if not meaning_native:
                source_lang = target_lang if first_word.get(f"meaning_{target_lang}") else "en"
                meaning_native = await translate_text(meaning_target, source_lang, native_language)
            
            return {
                "status": "new_session",
                "session_id": session_id,
                "target_lang": target_lang,
                "native_lang": native_language,
                "mode": mode,
                "greeting": {"target": greeting_target, "native": greeting_native, "audio_url": greeting_audio},
                "current_word": {
                    "word": first_word["word"],
                    "meaning": {
                        "target": meaning_target,
                        "native": meaning_native
                    },
                    "instruction": {"target": instruction_target, "native": instruction_native}
                },
                "phase": "word",
                "attempt_number": 1,
                "max_attempts": MAX_ATTEMPTS,
                "progress": {
                    "current_word_index": 1,
                    "total_words": len(lesson),
                    "completed_words": []
                }
            }
        
        
        lesson = session["lesson"]
        current_idx = session["current_word_index"]
        current_phase = session["current_phase"]
        current_word = lesson[current_idx]
        
        
        
        native_lang = session.get("native_language", "en")
        
        if action == "next":
            
            session_mode = session.get("mode", "normal")
            
            if current_phase == "word":
                
                session["current_phase"] = "sentence"
                session["current_sentence_index"] = 0
                session["attempt_count"] = 0
                
                instruction_en = "Now practice this sentence."
                instruction_target = instruction_en if target_lang == "en" else await translate_text(instruction_en, "en", target_lang)
                instruction_native = instruction_target if native_lang == target_lang else await translate_text(instruction_en, "en", native_lang)
                
                
                if "sentences" in current_word and current_word["sentences"]:
                    sentence = current_word["sentences"][0]
                    sentence_target = safe_get_sentence_text(sentence, target_lang)
                    source_lang = target_lang if isinstance(sentence, dict) and sentence.get(target_lang) else "en"
                    sentence_native = safe_get_sentence_text(sentence, native_lang) or await translate_text(sentence_target, source_lang, native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_phase",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": sentence_target, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": 1,
                        "total_sentences": len(current_word["sentences"]),
                        "instruction": {"target": instruction_target, "native": instruction_native},
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
                else:
                    
                    sentence_en = current_word.get("sentence", f"Practice saying {current_word['word']}.")
                    sentence_target = sentence_en if target_lang == "en" else await translate_text(sentence_en, "en", target_lang)
                    sentence_native = await translate_text(sentence_target, target_lang if target_lang != "en" else "en", native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_phase",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": sentence_target, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": 1,
                        "total_sentences": 1,
                        "instruction": {"target": instruction_target, "native": instruction_native},
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
            else:
                
                sentence_idx = session.get("current_sentence_index", 0)
                
                
                if "sentences" in current_word and sentence_idx + 1 < len(current_word["sentences"]):
                    
                    session["current_sentence_index"] = sentence_idx + 1
                    next_sentence = current_word["sentences"][session["current_sentence_index"]]
                    next_sentence_target = safe_get_sentence_text(next_sentence, target_lang)
                    source_lang = target_lang if isinstance(next_sentence, dict) and next_sentence.get(target_lang) else "en"
                    sentence_native = safe_get_sentence_text(next_sentence, native_lang) or await translate_text(next_sentence_target, source_lang, native_lang)
                    
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "next_sentence",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "current_word": {"word": current_word["word"]},
                        "current_sentence": {
                            "text": {"target": next_sentence_target, "native": sentence_native}
                        },
                        "phase": "sentence",
                        "sentence_number": session["current_sentence_index"] + 1,
                        "total_sentences": len(current_word["sentences"]),
                        "progress": {
                            "current_word_index": current_idx + 1,
                            "total_words": len(lesson)
                        }
                    }
                
                
                session["current_word_index"] += 1
                session["current_phase"] = "word"
                session["current_sentence_index"] = 0
                session["attempt_count"] = 0
                
                if session["current_word_index"] >= len(lesson):
                    
                    summary = await generate_session_summary(session)
                    
                    summary_bilingual = await make_bilingual(summary, "en", native_lang)
                    msg_en = "Excellent work! You've completed all words."
                    msg_native = await translate_text(msg_en, "en", native_lang)
                    
                    # Build response first, then save it
                    response = {
                        "status": "complete",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "is_session_complete": True,
                        "session_summary": summary_bilingual,
                        "message": {"target": msg_en, "native": msg_native}
                    }
                    
                    await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                    
                    return response
                
                next_word = lesson[session["current_word_index"]]
                instruction_en = f"Next word: {next_word['word']}"
                instruction_target = instruction_en if target_lang == "en" else await translate_text(instruction_en, "en", target_lang)
                instruction_native = instruction_target if native_lang == target_lang else await translate_text(instruction_en, "en", native_lang)
                
                
                meaning_target = next_word.get(f"meaning_{target_lang}", next_word.get("meaning_en", ""))
                meaning_native = next_word.get(f"meaning_{native_lang}", "")
                if not meaning_native:
                    source_lang = target_lang if next_word.get(f"meaning_{target_lang}") else "en"
                    meaning_native = await translate_text(meaning_target, source_lang, native_lang)
                
                
                await db.update_session(session_id, session)
                
                return {
                    "status": "next_word",
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "current_word": {
                        "word": next_word["word"],
                        "meaning": {
                            "target": meaning_target,
                            "native": meaning_native
                        },
                        "instruction": {"target": instruction_target, "native": instruction_native}
                    },
                    "phase": "word",
                    "attempt_number": 1,
                    "max_attempts": MAX_ATTEMPTS,
                    "progress": {
                        "current_word_index": session["current_word_index"] + 1,
                        "total_words": len(lesson)
                    }
                }
        
        
        if not audio_file:
            msg_en = "please provide audio to continue"
            msg_native = await translate_text(msg_en, "en", native_lang)
            return {
                "status": "waiting_audio",
                "session_id": session_id,
                "target_lang": session.get("target_lang", "en"),
                "native_lang": native_lang,
                "message": {"target": msg_en, "native": msg_native},
                "current_word": {"word": current_word["word"]},
                "phase": current_phase
            }
        
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"audio_{session_id}.wav")
        
        try:
            
            content = await audio_file.read()
            original_filename = audio_file.filename or "audio.wav"
            original_ext = os.path.splitext(original_filename)[1].lower()
            temp_input_path = os.path.join(temp_dir, f"input_{session_id}{original_ext or '.wav'}")
            
            with open(temp_input_path, "wb") as f:
                f.write(content)
            
            
            if original_ext in ['.mp3', '.m4a', '.ogg', '.flac', '.aac', '.webm']:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(temp_input_path)
                    audio.export(audio_path, format="wav")
                except Exception as conv_err:
                    logger.warning(f"Audio conversion failed, using original: {conv_err}")
                    shutil.copy(temp_input_path, audio_path)
            else:
                
                shutil.copy(temp_input_path, audio_path)
            
            
            target_lang_for_audio = session.get("target_lang", "en")
            transcription = await transcribe_audio(audio_path, target_lang_for_audio)
            
            if not transcription:
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {
                    "status": "transcription_failed",
                    "session_id": session_id,
                    "message": "could not understand audio. please try again.",
                    "phase": current_phase
                }

            
            
            if current_phase == "word":
                expected = current_word["word"]
                
                
                word_analysis = await analyze_word_pronunciation(audio_path, expected, target_lang_for_audio)
                score = word_analysis["score"]
                transcription = word_analysis["transcription"] or transcription
                
                
                try:
                    from pydub import AudioSegment
                    audio_for_wpm = AudioSegment.from_file(audio_path)
                    audio_duration_seconds = len(audio_for_wpm) / 1000
                    word_count = len(transcription.split()) if transcription else 1
                    word_wpm = int((word_count / audio_duration_seconds) * 60) if audio_duration_seconds > 0 else 120
                except:
                    word_wpm = 120  
                
                # Calculate speed_status for word
                if word_wpm < 100:
                    word_speed_status = "slow"
                elif word_wpm <= 150:
                    word_speed_status = "normal"
                else:
                    word_speed_status = "fast"
                
                session["attempt_count"] += 1
                
                feedback = await generate_word_feedback(expected, transcription, score, session["attempt_count"], word_analysis, model=model)
                
                session["history"].append({
                    "phase": "word",
                    "word": {  
                        "target": current_word["word"],
                        "meaning_en": current_word.get("meaning_en", ""),
                        "meaning_native": current_word.get(f"meaning_{session.get('native_language', 'hi')}", current_word.get("meaning_native", ""))
                    },
                    "expected": expected,
                    "spoken": transcription,
                    "score": score,
                    "attempt": session["attempt_count"],
                    "confidence": word_analysis.get("confidence", 0),
                    "wpm": word_wpm,
                    "speed_status": word_speed_status,
                    "pronunciation_analysis": word_analysis,
                    # Store feedback text for /feedback endpoint
                    "feedback_message": feedback.get("message", ""),
                    "feedback_tip": feedback.get("tip", ""),
                    "feedback_status": feedback.get("status", "")
                })
                session["scores"]["pronunciation"].append(score)
                
                syllable_guide = None
                if word_analysis.get("needs_practice"):
                    
                    syllable_cache = session.get("syllable_cache", {})
                    if expected.lower() in syllable_cache:
                        syllable_guide = syllable_cache[expected.lower()]
                    else:
                        syllable_guide = await generate_syllable_guide(expected, model=model)
                        
                        if "syllable_cache" not in session:
                            session["syllable_cache"] = {}
                        session["syllable_cache"][expected.lower()] = syllable_guide
                
                
                current_attempt = session["attempt_count"]
                
                
                if feedback["next_action"] == "next_phase":
                    session["current_phase"] = "sentence"
                    session["current_sentence_index"] = 0
                    session["attempt_count"] = 0
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                
                await db.update_session(session_id, session)
                
                
                feedback_target = feedback["message"] if target_lang_for_audio == "en" else await translate_text(feedback["message"], "en", target_lang_for_audio)
                feedback_native = feedback_target if native_lang == target_lang_for_audio else await translate_text(feedback["message"], "en", native_lang)
                
                # Generate TTS audio URL for feedback
                feedback_audio = ""
                if request:
                    feedback_audio = await generate_tts_url(request, feedback_target, session.get("target_lang", "en"), voice_id=voice_id)
                
                response = {
                    "status": feedback["status"],
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "transcription": transcription,
                    "pronunciation_score": score,
                    "feedback": {"target": feedback_target, "native": feedback_native, "audio_url": feedback_audio},
                    "current_word": {"word": current_word["word"]},
                    "phase": "word" if feedback["next_action"] == "retry" else "sentence",
                    "attempt_number": current_attempt,
                    "max_attempts": MAX_ATTEMPTS,
                    "next_action": feedback["next_action"],
                    "progress": {
                        "current_word_index": current_idx + 1,
                        "total_words": len(lesson)
                    },
                    
                    "analysis": {
                        "pronunciation": {
                            "score": score,
                            "confidence": word_analysis.get("confidence", 0),
                            "expected": expected,
                            "spoken": transcription,
                            "detected": word_analysis.get("detected", False),
                            "match_type": word_analysis.get("match_type", "unknown")
                        }
                    }
                }
                
                
                if syllable_guide:
                    response["syllable_guide"] = syllable_guide
                
                
                if feedback["next_action"] == "next_phase":
                    
                    if "sentences" in current_word and current_word["sentences"]:
                        sentence = current_word["sentences"][0]
                        sentence_target = safe_get_sentence_text(sentence, target_lang)
                        source_lang = target_lang if isinstance(sentence, dict) and sentence.get(target_lang) else "en"
                        sentence_native = safe_get_sentence_text(sentence, native_lang) or await translate_text(sentence_target, source_lang, native_lang)
                        response["current_sentence"] = {
                            "text": {"target": sentence_target, "native": sentence_native}
                        }
                        response["sentence_number"] = 1
                        response["total_sentences"] = len(current_word["sentences"])
                    else:
                        
                        sentence_en = current_word.get("sentence", "")
                        sentence_target = sentence_en if target_lang == "en" else await translate_text(sentence_en, "en", target_lang) if sentence_en else ""
                        sentence_native = current_word.get(f"sentence_{native_lang}", "") or await translate_text(sentence_target, target_lang if target_lang != "en" else "en", native_lang) if sentence_target else ""
                        response["current_sentence"] = {
                            "text": {"target": sentence_target, "native": sentence_native},
                            "example": current_word.get("example", "")
                        }
                        response["sentence_number"] = 1
                        response["total_sentences"] = 1
                
                return response
            
            
            else:
                
                sentence_idx = session.get("current_sentence_index", 0)
                if "sentences" in current_word and current_word["sentences"]:
                    sentences = current_word["sentences"]
                    current_sentence = sentences[sentence_idx]
                    expected = safe_get_sentence_text(current_sentence, target_lang)
                else:
                    
                    expected = current_word.get("sentence", "")
                
                
                sentence_analysis = await analyze_sentence_pronunciation(audio_path, expected, transcription, target_lang_for_audio)
                score = sentence_analysis["score"]
                
                analysis = {
                    "pronunciation": {
                        "score": score,
                        "expected": expected,
                        "spoken": transcription,
                        "mismatches": sentence_analysis.get("mismatches", []),
                        "mismatch_count": sentence_analysis.get("mismatch_count", 0),
                        "mispronounced_words": sentence_analysis.get("mispronounced_words", []),
                        "well_pronounced_words": sentence_analysis.get("well_pronounced_words", []),
                        "fluency": sentence_analysis.get("fluency", {}),
                        "accuracy_percentage": sentence_analysis.get("accuracy_percentage", 0)
                    }
                }
                
                # Get speed_status from fluency analysis
                sentence_speed_status = sentence_analysis.get("fluency", {}).get("speed_status", "normal")
                
                feedback = await generate_sentence_feedback(expected, transcription, score, analysis, model=model)
                
                session["history"].append({
                    "phase": "sentence",
                    "word": {  
                        "target": current_word["word"],
                        "meaning_en": current_word.get("meaning_en", ""),
                        "meaning_native": current_word.get(f"meaning_{session.get('native_language', 'hi')}", current_word.get("meaning_native", ""))
                    },
                    "expected": expected,
                    "spoken": transcription,
                    "score": score,
                    "mismatches": sentence_analysis.get("mismatches", []),
                    "wpm": sentence_analysis.get("fluency", {}).get("wpm", 120),
                    "speed_status": sentence_speed_status,
                    "pronunciation_analysis": sentence_analysis,
                    # Store feedback text for /feedback endpointttttt
                    "feedback_message": feedback.get("message", ""),
                    "feedback_tip": feedback.get("tip", ""),
                    "feedback_status": feedback.get("status", ""),
                    "focus_word": feedback.get("focus_word", ""),
                    "fluency_note": feedback.get("fluency_note", "")
                })
                session["scores"]["pronunciation"].append(score)
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                
                next_action = "next_word"
                is_complete = False
                
                
                if "sentences" in current_word and current_word["sentences"]:
                    sentence_idx = session.get("current_sentence_index", 0)
                    if sentence_idx + 1 < len(current_word["sentences"]):
                        session["current_sentence_index"] = sentence_idx + 1
                        next_action = "next_sentence"
                    else:
                        
                        session["current_word_index"] += 1
                        session["current_phase"] = "word"
                        session["current_sentence_index"] = 0
                        session["attempt_count"] = 0
                        
                        if session["current_word_index"] >= len(lesson):
                            is_complete = True
                            next_action = "complete"
                else:
                    
                    session["current_word_index"] += 1
                    session["current_phase"] = "word"
                    session["attempt_count"] = 0
                    
                    if session["current_word_index"] >= len(lesson):
                        is_complete = True
                        next_action = "complete"
                
                
                feedback_target = feedback["message"] if target_lang_for_audio == "en" else await translate_text(feedback["message"], "en", target_lang_for_audio)
                feedback_native = feedback_target if native_lang == target_lang_for_audio else await translate_text(feedback["message"], "en", native_lang)
                
                # Generate TTS audio URL for feedback
                feedback_audio = ""
                if request:
                    feedback_audio = await generate_tts_url(request, feedback_target, session.get("target_lang", "en"), voice_id=voice_id)
                
                response_status = "complete" if is_complete else feedback["status"]
                
                response = {
                    "status": response_status,
                    "session_id": session_id,
                    "target_lang": session.get("target_lang", "en"),
                    "native_lang": native_lang,
                    "transcription": transcription,
                    "pronunciation_score": score,
                    "feedback": {"target": feedback_target, "native": feedback_native, "audio_url": feedback_audio},
                    "analysis": analysis,
                    "current_word": {"word": current_word["word"]},
                    "phase": "sentence",
                    "next_action": next_action,
                    "is_session_complete": is_complete,
                    "progress": {
                        "current_word_index": session["current_word_index"] + 1 if not is_complete else len(lesson),
                        "total_words": len(lesson)
                    }
                }
                
                
                if is_complete:
                    summary = await generate_session_summary(session, model=model)
                    
                    summary_bilingual = await make_bilingual(summary, "en", native_lang)
                    msg_en = "Excellent work! You've completed the session."
                    msg_target = msg_en if session.get("target_lang", "en") == "en" else await translate_text(msg_en, "en", session.get("target_lang", "en"))
                    msg_native = msg_target if native_lang == session.get("target_lang", "en") else await translate_text(msg_en, "en", native_lang)
                    
                    response = {
                        "status": "complete",
                        "session_id": session_id,
                        "target_lang": session.get("target_lang", "en"),
                        "native_lang": native_lang,
                        "is_session_complete": True,
                        "session_summary": summary_bilingual,
                        "message": {"target": msg_target, "native": msg_native}
                    }
                    
                    await db.complete_session(session_id, final_feedback=summary_bilingual, termination_response=response)
                    return response
                    
                elif next_action == "next_sentence" and "sentences" in current_word:
                    next_sentence = current_word["sentences"][session["current_sentence_index"]]
                    next_sentence_target = safe_get_sentence_text(next_sentence, target_lang)
                    source_lang = target_lang if isinstance(next_sentence, dict) and next_sentence.get(target_lang) else "en"
                    sentence_native = safe_get_sentence_text(next_sentence, native_lang) or await translate_text(next_sentence_target, source_lang, native_lang)
                    response["current_sentence"] = {
                        "text": {"target": next_sentence_target, "native": sentence_native}
                    }
                    response["sentence_number"] = session["current_sentence_index"] + 1
                    response["total_sentences"] = len(current_word["sentences"])
                elif next_action == "next_word" and session["current_word_index"] < len(lesson):
                    next_word = lesson[session["current_word_index"]]
                    
                    meaning_target = next_word.get(f"meaning_{target_lang}", next_word.get("meaning_en", ""))
                    meaning_native = next_word.get(f"meaning_{native_lang}", "")
                    if not meaning_native:
                        source_lang = target_lang if next_word.get(f"meaning_{target_lang}") else "en"
                        meaning_native = await translate_text(meaning_target, source_lang, native_lang)
                    response["next_word"] = {
                        "word": next_word["word"],
                        "meaning": {
                            "target": meaning_target,
                            "native": meaning_native
                        }
                    }
                
                
                await db.update_session(session_id, session)
                
                return response
        
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    except Exception as e:
        logger.error(f"pronunciation api error: {e}")
        return {
            "status": "error",
            "message": f"an error occurred: {str(e)}"
        }






@router.get("/list_pronunciation_sessions")
async def list_pronunciation_sessions():
    """list active pronunciation sessions from database"""
    sessions_list = await db.list_sessions(session_type="pronunciation")
    return {"sessions": sessions_list}


@router.get("/pronunciation_session/{session_id}")
async def get_pronunciation_session(session_id: str):
    """get pronunciation session data from database"""
    session_data = await db.get_user_session(session_id)
    if session_data:
        return {"status": "success", "session_id": session_id, "data": session_data}
    return {"status": "not_found", "session_id": session_id}


@router.get("/get_practice_modes")
async def get_practice_modes():
    """get available practice modes with feature details"""
    
    common_features = {
        "wer_score": True,
        "word_level_confidence": True,
        "mispronounced_words_list": True,
        "wpm_tracking": True,
        "pause_detection": True,
        "detailed_feedback": True
    }
    
    return {
        "modes": [
            {
                "id": "normal",
                "name": "Normal Mode",
                "description": "LLM-generated lessons based on topic. Full pronunciation analysis.",
                "words_per_session": "configurable via num_words",
                "sentences_per_word": SENTENCES_PER_WORD_NORMAL,
                "features": common_features
            },
            {
                "id": "strict",
                "name": "Strict Mode",
                "description": "Vocabulary file-based lessons with configurable word count. Full pronunciation analysis.",
                "words_per_session": "configurable via num_words",
                "sentences_per_word": SENTENCES_PER_WORD_STRICT,
                "features": common_features
            }
        ],
        "feature_descriptions": {
            "wer_score": "Word Error Rate based pronunciation score (0-100%)",
            "word_level_confidence": "Per-word pronunciation confidence from speech recognition",
            "mispronounced_words_list": "List of words with low confidence that need practice",
            "wpm_tracking": "Words per minute speaking speed analysis",
            "pause_detection": "Detection of speaking speed (slow/normal/fast)",
            "detailed_feedback": "AI-generated feedback with improvement tips"
        }
    }


@router.get("/pronunciation_vocab")
async def get_pronunciation_vocab():
    """Get all pronunciation vocabulary from database"""
    async with async_session() as sess:
        result = await sess.execute(
            text("SELECT id, word, meaning_en, sentences, set_number, created_at FROM pronunciation_vocab ORDER BY id")
        )
        rows = result.fetchall()
        vocab_list = []
        for row in rows:
            vocab_list.append({
                "id": row[0],
                "word": row[1],
                "meaning_en": row[2],
                "sentences": row[3] if isinstance(row[3], list) else json.loads(row[3]) if row[3] else [],
                "set_number": row[4],
                "created_at": str(row[5]) if row[5] else None
            })
        return {
            "status": "success",
            "total": len(vocab_list),
            "vocabulary": vocab_list
        }


@router.post("/pronunciation_vocab/upload")
async def upload_pronunciation_vocab(
    file: UploadFile = File(...),
    replace_all: bool = Form(default=False),
    default_set_number: int = Form(default=None)  
):
    """
    Upload pronunciation vocabulary from Excel file.
    
    Excel format (with optional set_number column):
    | set_number | word | meaning_en | sentence_1 | sentence_2 | sentence_3 |
    
    Or:
    | word | meaning_en | sentences (comma-separated) |
    
    If set_number column not in Excel, uses default_set_number param.
    """
    import pandas as pd
    import io
    
    
    content = await file.read()
    
    try:
        
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            return {"status": "error", "message": "Unsupported file format. Use .xlsx, .xls, or .csv"}
        
        
        required_cols = ['word', 'meaning_en']
        if not all(col in df.columns for col in required_cols):
            return {"status": "error", "message": f"Missing required columns: {required_cols}"}
        
        
        vocab_items = []
        for _, row in df.iterrows():
            word = str(row['word']).strip()
            meaning_en = str(row['meaning_en']).strip()
            
            
            sentences = []
            
            
            for col in df.columns:
                if col.startswith('sentence_') and pd.notna(row.get(col)):
                    sentences.append(str(row[col]).strip())
            
            
            if not sentences and 'sentences' in df.columns and pd.notna(row.get('sentences')):
                sentences_str = str(row['sentences'])
                sentences = [s.strip() for s in sentences_str.split(',') if s.strip()]
            
            
            if not sentences:
                sentences = [f"I use the word {word} every day."]
            
            
            row_set_number = None
            if 'set_number' in df.columns and pd.notna(row.get('set_number')):
                row_set_number = int(row['set_number'])
            elif default_set_number is not None:
                row_set_number = default_set_number
            
            vocab_items.append({
                "word": word,
                "meaning_en": meaning_en,
                "sentences": sentences,
                "set_number": row_set_number
            })
        
        
        async with async_session() as sess:
            
            if replace_all:
                await sess.execute(text("DELETE FROM pronunciation_vocab"))
            
            inserted = 0
            skipped = 0
            
            for item in vocab_items:
                try:
                    await sess.execute(
                        text("INSERT INTO pronunciation_vocab (word, meaning_en, sentences, set_number, created_at) VALUES (:word, :meaning_en, cast(:sentences as jsonb), :set_number, NOW()) ON CONFLICT (word) DO UPDATE SET meaning_en = :meaning_en, sentences = cast(:sentences as jsonb), set_number = :set_number"),
                        {"word": item["word"], "meaning_en": item["meaning_en"], "sentences": json.dumps(item["sentences"]), "set_number": item.get("set_number")}
                    )
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting {item['word']}: {e}")
                    skipped += 1
            
            await sess.commit()
        
        return {
            "status": "success",
            "message": f"Uploaded vocabulary successfully",
            "inserted": inserted,
            "skipped": skipped,
            "total_in_file": len(vocab_items)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/completed_sessions")
async def get_completed_pronunciation_sessions(current_user: User = Depends(get_current_user)):
    """
    Get only completed pronunciation sessions for the current user.
    Returns sessions where status='completed' and termination_response exists.
    """
    user_id = current_user.id if current_user else None
    sessions = await db.get_sessions_by_user_id(user_id, session_type="pronunciation")
    
    completed_sessions = []
    for s in sessions:
        session_data = await db.get_user_session(s.get("session_id"))
        if not session_data:
            continue
        if session_data.get("status") != "completed":
            continue
        if not session_data.get("termination_response"):
            continue
        
        completed_sessions.append({
            "session_id": s.get("session_id"),
            "created_at": s.get("created_at"),
            "target_lang": session_data.get("target_lang", "en"),
            "native_lang": session_data.get("native_language", "en"),
            "mode": session_data.get("mode", "normal")
        })
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_completed": len(completed_sessions),
        "sessions": completed_sessions
    }


@router.get("/feedback/{session_id}")
async def get_pronunciation_feedback(session_id: str):
    """
    Get detailed per-turn feedback for a pronunciation session.
    
    Returns the same response that was returned when the session ended.
    Falls back to structured per-turn feedback if termination_response not available.
    """
    # First try to get the stored termination response
    session_data = await db.get_user_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return the stored termination response if available (same as session end response)
    if "termination_response" in session_data:
        return session_data["termination_response"]
    
    # Fall back to get_session_feedback for older sessions without termination_response
    feedback = await db.get_session_feedback(session_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Session not found")
    if feedback["session_type"] != "pronunciation":
        raise HTTPException(status_code=400, detail="Not a pronunciation session")
    return feedback


async def init_pronunciation_tables():
    """
    Create/sync pronunciation tables - call at app startup.
    Adds missing columns to existing tables.
    """
    async with async_session() as sess:
        # Create pronunciation_vocab table if not exists
        await sess.execute(text("""
            CREATE TABLE IF NOT EXISTS pronunciation_vocab (
                id SERIAL PRIMARY KEY,
                word VARCHAR(255) UNIQUE NOT NULL,
                meaning_en TEXT,
                sentences JSONB,
                set_number INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Add missing columns to pronunciation_vocab (safe - ignores if exists)
        migration_queries = [
            "ALTER TABLE pronunciation_vocab ADD COLUMN IF NOT EXISTS meaning_en TEXT",
            "ALTER TABLE pronunciation_vocab ADD COLUMN IF NOT EXISTS sentences JSONB",
            "ALTER TABLE pronunciation_vocab ADD COLUMN IF NOT EXISTS set_number INTEGER",
            "ALTER TABLE pronunciation_vocab ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        ]
        for query in migration_queries:
            try:
                await sess.execute(text(query))
            except Exception as e:
                logger.debug(f"Migration note: {e}")
        
        await sess.commit()
    logger.info("pronunciation_vocab table ready")


async def init_bookmarks_table():
    """Create/sync user_bookmarks table - call at app startup"""
    async with async_session() as sess:
        await sess.execute(text("""
            CREATE TABLE IF NOT EXISTS user_bookmarks (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                word VARCHAR(255) NOT NULL,
                meaning_target TEXT,
                meaning_native TEXT,
                target_lang VARCHAR(10) DEFAULT 'en',
                native_lang VARCHAR(10) DEFAULT 'hi',
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, word)
            )
        """))
        
        # Add missing columns (safe - ignores if exists)
        migration_queries = [
            "ALTER TABLE user_bookmarks ADD COLUMN IF NOT EXISTS meaning_target TEXT",
            "ALTER TABLE user_bookmarks ADD COLUMN IF NOT EXISTS meaning_native TEXT",
            "ALTER TABLE user_bookmarks ADD COLUMN IF NOT EXISTS target_lang VARCHAR(10) DEFAULT 'en'",
            "ALTER TABLE user_bookmarks ADD COLUMN IF NOT EXISTS native_lang VARCHAR(10) DEFAULT 'hi'",
            "ALTER TABLE user_bookmarks ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        ]
        for query in migration_queries:
            try:
                await sess.execute(text(query))
            except Exception as e:
                logger.debug(f"Migration note: {e}")
        
        await sess.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bookmarks_user_lang ON user_bookmarks(user_id, native_lang)
        """))
        await sess.commit()
    logger.info("user_bookmarks table ready")


async def init_all_pronunciation_tables():
    """Initialize all pronunciation-related tables - call this at app startup"""
    await init_pronunciation_tables()
    await init_bookmarks_table()
    logger.info("All pronunciation tables initialized")

@router.post("/bookmark")
async def add_bookmark(
    word: str = Form(...),
    meaning_target: str = Form(default=""),  
    meaning_native: str = Form(default=""),  
    target_lang: str = Form(default="en"),
    native_lang: str = Form(default="hi"),
    current_user: User = Depends(get_current_user)
):
    """
    Bookmark a word with meanings in both target and native languages.
    
    When moving to auth project:
    - Remove user_id from Form
    - Add: current_user: User = Depends(get_current_user)
    - Use: user_id = current_user.id
    """
    user_id = current_user.id
    async with async_session() as sess:
        
        existing = await sess.execute(
            text("SELECT id FROM user_bookmarks WHERE user_id = :uid AND word = :word"),
            {"uid": user_id, "word": word.lower().strip()}
        )
        if existing.fetchone():
            return {"status": "already_bookmarked", "word": word}
        
        
        await sess.execute(
            text("""
                INSERT INTO user_bookmarks (user_id, word, meaning_target, meaning_native, target_lang, native_lang, created_at)
                VALUES (:uid, :word, :m_target, :m_native, :t_lang, :n_lang, NOW())
            """),
            {
                "uid": user_id,
                "word": word.lower().strip(),
                "m_target": meaning_target,
                "m_native": meaning_native,
                "t_lang": target_lang,
                "n_lang": native_lang
            }
        )
        await sess.commit()
    
    return {"status": "bookmarked", "word": word, "user_id": user_id}


@router.get("/bookmarks")
async def get_bookmarks(
    native_lang: Optional[str] = None,  
    target_lang: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all bookmarked words for a user.
    Optionally filter by native_lang, target_lang, or both.
    
    When moving to auth project:
    - Remove user_id query param
    - Add: current_user: User = Depends(get_current_user)
    - Use: user_id = current_user.id
    """
    user_id = current_user.id
    async with async_session() as sess:
        
        base_query = """
            SELECT word, meaning_target, meaning_native, target_lang, native_lang, created_at
            FROM user_bookmarks
            WHERE user_id = :uid
        """
        params = {"uid": user_id}
        
        if native_lang and target_lang:
            query = base_query + " AND native_lang = :n_lang AND target_lang = :t_lang ORDER BY created_at DESC"
            params["n_lang"] = native_lang
            params["t_lang"] = target_lang
        elif native_lang:
            query = base_query + " AND native_lang = :n_lang ORDER BY created_at DESC"
            params["n_lang"] = native_lang
        elif target_lang:
            query = base_query + " AND target_lang = :t_lang ORDER BY created_at DESC"
            params["t_lang"] = target_lang
        else:
            query = base_query + " ORDER BY created_at DESC"
        
        result = await sess.execute(text(query), params)
        rows = result.fetchall()
    
    words = [
        {
            "word": r[0],
            "meaning_target": r[1],
            "meaning_native": r[2],
            "target_lang": r[3],
            "native_lang": r[4],
            "created_at": str(r[5]) if r[5] else None
        }
        for r in rows
    ]
    
    
    filter_info = {}
    if native_lang:
        filter_info["native_lang"] = native_lang
    if target_lang:
        filter_info["target_lang"] = target_lang
    
    return {
        "user_id": user_id,
        "total": len(words),
        "filter": filter_info if filter_info else None,
        "words": words
    }


@router.delete("/bookmark/{word}")
async def remove_bookmark(
    word: str,
    current_user: User = Depends(get_current_user)
):
    """
    Remove a bookmarked word.
    
    When moving to auth project:
    - Remove user_id query param
    - Add: current_user: User = Depends(get_current_user)
    - Use: user_id = current_user.id
    """
    user_id = current_user.id
    async with async_session() as sess:
        result = await sess.execute(
            text("DELETE FROM user_bookmarks WHERE user_id = :uid AND word = :word RETURNING id"),
            {"uid": user_id, "word": word.lower().strip()}
        )
        deleted = result.fetchone()
        await sess.commit()
    
    if deleted:
        return {"status": "removed", "word": word}
    else:
        return {"status": "not_found", "word": word}


@router.get("/user_sessions")
async def get_pronunciation_sessions_by_user(current_user: User = Depends(get_current_user)):
    """
    Get all Pronunciation sessions for a specific user.
    
    Returns all session_ids, scores, and status for the user.
    """
    user_id = current_user.id
    sessions = await db.get_sessions_by_user_id(user_id, session_type="pronunciation")
    # Add session_number for frontend display
    for idx, session in enumerate(sessions, 1):
        session["session_number"] = f"Session {idx}"
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@router.get("/user_sessions/ids")
async def get_pronunciation_session_ids(current_user: User = Depends(get_current_user)):
    """
    Get just the session IDs for a user.
    
    Returns only session_ids list that can be used to fetch individual feedback.
    """
    user_id = current_user.id
    sessions = await db.get_sessions_by_user_id(user_id, session_type="pronunciation")
    session_ids = [s.get("session_id") for s in sessions]
    return {
        "user_id": user_id,
        "total_sessions": len(session_ids),
        "session_ids": session_ids
    }


@router.get("/user_sessions/detailed")
async def get_pronunciation_sessions_detailed(current_user: User = Depends(get_current_user)):
    """

    Returns sessions labeled as Session 1, Session 2, etc.
    with complete turn-by-turn feedback.
    """
    user_id = current_user.id
    sessions = await db.get_sessions_by_user_id(user_id, session_type="pronunciation")
    
    detailed_sessions = []
    for idx, session in enumerate(sessions, 1):
        session_id = session.get("session_id")
        session_data = await db.get_user_session(session_id)
        feedback = await db.get_session_feedback(session_id)
        
        # Get session metadata
        lesson_id = session_data.get("lesson_id") if session_data else None
        target_lang = session_data.get("target_lang", "en") if session_data else "en"
        
        # Get full turn feedback
        full_turns = []
        if feedback and feedback.get("turn_feedback"):
            for turn in feedback.get("turn_feedback", []):
                full_turns.append({
                    "turn": turn.get("turn"),
                    "word": turn.get("word", {}),
                    "transcription": turn.get("transcription", ""),
                    "pronunciation_score": turn.get("pronunciation_score", 0),
                    "pronunciation": turn.get("pronunciation", {}),
                    "fluency": turn.get("fluency", {}),
                    "wpm": turn.get("wpm", 0)
                })
        
        # Get full final feedback
        final = feedback.get("final_feedback", {}) if feedback else {}
        
        detailed_sessions.append({
            "session_number": f"Session {idx}",
            "session_id": session_id,
            "lesson_id": lesson_id,
            "target_lang": target_lang,
            "overall_score": session.get("overall_score", 0),
            "status": session.get("status", "active"),
            "created_at": session.get("created_at"),
            "total_turns": len(full_turns),
            "turns": full_turns,
            "final_feedback": final
        })
    
    return {
        "user_id": user_id,
        "total_sessions": len(detailed_sessions),
        "sessions": detailed_sessions
    }


@router.get("/pronunciation_vocab_sets")
async def get_pronunciation_vocab_sets():
   """Get all unique set numbers from pronunciation vocabulary"""
   async with async_session() as sess:
       result = await sess.execute(
           text("SELECT DISTINCT set_number FROM pronunciation_vocab WHERE set_number IS NOT NULL ORDER BY set_number")
       )
       rows = result.fetchall()

       set_numbers = [row[0] for row in rows]

       return {
           "status": "success",
           "total": len(set_numbers),
           "set_numbers": set_numbers
       }
