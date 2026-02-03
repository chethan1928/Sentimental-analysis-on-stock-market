import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from typing import Optional
from faster_whisper import WhisperModel
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from deep_translator import GoogleTranslator
from openai import AzureOpenAI
from pydub import AudioSegment

from utils.tts_utils import analyze_speaking_advanced, load_language_mapping, normalize_language_code
from ai_fluent import speech_to_text
from db import chat_session_db as db

# from utils.common_utils import llm_client

# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


_whisper_model = WhisperModel("small", compute_type="int8")


logger = logging.getLogger(__name__)


router = APIRouter()

QWEN_ENABLED = False
qwen_client = None  
QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B" 


PASSING_SCORE = 50
TERMINATION_PHRASES = ["exit", "stop", "end", "finish", "quit", "done", "bye", "goodbye"]


LEVEL_KEYWORDS = {
    "beginner": "B1",
    "basic": "B1",
    "starter": "B1",
    "elementary": "B1",
    "intermediate": "B2",
    "medium": "B2",
    "moderate": "B2",
    "advanced": "C1",
    "fluent": "C1",
    "experienced": "C1",
    "proficient": "C2",
    "expert": "C2",
    "native": "C2",
    "master": "C2"
}


LEVEL_DISPLAY = {
    "B1": "Beginner",
    "B2": "Intermediate",
    "C1": "Advanced",
    "C2": "Proficient"
}

KNOWLEDGE_BASE = """
- Conversational Practice: Engage in real-time conversations on various topics.
- Vocabulary Building: Introduce new words and phrases with definitions.
- Grammar Exercises: Offer explanations and practice for grammatical structures.
- Pronunciation Practice: Provide feedback on pronunciation.
- Provide immediate, constructive feedback on the user's language use.
- Praise progress and highlight improvements.
"""

# Language-specific rules for grammar, vocabulary, and pronunciation analysis
# NOTE: English rules are already in the prompts, so only non-English languages are defined here
# These are REFERENCE guidelines for the LLM, not strict rules
LANGUAGE_RULES = {
    "de": {
        "name": "German",
        "grammar": {
            "articles": ["der (masc)", "die (fem)", "das (neut)", "ein/eine"],
            "cases": ["Nominativ (subject)", "Akkusativ (direct object)", "Dativ (indirect object)", "Genitiv (possession)"],
            "common_errors": [
                {"error": "wrong gender", "example": "die Tisch → der Tisch", "tip": "Memorize noun genders with the article"},
                {"error": "case errors", "example": "Ich sehe der Mann → Ich sehe den Mann", "tip": "Akkusativ changes der→den"},
                {"error": "verb position", "example": "Ich heute gehe → Ich gehe heute", "tip": "Verb must be 2nd element in main clause"},
                {"error": "separable verbs", "example": "Ich aufstehe → Ich stehe auf", "tip": "Prefix goes to end of clause"},
                {"error": "subordinate clause order", "example": "weil ich gehe → weil ich gehe (verb at end)", "tip": "In subordinate clauses, verb goes to the end"}
            ],
            "word_order": "V2 (verb second) in main clause, verb-final in subordinate clauses",
            "verb_conjugation": "Conjugates for person (ich/du/er), number (singular/plural), tense (Präsens/Präteritum/Perfekt), mood (Indikativ/Konjunktiv)",
            "special_rules": [
                "All nouns are capitalized",
                "Verb is always the second element in declarative sentences",
                "Modal verbs push main verb to the end in infinitive form",
                "Adjective endings change based on article type and case"
            ],
            "example_sentences": [
                {"wrong": "Ich habe gestern ein Buch gekauft.", "correct": "Ich habe gestern ein Buch gekauft.", "note": "Participle at end"},
                {"wrong": "Der Mann, der ich sehe", "correct": "Der Mann, den ich sehe", "note": "Relative pronoun case"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["du (informal/friends)", "Sie (formal/strangers/business)"],
            "compound_words": "Very common - combine words: Handschuh (hand+shoe = glove)",
            "common_collocations": ["eine Entscheidung treffen (make a decision)", "Pause machen (take a break)", "Bescheid sagen (let know)"],
            "false_friends": ["bekommen ≠ become (means 'to get')", "Gift ≠ gift (means 'poison')"],
            "separable_prefixes": ["auf-, an-, aus-, ein-, mit-, vor-, zu-, ab-, weg-"],
            "register_markers": ["Formal: würden Sie, könnten Sie", "Informal: kannst du, willst du"]
        },
        "pronunciation": {
            "stress_pattern": "Usually on first syllable, except for loanwords and prefixed verbs",
            "difficult_sounds": [
                "ü [y] - round lips like 'oo', say 'ee'",
                "ö [ø] - round lips like 'o', say 'e'",
                "ch [ç/x] - after front vowels soft (ich), after back vowels hard (ach)",
                "r - uvular/throat sound, not like English",
                "ß - always voiceless 's' sound"
            ],
            "umlauts": ["ä = 'e' sound", "ö = between 'o' and 'e'", "ü = between 'u' and 'i'"],
            "final_devoicing": "Final b/d/g sound like p/t/k",
            "intonation": "Generally flatter than English, rises for yes/no questions"
        }
    },
    "fr": {
        "name": "French",
        "grammar": {
            "articles": ["le (masc sing)", "la (fem sing)", "les (plural)", "un/une (indefinite)", "du/de la (partitive)"],
            "common_errors": [
                {"error": "wrong gender", "example": "le table → la table", "tip": "Most -tion/-sion words are feminine"},
                {"error": "agreement errors", "example": "les petit filles → les petites filles", "tip": "Adjectives agree in gender AND number"},
                {"error": "wrong preposition", "example": "penser à → penser de", "tip": "Prepositions are verb-specific"},
                {"error": "past tense auxiliary", "example": "J'ai allé → Je suis allé", "tip": "Motion/state change verbs use être"},
                {"error": "negation structure", "example": "Je ne sais → Je ne sais pas", "tip": "Ne...pas wraps around verb"}
            ],
            "word_order": "Subject-Verb-Object, most adjectives AFTER noun (une maison grande)",
            "verb_conjugation": "Complex: person, number, tense (8+ tenses), mood (indicatif/subjonctif/conditionnel)",
            "special_rules": [
                "Adjectives agree in gender and number with noun",
                "Liaison: final consonant pronounced before vowel",
                "Partitive articles: du pain (some bread), de l'eau (some water)",
                "Object pronouns go BEFORE the verb (Je le vois)"
            ],
            "example_sentences": [
                {"wrong": "Je suis allé à le magasin", "correct": "Je suis allé au magasin", "note": "à + le = au"},
                {"wrong": "Il faut que je vais", "correct": "Il faut que j'aille", "note": "Subjunctive after 'il faut que'"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["tu (informal/friends)", "vous (formal/plural)"],
            "false_friends": ["actuellement = currently (not actually)", "librairie = bookstore (not library)", "assister = attend (not assist)"],
            "common_collocations": ["prendre une décision", "faire une pause", "avoir l'air (to seem)", "faire semblant (pretend)"],
            "register_markers": ["Formal: Veuillez..., Je vous prie de...", "Informal: T'inquiète, Ça va?"],
            "liaisons_importantes": ["les amis [lez-ami]", "nous avons [nuz-avõ]"]
        },
        "pronunciation": {
            "stress_pattern": "ALWAYS on last syllable of phrase/word group",
            "silent_letters": [
                "Final consonants usually silent (petit, grand, français)",
                "h is always silent (l'hôtel, l'homme)",
                "e at end often silent (je parle)"
            ],
            "difficult_sounds": [
                "r [ʁ] - uvular/throat, not rolled",
                "Nasal vowels: on [õ], an/en [ɑ̃], in [ɛ̃], un [œ̃]",
                "u [y] vs ou [u] - both are different",
                "eu [ø/œ] - rounded front vowel"
            ],
            "liaison": "Connect final consonant to next vowel (les enfants = lez-enfants)",
            "enchainement": "Final pronounced consonant links to next vowel",
            "intonation": "Rising at end of phrases, falling at end of statements"
        }
    },
    "ja": {
        "name": "Japanese",
        "grammar": {
            "articles": [],
            "particles": ["は (wa) topic", "が (ga) subject", "を (wo) object", "に (ni) direction/time", "で (de) location/means", "と (to) with/and"],
            "common_errors": [
                {"error": "particle confusion", "example": "私を学校に行く → 私は学校に行く", "tip": "は marks topic, を marks direct object"},
                {"error": "verb form mixing", "example": "食べるました → 食べました", "tip": "Don't mix plain and polite forms"},
                {"error": "counter errors", "example": "三つ人 → 三人", "tip": "人 has special counter (にん/人)"},
                {"error": "honorific levels", "example": "先生が言った → 先生がおっしゃった", "tip": "Use humble/respectful forms appropriately"},
                {"error": "は vs が confusion", "example": "猫は好きです (cats in general) vs 猫が好きです (emphasis)", "tip": "は = topic/contrast, が = subject/new info"}
            ],
            "word_order": "Subject-Object-Verb (SOV) - verb ALWAYS at the end",
            "verb_conjugation": "Conjugates for tense, politeness, and aspect (not person/number)",
            "special_rules": [
                "Particles mark grammatical function (like prepositions but after noun)",
                "Verb is always at the end of the sentence",
                "No plural forms for nouns (context determines)",
                "Subjects/topics often omitted if understood from context",
                "Three politeness levels: casual, polite (です/ます), formal (keigo)"
            ],
            "example_sentences": [
                {"context": "Polite", "example": "私は学生です。", "note": "です for polite copula"},
                {"context": "Casual", "example": "俺は学生だ。", "note": "だ for casual copula"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["casual (だ/る)", "polite (です/ます)", "humble (謙譲語)", "respectful (尊敬語)"],
            "writing_systems": ["Hiragana (native words)", "Katakana (foreign words/emphasis)", "Kanji (Chinese characters)"],
            "counters": "Use specific counters: 人(people), 本(long things), 枚(flat things), 匹(small animals)",
            "onomatopoeia": "Very common: ワクワク(excited), ドキドキ(nervous), フワフワ(fluffy)",
            "loan_words": "Many from English written in katakana: パソコン, コーヒー"
        },
        "pronunciation": {
            "stress_pattern": "Pitch accent (high-low patterns) - pitch changes meaning",
            "difficult_sounds": [
                "Long vowels (おばさん aunt vs おばあさん grandmother)",
                "Double consonants (きて come vs きって stamp)",
                "r - single tap, between 'r' and 'l'",
                "Pitch accent: 橋 (hashi, low-high = chopsticks) vs 箸 (hashi, high-low = bridge)"
            ],
            "mora_timing": "Each mora (hiragana) has equal length",
            "vowel_devoicing": "i and u often whispered between voiceless consonants",
            "intonation": "Relatively flat, pitch accent is lexical"
        }
    },
    "zh": {
        "name": "Chinese (Mandarin)",
        "grammar": {
            "articles": [],
            "measure_words": ["个 (general)", "本 (books)", "张 (flat things)", "条 (long things)", "只 (animals)"],
            "common_errors": [
                {"error": "missing measure word", "example": "三书 → 三本书", "tip": "Measure word required between number and noun"},
                {"error": "wrong aspect marker", "example": "我吃饭 → 我吃了饭", "tip": "了 for completed, 过 for experience, 着 for ongoing"},
                {"error": "word order in questions", "example": "哪里你去？ → 你去哪里？", "tip": "Question word stays in place"},
                {"error": "complement errors", "example": "学好 vs 学得好", "tip": "得 introduces result/degree complements"},
                {"error": "把 construction", "example": "我打破了杯子 → 我把杯子打破了", "tip": "把 moves object before verb for disposal/result"}
            ],
            "word_order": "Subject-Verb-Object (SVO), Time-Place order (big to small)",
            "verb_conjugation": "NO conjugation - use aspect markers (了, 过, 着) and time words",
            "special_rules": [
                "No verb conjugation at all",
                "Measure words are REQUIRED between numbers and nouns",
                "Topic-prominent: what you're talking about comes first",
                "Verb complements show result/degree (看得见 = can see)",
                "Serial verb constructions are common"
            ],
            "example_sentences": [
                {"example": "我昨天买了三本书", "analysis": "Subject + Time + Verb + 了 + Number + Measure + Noun"},
                {"example": "他跑得很快", "analysis": "Degree complement with 得"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["casual (口语)", "formal (书面语)", "literary (文言)"],
            "measure_words": "Must use correct classifier for each noun category",
            "chengyu": "Four-character idioms are important (成语): 一举两得, 画蛇添足",
            "homophones": "Many due to limited syllables (~400) - context is crucial",
            "register": ["Formal: 请问, 您 | Casual: 问一下, 你"]
        },
        "pronunciation": {
            "tones": [
                "1st tone (ˉ): high flat - 妈 (mā) mother",
                "2nd tone (ˊ): rising - 麻 (má) hemp",
                "3rd tone (ˇ): dipping - 马 (mǎ) horse",
                "4th tone (ˋ): falling - 骂 (mà) scold",
                "Neutral: short, unstressed - 吗 (ma) question particle"
            ],
            "difficult_sounds": [
                "zh/ch/sh (retroflex) vs z/c/s (alveolar)",
                "ü [y] - round lips, say 'ee'",
                "Retroflex finals: er, 儿化",
                "j/q/x (palatal sounds)"
            ],
            "tone_sandhi": "Two 3rd tones → first becomes 2nd tone (你好 = ní hǎo)",
            "intonation": "Tones are LEXICAL (change meaning), not just intonational"
        }
    },
    "hi": {
        "name": "Hindi",
        "grammar": {
            "articles": [],
            "postpositions": ["में (in)", "पर (on)", "से (from/with)", "को (to/object marker)", "के लिए (for)"],
            "common_errors": [
                {"error": "gender agreement", "example": "बड़ा लड़की → बड़ी लड़की", "tip": "Adjectives agree with noun gender"},
                {"error": "postposition errors", "example": "घर को जाना → घर जाना", "tip": "को not needed with जाना for destinations"},
                {"error": "verb agreement", "example": "लड़की गया → लड़की गई", "tip": "Verb agrees with subject in gender"},
                {"error": "honorific forms", "example": "आप क्या चाहते हो → आप क्या चाहते हैं", "tip": "आप takes हैं, not हो"},
                {"error": "ergative case", "example": "मैंने किताब पढ़ा → मैंने किताब पढ़ी", "tip": "In perfective tense, verb agrees with object if subject has ने"}
            ],
            "word_order": "Subject-Object-Verb (SOV) - verb at the end",
            "verb_conjugation": "Agrees with subject in gender, number, person; also honorific level",
            "special_rules": [
                "Postpositions (not prepositions) - come AFTER noun",
                "Verb is always at the end",
                "All nouns have gender (masculine/feminine)",
                "Ergative construction: ने marks agent in perfective",
                "Three levels of respect: तू < तुम < आप"
            ],
            "example_sentences": [
                {"example": "मैं स्कूल जाता हूँ", "analysis": "I school go-MASC am (male speaker)"},
                {"example": "मैं स्कूल जाती हूँ", "analysis": "I school go-FEM am (female speaker)"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["तू (very intimate)", "तुम (informal/friends)", "आप (formal/respect)"],
            "sanskrit_influence": "Formal/literary uses Sanskrit words (शिक्षा, विद्यालय)",
            "urdu_influence": "Everyday speech uses Persian/Arabic words (दोस्त, खबर)",
            "honorifics": "जी suffix for respect (जी, साहब, श्री, श्रीमती)",
            "echo_words": "Reduplication: खाना-वाना (food and stuff), चाय-वाय"
        },
        "pronunciation": {
            "stress_pattern": "Generally on second-to-last syllable; longer syllables get stress",
            "difficult_sounds": [
                "Aspirated vs unaspirated: क [k] vs ख [kʰ], प [p] vs फ [pʰ]",
                "Retroflex sounds: ट, ठ, ड, ढ, ण (tongue curled back)",
                "Nasal vowels: माँ, हाँ (with ँ/ं)",
                "ड़ and ढ़ - flapped retroflex"
            ],
            "schwa_deletion": "Final inherent 'a' vowel often not pronounced (राम = raam, not raama)",
            "gemination": "Double consonants are longer: पका (ripe) vs पक्का (firm)",
            "nasalization": "Vowels can be nasalized with chandrabindu (ँ)"
        }
    },
    "te": {
        "name": "Telugu",
        "grammar": {
            "articles": [],
            "case_suffixes": ["ని (accusative)", "కు/కి (dative)", "లో (locative)", "తో (instrumental)", "నుంచి (ablative)"],
            "common_errors": [
                {"error": "case suffix errors", "example": "నేను ఇంటికి వెళ్ళాను → proper case endings", "tip": "Use correct case suffix for each function"},
                {"error": "verb agreement", "example": "అతను వచ్చింది → అతను వచ్చాడు", "tip": "Verb ending must match subject gender/number"},
                {"error": "sandhi errors", "example": "Word junction rules", "tip": "Consonant/vowel changes at word boundaries"},
                {"error": "respectful forms", "example": "వచ్చారు vs వచ్చాడు", "tip": "Use -ారు suffix for respect"},
                {"error": "tense markers", "example": "Mixing simple past with other forms", "tip": "Past: -ాను/-ావు/-ాడు, Future: -తాను/-తావు"}
            ],
            "word_order": "Subject-Object-Verb (SOV) - verb always at end",
            "verb_conjugation": "Agrees with subject in person, number, gender, and respect level",
            "special_rules": [
                "Agglutinative: suffixes stack on words",
                "Sandhi: sound changes at word boundaries",
                "Case suffixes mark grammatical relations",
                "Verb at end of sentence",
                "Extensive use of participles and complex clauses"
            ],
            "example_sentences": [
                {"example": "నేను పుస్తకం చదువుతున్నాను", "analysis": "I book reading-am (present continuous)"},
                {"example": "అతను బడికి వెళ్ళాడు", "analysis": "He school-to went (male subject)"}
            ]
        },
        "vocabulary": {
            "formality_levels": ["నువ్వు (informal)", "మీరు (formal/plural)"],
            "sanskrit_influence": "Literary/formal uses Sanskrit: విద్య (knowledge), గ్రంథాలయం (library)",
            "honorific_suffixes": ["గారు (respect)", "వారు (high respect)"],
            "native_vs_borrowed": "తత్సమ (direct Sanskrit) vs తద్భవ (adapted native forms)",
            "echo_words": "Reduplication common: అన్నం-గిన్నం (rice and such)"
        },
        "pronunciation": {
            "stress_pattern": "Generally on first syllable of word",
            "difficult_sounds": [
                "Retroflex consonants: ట, డ, ణ (tongue curled back)",
                "Aspirated sounds: ఖ, ఛ, థ, ఫ (with breath)",
                "ళ [ɭ] - retroflex lateral (unique to Dravidian)",
                "ఱ - older retroflex flap (rare now)"
            ],
            "vowel_length": "Short vs long vowels change meaning: పలు (teeth) vs పాలు (milk)",
            "gemination": "Double consonants are important: కల (dream) vs కళ్ళ (eyes)",
            "sandhi_phonetics": "Sound changes occur at morpheme/word boundaries"
        }
    }
}

def get_language_rules(lang_code: str) -> dict:
    """Get language-specific rules for a given language code. Returns None for English."""
    lang_lower = lang_code.lower()
    # Map common language names/codes to our keys
    lang_mapping = {
        "english": "en", "en": "en",
        "german": "de", "de": "de", "deutsch": "de",
        "french": "fr", "fr": "fr", "français": "fr", "francais": "fr",
        "japanese": "ja", "ja": "ja", "jp": "ja", "日本語": "ja",
        "chinese": "zh", "zh": "zh", "mandarin": "zh", "中文": "zh",
        "hindi": "hi", "hi": "hi", "हिंदी": "hi",
        "telugu": "te", "te": "te", "తెలుగు": "te"
    }
    key = lang_mapping.get(lang_lower, "en")
    # Return None for English (rules already in prompts)
    if key == "en":
        return None
    return LANGUAGE_RULES.get(key)

 


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


BOT_NAME = "sara"
BOT_ROLE = "language tutor"


async def analyze_fluency_metrics(user_text: str, audio_duration: float) -> dict:
    """async wrapper for fluency metrics from text and duration"""
    
    word_count = len(re.findall(r"\b\w+\b", user_text or ""))
    return calculate_fluency(word_count, audio_duration)


async def compare_attempts(attempts: list, level: str = "B1", user_type: str = "student", model: str = "gpt", target_language: str = "en") -> dict:
    """
    Compare attempts using LLM for detailed, elaborative feedback on ALL aspects:
    grammar, vocabulary, pronunciation, and fluency.
    """
    if len(attempts) < 2:
        return {
            "improvement": 0,
            "trend": "first_attempt",
            "message": "This is your first attempt. Let's see how you do!",
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
    
    prev_overall = prev.get("overall_score", 0) or 0
    current_overall = current.get("overall_score", 0) or 0
    
    
    grammar_diff = round(current_grammar - prev_grammar, 1)
    vocab_diff = round(current_vocab - prev_vocab, 1)
    pron_diff = round(current_pron - prev_pron, 1)
    fluency_diff = round(current_fluency - prev_fluency, 1)
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
    
    
    prev_grammar_errors = (prev.get("grammar") or {}).get("errors", [])[:3]
    current_grammar_errors = (current.get("grammar") or {}).get("errors", [])[:3]
    prev_vocab_suggestions = (prev.get("vocabulary") or {}).get("suggestions", [])[:3]
    current_vocab_suggestions = (current.get("vocabulary") or {}).get("suggestions", [])[:3]
    prev_words_to_practice = (prev.get("pronunciation") or {}).get("words_to_practice", [])[:5]
    current_words_to_practice = (current.get("pronunciation") or {}).get("words_to_practice", [])[:5]
    
    prompt = f"""You are an expert language coach comparing TWO attempts at the SAME question.
Provide DETAILED, ELABORATIVE feedback on improvement or decline in ALL areas.

PREVIOUS ATTEMPT:
- Overall Score: {prev_overall}%
- Grammar: {prev_grammar}% (Errors: {[e.get('you_said', '') for e in prev_grammar_errors if isinstance(e, dict)]})
- Vocabulary: {prev_vocab}% (Weak words: {[s.get('word', '') for s in prev_vocab_suggestions if isinstance(s, dict)]})
- Pronunciation: {prev_pron}% (Needs practice: {[w.get('word', w) if isinstance(w, dict) else w for w in prev_words_to_practice]})
- Fluency: {prev_fluency}% (WPM: {(prev.get('fluency') or {}).get('wpm', 0)})
- What they said: "{prev.get('transcription', '')[:200]}"

CURRENT ATTEMPT:
- Overall Score: {current_overall}%
- Grammar: {current_grammar}% ({'+' if grammar_diff > 0 else ''}{grammar_diff}%)
- Vocabulary: {current_vocab}% ({'+' if vocab_diff > 0 else ''}{vocab_diff}%)
- Pronunciation: {current_pron}% ({'+' if pron_diff > 0 else ''}{pron_diff}%)
- Fluency: {current_fluency}% ({'+' if fluency_diff > 0 else ''}{fluency_diff}%)
- What they said: "{current.get('transcription', '')[:200]}"

USER CONTEXT:
- Level: {level}
- User Type: {user_type}

Analyze EACH category's improvement and provide detailed, encouraging feedback.

Return STRICTLY valid JSON:
{{
    "overall_summary": "3-4 sentences summarizing the overall improvement journey. Be specific about what changed.",
    "grammar_analysis": {{
        "previous_score": {prev_grammar},
        "current_score": {current_grammar},
        "difference": {grammar_diff},
        "improved": {str(grammar_diff > 0).lower()},
        "feedback": "Specific feedback about grammar improvement or what still needs work. Mention specific errors fixed or remaining."
    }},
    "vocabulary_analysis": {{
        "previous_score": {prev_vocab},
        "current_score": {current_vocab},
        "difference": {vocab_diff},
        "improved": {str(vocab_diff > 0).lower()},
        "feedback": "Specific feedback about vocabulary improvement. Mention better word choices used or areas to improve."
    }},
    "pronunciation_analysis": {{
        "previous_score": {prev_pron},
        "current_score": {current_pron},
        "difference": {pron_diff},
        "improved": {str(pron_diff > 0).lower()},
        "words_improved": ["words that sound better now"],
        "still_needs_work": ["words still needing practice"],
        "feedback": "Specific feedback about pronunciation. What words improved? What still needs practice?"
    }},
    "fluency_analysis": {{
        "previous_score": {prev_fluency},
        "current_score": {current_fluency},
        "difference": {fluency_diff},
        "improved": {str(fluency_diff > 0).lower()},
        "feedback": "Specific feedback about speaking pace and flow. Did they speak more naturally?"
    }},
    "biggest_improvement": "Which area improved the most and by how much",
    "area_needing_focus": "Which area still needs the most work",
    "encouragement": "A warm, personalized encouraging message mentioning specific progress",
    "next_step_tip": "One specific, actionable tip for continued improvement"
}}

Be ELABORATIVE. Don't just say 'improved' - explain HOW and WHAT specifically."""

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
            "grammar_analysis": {"previous_score": prev_grammar, "current_score": current_grammar, "difference": grammar_diff, "improved": grammar_diff > 0, "feedback": f"Grammar {'improved' if grammar_diff > 0 else 'needs more focus'} ({'+' if grammar_diff > 0 else ''}{grammar_diff}%)"},
            "vocabulary_analysis": {"previous_score": prev_vocab, "current_score": current_vocab, "difference": vocab_diff, "improved": vocab_diff > 0, "feedback": f"Vocabulary {'improved' if vocab_diff > 0 else 'needs more focus'} ({'+' if vocab_diff > 0 else ''}{vocab_diff}%)"},
            "pronunciation_analysis": {"previous_score": prev_pron, "current_score": current_pron, "difference": pron_diff, "improved": pron_diff > 0, "words_improved": [], "still_needs_work": [], "feedback": f"Pronunciation {'improved' if pron_diff > 0 else 'needs more focus'} ({'+' if pron_diff > 0 else ''}{pron_diff}%)"},
            "fluency_analysis": {"previous_score": prev_fluency, "current_score": current_fluency, "difference": fluency_diff, "improved": fluency_diff > 0, "feedback": f"Fluency {'improved' if fluency_diff > 0 else 'needs more focus'} ({'+' if fluency_diff > 0 else ''}{fluency_diff}%)"},
            "biggest_improvement": "grammar" if grammar_diff == max(grammar_diff, vocab_diff, pron_diff, fluency_diff) else "vocabulary" if vocab_diff == max(grammar_diff, vocab_diff, pron_diff, fluency_diff) else "pronunciation",
            "area_needing_focus": "grammar" if grammar_diff == min(grammar_diff, vocab_diff, pron_diff, fluency_diff) else "vocabulary",
            "encouragement": f"Keep practicing! Your overall score {'improved' if overall_diff > 0 else 'stayed consistent'}.",
            "next_step_tip": "Focus on speaking slowly and clearly."
        }
    
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
        "biggest_improvement": llm_data.get("biggest_improvement", ""),
        "area_needing_focus": llm_data.get("area_needing_focus", ""),
        "encouragement": llm_data.get("encouragement", ""),
        "next_step_tip": llm_data.get("next_step_tip", "")
    }


async def call_llm(prompt: str, mode: str = "chat", timeout: int = 30, model: str = "gpt", target_language: str = "en") -> str:
    """async llm call with proper error handling and timeout. Supports gpt (default) or qwen."""
    # Base system prompts
    base_prompts = {
        "chat": "You are a kind, human-like language tutor helping users practice conversational skills.",
        "analysis": "You are an expert language evaluator. Analyze objectively and concisely.",
        "strict_json": "You are a structured evaluator. Respond ONLY in valid JSON. No extra text."
    }
    
    # Add language instruction if not English
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



GRAMMAR_FIELDS = ["feedback", "filler_feedback", "errors", "word_suggestions", "corrected_sentence", "improved_sentence"]
VOCAB_FIELDS = ["feedback", "suggestions"]
PRON_FIELDS = ["feedback", "words_to_practice"]
FLUENCY_FIELDS = ["feedback"]
PERSONAL_FIELDS = ["message", "improvement_areas", "strengths"]


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


async def transcribe_audio_file(audio_file, target_lang: str = "en") -> str:
    """Helper function to transcribe audio file using Whisper with auto-detection and translation"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        temp_upload = tmp.name
    
    audio_path = None
    try:
        
        def convert_audio():
            audio = AudioSegment.from_file(temp_upload)
            audio = audio.set_frame_rate(16000).set_channels(1)
            converted_path = temp_upload.replace('.tmp', '_converted.wav')
            audio.export(converted_path, format="wav")
            return converted_path
        
        audio_path = await asyncio.to_thread(convert_audio)
        os.unlink(temp_upload)  
        
        # Normalize target language code
        languages_data = load_language_mapping()
        normalized_target = languages_data.get(target_lang.lower(), target_lang.lower()) if target_lang else "en"
        
        # Auto-detect language from audio (don't force language)
        segments, info = await asyncio.to_thread(_whisper_model.transcribe, audio_path, task="transcribe")
        user_text = " ".join([seg.text for seg in segments]).strip()
        
        # Get detected language from Whisper
        detected_lang = info.language if info else "en"
        logger.debug(f"Whisper detected language: {detected_lang}, target: {normalized_target}")
        
        # If detected language doesn't match target, translate to target language
        if user_text and detected_lang != normalized_target:
            try:
                translated = await asyncio.to_thread(
                    GoogleTranslator(source=detected_lang, target=normalized_target).translate,
                    user_text
                )
                if translated:
                    logger.debug(f"Translated from {detected_lang} to {normalized_target}: {user_text[:50]} -> {translated[:50]}")
                    user_text = translated
            except Exception as e:
                logger.debug(f"Translation to {normalized_target} failed: {e}")
        
        return user_text
    except Exception as e:
        logger.debug(f"Audio transcription failed: {e}")
        return ""
    finally:
        
        for path in [temp_upload, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

SCENARIO_KEYWORDS = {
    "restaurant": "ordering_food", "food": "ordering_food", "eat": "ordering_food", "menu": "ordering_food",
    "hotel": "hotel_checkin", "room": "hotel_checkin", "reservation": "hotel_checkin", "book": "hotel_checkin",
    "travel": "travel", "trip": "travel", "vacation": "travel", "airport": "travel", "flight": "travel",
    "shopping": "shopping", "buy": "shopping", "store": "shopping", "shop": "shopping",
    "work": "workplace", "office": "workplace", "job": "workplace", "meeting": "workplace", "colleague": "workplace",
    "doctor": "medical", "hospital": "medical", "health": "medical", "sick": "medical",
    "phone": "phone_call", "call": "phone_call", "appointment": "phone_call",
    "direction": "asking_directions", "lost": "asking_directions", "way": "asking_directions", "find": "asking_directions",
    "introduce": "self_introduction", "myself": "self_introduction", "about me": "self_introduction",
    "casual": "casual_conversation", "chat": "casual_conversation", "talk": "casual_conversation", "friend": "casual_conversation",
    "daily": "daily_routine", "everyday": "daily_routine", "routine": "daily_routine"
}


async def extract_scenario_from_text(user_text: str, model: str = "gpt") -> dict:
    """Extract conversation scenario from natural language using fuzzy matching + LLM"""
    user_lower = user_text.lower()
    
    
    for keyword, scenario in SCENARIO_KEYWORDS.items():
        if keyword in user_lower:
            return {"success": True, "scenario": scenario, "confidence": "high"}
    
    
    prompt = f"""Extract the conversation practice scenario from: "{user_text}"

IMPORTANT: Accept ANY topic the user mentions for conversation practice.
Examples: ordering food, hotel check-in, shopping, making appointments, casual talk, daily routine, travel, work, etc.

If user mentions ANY topic, extract and format it:
- Return: {{"success": true, "scenario": "extracted_scenario_in_lowercase", "confidence": "high"}}
- Example: "I want to practice ordering food" → {{"success": true, "scenario": "ordering_food", "confidence": "high"}}
- Example: "let's talk about travel" → {{"success": true, "scenario": "travel", "confidence": "high"}}

If completely unclear:
- Return: {{"success": false, "scenario": "casual_conversation", "confidence": "low"}}

Return ONLY valid JSON."""
    
    try:
        raw = await call_llm(prompt, mode="strict_json", timeout=10, model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            if result.get("scenario"):
                result["success"] = True
            return result
    except:
        pass
    return {"success": True, "scenario": "casual_conversation", "confidence": "low"}


async def extract_level_from_text(user_text: str, model: str = "gpt") -> dict:
    """Extract language level from natural language - returns CEFR codes (B1, B2, C1, C2)"""
    user_lower = user_text.lower()
    
    
    for keyword, level in LEVEL_KEYWORDS.items():
        if keyword in user_lower:
            return {"success": True, "level": level, "confidence": "high"}
    
    
    cefr_codes = {
        "b1": "B1", "b2": "B2",
        "c1": "C1", "c2": "C2"
    }
    for cefr, level in cefr_codes.items():
        if cefr in user_lower:
            return {"success": True, "level": level, "confidence": "high"}
    
    
    prompt = f"""Extract the language proficiency level from: "{user_text}"

Map to these CEFR codes ONLY (B1 is the minimum, no A1/A2):
- B1 (Beginner): Can handle familiar situations, routine conversations
- B2 (Intermediate): Independent user, can discuss abstract topics
- C1 (Advanced): Fluent, can express complex ideas spontaneously
- C2 (Proficient): Native-like mastery, nuanced and precise

Return: {{"success": true, "level": "B2", "confidence": "high"}}
If unclear, default to B2: {{"success": true, "level": "B2", "confidence": "low"}}

Return ONLY valid JSON."""
    
    try:
        raw = await call_llm(prompt, mode="strict_json", timeout=10, model=model)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return {"success": True, "level": "B2", "confidence": "low"}


async def generate_context_aware_follow_up(user_response: str, chat_history: list, scenario: str, user_type: str, model: str = "gpt", target_language: str = "en") -> tuple:
    """Generate context-aware follow-up question with natural transitions"""
    
    recent_chat = chat_history[-6:] if chat_history else []
    chat_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_chat])
    
    prompt = f"""You are {BOT_NAME}, a warm and friendly language tutor practicing {scenario} conversations.

CONVERSATION SO FAR:
{chat_context}

The learner just said: "{user_response}"

IMPORTANT RULES:
1. YOU (the bot) ALWAYS ask the questions
2. The USER ALWAYS answers/responds
3. This is a REAL CHAT - remember everything from the conversation
4. Start with an interactive reaction to their answer (be specific!)
5. Then ask a follow-up question they can ANSWER
6. NEVER ask them to ask YOU something
7. NEVER repeat questions already asked

Return STRICTLY valid JSON:
{{"question": "[your interactive reaction + follow-up question for user to answer]", "hint": "[example response they could give]"}}"""

    try:
        raw = await call_llm(prompt, mode="strict_json", model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", "Tell me more about that."), data.get("hint", "Share more details.")
    except:
        pass
    return "Tell me more about that.", "Share more details."


async def generate_retry_encouragement(scenario: str, retry_count: int, previous_score: int, user_name: str = "there", model: str = "gpt", target_language: str = "en") -> str:
    """Generate friendly, encouraging retry message using LLM"""
    prompt = f"""You are a warm, encouraging language tutor.

Context:
- Student: {user_name}
- Scenario: {scenario}
- Previous score: {previous_score}%
- Retry attempt: 

Generate a SHORT (1-2 sentences) encouraging message for retrying. Be:
- Warm and supportive (like a friend, not a teacher)
- Specific about the scenario if possible
- Use an emoji occasionally 
- Never make them feel bad about their score

Examples:
- "No worries! Let's give it another shot. Take a deep breath and try again! 💪"
- "You've got this! Practice makes perfect. Let's try once more."
- "Great attitude wanting to improve! Ready when you are 😊"

Return ONLY the message, no JSON."""
    
    try:
        msg = await call_llm(prompt, timeout=10, model=model, target_language=target_language)
        if msg and len(msg) < 200:  
            return msg.strip('\"\'')
    except:
        pass
    
    
    fallbacks = [
        f"No problem, {user_name}! Let's try this again. Take your time! 💪",
        "Great attitude! Practice makes perfect. Ready when you are 😊",
        "You've got this! Let's give it another shot.",
        "No worries! Take a breath and try again. I believe in you!"
    ]
    import random
    return random.choice(fallbacks)


async def generate_skip_message(scenario: str, user_name: str = "there", model: str = "gpt", target_language: str = "en") -> str:
    """Generate friendly skip/transition message using LLM"""
    prompt = f"""You are a warm, friendly language tutor.

The student {user_name} wants to skip to the next question in {scenario} practice.

Generate a SHORT (1 sentence) friendly transition message that:
- Doesn't make them feel bad for skipping
- Keeps energy positive
- Maybe adds a small emoji

Examples:
- "Absolutely! Let's move on to something fresh ✨"
- "Sure thing! Here's a new one for you."
- "No problem! Let's try a different question."

Return ONLY the message, no JSON."""
    
    try:
        msg = await call_llm(prompt, timeout=8, model=model, target_language=target_language)
        if msg and len(msg) < 150:
            return msg.strip('\"\'')
    except:
        pass
    return "No problem! Let's try something new 😊"


async def analyze_grammar_llm(user_text: str, level: str = "Intermediate", user_type: str = "student", model: str = "gpt", target_language: str = "en") -> dict:
    """llm-based grammar analysis for spoken language with detailed suggestions"""
    
    level_context = ""
    if level == "Beginner":
        level_context = f"User is at Beginner level. Focus on basic grammar errors and simple corrections."
    elif level == "Intermediate":
        level_context = f"User is at Intermediate level. Check for intermediate grammar issues and provide detailed explanations."
    elif level == "Advanced":
        level_context = f"User is at Advanced level. Focus on subtle grammar nuances and advanced corrections."
    else:
        level_context = f"User is at Proficient level. Focus on native-like polish and professional refinement."
    
    user_type_context = ""
    if user_type == "professional":
        user_type_context = "This is a professional user. Provide business-appropriate grammar feedback."
    elif user_type == "student":
        user_type_context = "This is a student. Provide educational grammar feedback with clear explanations."
    
    # Language instruction for response
    lang_lower = target_language.lower() if target_language else "en"
    is_english = lang_lower in ["en", "english"]
    lang_instruction = f"IMPORTANT: Provide ALL feedback, explanations, and suggestions in {target_language} language." if not is_english else ""
    
    # Get language-specific grammar rules
    lang_rules = get_language_rules(target_language)
    lang_rules_text = ""
    if lang_rules:
        grammar_rules = lang_rules.get("grammar", {})
        common_errors = grammar_rules.get('common_errors', [])
        error_lines = []
        for e in common_errors:
            tip = f" (Tip: {e.get('tip', '')})" if e.get('tip') else ""
            error_lines.append(f"      * {e.get('error', '')}: {e.get('example', '')}{tip}")
        
        lang_rules_text = f"""
    LANGUAGE-SPECIFIC REFERENCE GUIDELINES FOR {lang_rules.get('name', target_language).upper()}:
    (Use these as REFERENCE - adapt based on actual user text, not strict rules)
    
    - Word Order: {grammar_rules.get('word_order', '')}
    - Articles: {grammar_rules.get('articles', []) if grammar_rules.get('articles') else 'No articles in this language'}
    - Cases: {grammar_rules.get('cases', grammar_rules.get('case_suffixes', 'N/A'))}
    - Verb Conjugation: {grammar_rules.get('verb_conjugation', '')}
    - Special Rules: {grammar_rules.get('special_rules', [])}
    - Common Errors to Watch For:
{chr(10).join(error_lines)}
    
    NOTE: These are reference patterns. Focus on what's actually wrong in the user's text, not forcing all these checks.
    """
    
    prompt = f"""
    Analyze grammar in this SPOKEN text: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (adapt complexity of explanations accordingly)
    - User Type: {user_type} (make feedback relevant to their context)
    - Target Language: {target_language}
    
    {lang_instruction}
    {lang_rules_text}
    
    Based on the user's level and type, provide appropriate feedback.
    
    CRITICAL: This is transcribed speech. IGNORE:
    - Punctuation/capitalization/spelling errors
    
    CHECK for:
    1. Filler words (um, uh, like, you know)
    2. Wrong prepositions
    3. Wrong verb tense
    4. Subject-verb agreement
    5. Missing/wrong articles
    6. Word order issues
    7. Missing words
    8. Weak vocabulary
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "is_correct": true/false,
      "filler_words": ["um", "like"],
      "filler_count": 0,
      
      "you_said": "{user_text}",
      "you_should_say": "the grammatically correct version",
      
      "errors": [
        {{
          "type": "verb_tense/preposition/article/subject_verb/word_order/missing_word",
          "you_said": "I goed to store",
          "should_be": "I went to the store",
          "wrong_word": "goed",
          "correct_word": "went",
          "explanation": "Go is irregular - past tense is went, not goed",
          "example_sentence": "Yesterday, I went to the park with my friends."
        }}
      ],
      
      "word_suggestions": [
        {{"you_used": "good", "use_instead": "excellent", "why": "more impactful for {user_type}", "example": "The food was excellent."}}
      ],
      
      "corrected_sentence": "sentence with grammar fixed",
      "improved_sentence": "more natural version with better words",
      "feedback": "2-3 specific sentences about their grammar, tailored to {level} level and {user_type} context"
    }}
    
    RULES:
    - Tailor feedback complexity to {level} level
    - Make suggestions relevant for {user_type}
    - For EACH error, show: you_said, should_be, wrong_word, correct_word
    - Include example_sentence showing correct usage
    - Empty arrays [] if no issues
    - ALL text feedback must be in {target_language} language
    """
    try:
        raw = await call_llm(prompt, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                if not data.get("improved_sentence"):
                    data["improved_sentence"] = data.get("corrected_sentence", user_text)
                
                
                if "errors" in data and isinstance(data["errors"], list):
                    cleaned_errors = []
                    for error in data["errors"]:
                        if isinstance(error, dict):
                            
                            error_type = error.get("type", "").lower()
                            if error_type in ["punctuation", "capitalization", "spelling", "typo"]:
                                continue
                            
                            
                            if not error.get("better_word"):
                                error.pop("better_word", None)
                                error.pop("explanation", None)
                            cleaned_errors.append(error)
                    data["errors"] = cleaned_errors
                
                return data
    except Exception as e:
        logger.debug(f"Grammar analysis fallback: {e}")
    word_count = len(user_text.split())
    return {
        "score": 70, "is_correct": True, "filler_words": [], "filler_count": 0,
        "you_said": user_text, "you_should_say": user_text, "errors": [],
        "word_suggestions": [], "corrected_sentence": user_text, "improved_sentence": user_text,
        "feedback": f"Analyzed {word_count} words. No major grammatical issues detected."
    }


async def analyze_vocab_llm(user_text: str, user_type: str = "student", level: str = "Intermediate", model: str = "gpt", target_language: str = "en") -> dict:
    """llm-based vocabulary analysis with cefr levels and percentages"""
    
    level_context = ""
    if level == "Beginner":
        level_context = "User is at Beginner level. Suggest simple vocabulary improvements."
    elif level == "Intermediate":
        level_context = "User is at Intermediate level. Suggest intermediate-level vocabulary enhancements."
    elif level == "Advanced":
        level_context = "User is at Advanced level. Suggest sophisticated vocabulary alternatives."
    else:
        level_context = "User is at Proficient level. Suggest native-like vocabulary refinements."
    
    user_type_context = ""
    if user_type == "professional":
        user_type_context = "This is a professional user. Suggest business-appropriate vocabulary."
    elif user_type == "student":
        user_type_context = "This is a student. Suggest academic-appropriate vocabulary."
    
    # Language instruction for response
    lang_lower = target_language.lower() if target_language else "en"
    is_english = lang_lower in ["en", "english"]
    lang_instruction = f"IMPORTANT: Provide ALL feedback, explanations, and suggestions in {target_language} language." if not is_english else ""
    
    # Get language-specific vocabulary rules
    lang_rules = get_language_rules(target_language)
    lang_rules_text = ""
    if lang_rules:
        vocab_rules = lang_rules.get("vocabulary", {})
        # Build detailed vocab features
        features = []
        for k, v in vocab_rules.items():
            if k != 'formality_levels':
                features.append(f"    - {k.replace('_', ' ').title()}: {v}")
        
        lang_rules_text = f"""
    LANGUAGE-SPECIFIC VOCABULARY REFERENCE FOR {lang_rules.get('name', target_language).upper()}:
    (Use as GUIDANCE - these are typical patterns, not strict requirements)
    
    - Formality Levels: {vocab_rules.get('formality_levels', [])}
{chr(10).join(features)}
    
    NOTE: Consider these cultural/linguistic features when evaluating vocabulary, but base your analysis on the actual words used.
    """
    
    prompt = f"""
    Analyze vocabulary CEFR levels for: "{user_text}"
    
    USER CONTEXT:
    - Level: {level} (tailor suggestions to this level)
    - User Type: {user_type} (make suggestions relevant to their context)
    - Target Language: {target_language}
    
    {lang_instruction}
    {lang_rules_text}
    
    CRITICAL - SPELLING ERRORS:
    If a word is MISSPELLED (e.g., "awareded", "recieved", "definately"):
    - Do NOT assign it a high CEFR level like C2
    - Include it in "suggestions" with the CORRECT SPELLING as "better_word"
    - Example: {{"word": "awareded", "current_level": "spelling_error", "better_word": "awarded", "suggested_level": "B1"}}
    
    Calculate percentage of words at each CEFR level.
    Percentages should sum to 100.
    
    Return STRICTLY valid JSON:
    {{
      "score": 0-100,
      "overall_level": "A1/A2/B1/B2/C1/C2",
      "total_words": 10,
      "cefr_distribution": {{
        "A1": {{"percentage": 20, "words": ["I", "is"]}},
        "A2": {{"percentage": 30, "words": ["name", "good"]}},
        "B1": {{"percentage": 40, "words": ["actually", "however"]}},
        "B2": {{"percentage": 10, "words": ["sophisticated"]}},
        "C1": {{"percentage": 0, "words": []}},
        "C2": {{"percentage": 0, "words": []}}
      }},
      "feedback": "vocabulary feedback tailored to {level} level and {user_type} context",
      "suggestions": [
        {{"word": "good", "current_level": "A2", "better_word": "excellent", "suggested_level": "B1", "context": "appropriate for {user_type}"}},
        {{"word": "awareded", "current_level": "spelling_error", "better_word": "awarded", "suggested_level": "B1", "context": "correct spelling"}}
      ]
    }}
    
    IMPORTANT:
    - For MISSPELLED words: current_level = "spelling_error", better_word = correct spelling
    - Tailor suggestions to {level} level (don't suggest C1 words to A1 learners)
    - Make suggestions relevant for {user_type} context
    - ALL text feedback must be in {target_language} language
    """
    try:
        raw = await call_llm(prompt, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                
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
                        elif not isinstance(data["cefr_distribution"][level_key], dict):
                            data["cefr_distribution"][level_key] = default_cefr[level_key]
                        else:
                            
                            if "percentage" not in data["cefr_distribution"][level_key]:
                                data["cefr_distribution"][level_key]["percentage"] = 0
                            if "words" not in data["cefr_distribution"][level_key]:
                                data["cefr_distribution"][level_key]["words"] = []
                return data
    except:
        pass
    return {
        "score": 70, "overall_level": "B1", "total_words": 0,
        "cefr_distribution": {
            "A1": {"percentage": 0, "words": []}, "A2": {"percentage": 0, "words": []},
            "B1": {"percentage": 0, "words": []}, "B2": {"percentage": 0, "words": []},
            "C1": {"percentage": 0, "words": []}, "C2": {"percentage": 0, "words": []}
        },
        "feedback": "", "suggestions": []
    }


async def analyze_pronunciation_llm(audio_path: str = None, spoken_text: str = None, level: str = "B1", user_type: str = "student", model: str = "gpt", target_language: str = "en") -> dict:
    """pronunciation analysis using Whisper word-level confidence to detect mispronounced words"""
    if not audio_path:
        return {
            "accuracy": 70, "transcription": spoken_text or "", 
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": spoken_text.split() if spoken_text else [],
            "feedback": "No audio provided for pronunciation analysis",
            "tips": ["Record audio for pronunciation feedback"], 
            "mispronounced_count": 0, "level": level, "user_type": user_type
        }
    
    try:
        target_lang = target_language.lower()
        languages_data = load_language_mapping()
        normalized_target = languages_data.get(target_lang, target_lang) if target_lang in languages_data else target_lang

        # Auto-detect language from audio (don't force language)
        segments, info = await asyncio.to_thread(
            _whisper_model.transcribe, audio_path, word_timestamps=True, task="transcribe"
        )
        
        # Get detected language
        detected_lang = info.language if info else "en"
        
        words_data = []
        transcription = ""
        for seg in segments:
            transcription += seg.text + " "
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip().lower(),
                        "confidence": w.probability,
                        "start": w.start,
                        "end": w.end
                    })
        
        transcription = transcription.strip()
        
        # Keep original transcription for word scores alignment
        original_transcription = transcription
        translated_transcription = None
        
        # Only translate if detected language doesn't match target (keep original for word scores)
        if transcription and detected_lang != normalized_target:
            try:
                translated = await asyncio.to_thread(
                    GoogleTranslator(source=detected_lang, target=normalized_target).translate,
                    transcription
                )
                if translated:
                    logger.debug(f"Pronunciation: Translated from {detected_lang} to {normalized_target}")
                    translated_transcription = translated
            except Exception as e:
                logger.debug(f"Pronunciation translation failed: {e}")
        
        if not words_data:
            return {
                "accuracy": 0, "transcription": transcription, 
                "word_pronunciation_scores": [],
                "words_to_practice": [], "well_pronounced_words": [],
                "feedback": "No speech detected in audio",
                "tips": ["Speak clearly into the microphone"], 
                "mispronounced_count": 0, "level": level, "user_type": user_type
            }
        
        
        
        CONFIDENCE_THRESHOLD = 0.70
        
        mispronounced_words = []
        well_pronounced = []
        all_words_pronunciation = []  
        
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
            
            all_words_pronunciation.append({
                "word": word,
                "pronunciation_percentage": pronunciation_percentage,
                "status": status
            })
                
            if wd["confidence"] < CONFIDENCE_THRESHOLD:
                mispronounced_words.append({
                    "word": word,
                    "confidence": round(wd["confidence"] * 100, 1),
                    "issue": "unclear pronunciation" if wd["confidence"] < 0.5 else "slight pronunciation issue"
                })
            else:
                well_pronounced.append(word)
        
        
        if words_data:
            avg_confidence = sum(w["confidence"] for w in words_data) / len(words_data)
            accuracy = int(avg_confidence * 100)
        else:
            accuracy = 70
        
        # Get language-specific pronunciation rules
        lang_rules = get_language_rules(target_language)
        lang_pron_text = ""
        if lang_rules:
            pron_rules = lang_rules.get("pronunciation", {})
            lang_pron_text = f"""
LANGUAGE-SPECIFIC PRONUNCIATION RULES FOR {lang_rules.get('name', target_language).upper()}:
- Stress Pattern: {pron_rules.get('stress_pattern', '')}
- Difficult Sounds: {pron_rules.get('difficult_sounds', [])}
- Special Features: {', '.join([f'{k}: {v}' for k, v in pron_rules.items() if k not in ['stress_pattern', 'difficult_sounds']])}

Use these rules when analyzing pronunciation for this language."""
        
        llm_prompt = f"""You are a pronunciation coach.

USER CONTEXT:
- Level: {level}
- User Type: {user_type}
- Target Language: {target_language}
{lang_pron_text}

TRANSCRIPTION: "{transcription}"

PER-WORD PRONUNCIATION SCORES (confidence-based):
{all_words_pronunciation}

MISPRONOUNCED WORDS (low confidence from speech recognition):
{mispronounced_words if mispronounced_words else "None - all words were clear!"}

WELL PRONOUNCED WORDS: {well_pronounced[:10]}

OVERALL ACCURACY: {accuracy}%

For each word, analyze their pronunciation percentage and provide specific guidance.

Return STRICTLY valid JSON:
{{
    "word_analysis": [
        {{
            "word": "the word",
            "pronunciation_match": 85.5,
            "rating": "excellent/good/needs_improvement/poor",
            "phonetic_guide": "how to pronounce: ex-AM-ple",
            "improvement_tip": "specific tip if needed, or null if pronunciation is good"
        }}
    ],
    "words_to_practice": [
        {{
            "word": "the word",
            "how_to_say": "syllable breakdown with stress: ex-AM-ple",
            "tip": "specific tip to pronounce this word better"
        }}
    ],
    "well_pronounced_words": ["word1", "word2"],
    "feedback": "2-3 encouraging sentences about their pronunciation",
    "tips": ["general pronunciation tip 1", "general tip 2"]
}}"""

        try:
            llm_response = await call_llm(llm_prompt, mode="strict_json", timeout=30, model=model, target_language=target_language)
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                llm_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON")
        except Exception as llm_error:
            logger.error(f"LLM pronunciation error: {llm_error}")
            
            llm_data = {
                "words_to_practice": [
                    {
                        "word": w["word"],
                        "how_to_say": f"Say '{w['word']}' more clearly",
                        "tip": f"Confidence was {w['confidence']}%. Speak slower and clearer."
                    } for w in mispronounced_words[:5]
                ],
                "well_pronounced_words": well_pronounced[:5],
                "feedback": f"Pronunciation accuracy: {accuracy}%. " + (
                    f"Focus on: {', '.join([w['word'] for w in mispronounced_words[:3]])}" 
                    if mispronounced_words else "Great clarity!"
                ),
                "tips": ["Speak slowly and clearly", "Stress syllables properly"]
            }
        
        
        llm_word_analysis = llm_data.get("word_analysis", [])
        
        
        word_pronunciation_scores = []
        llm_analysis_map = {w.get("word", "").lower(): w for w in llm_word_analysis}
        
        for wp in all_words_pronunciation:
            word_key = wp["word"].lower()
            llm_info = llm_analysis_map.get(word_key, {})
            word_pronunciation_scores.append({
                "word": wp["word"],
                "pronunciation_match_percentage": wp["pronunciation_percentage"],
                "status": wp["status"],
                "phonetic_guide": llm_info.get("phonetic_guide", ""),
                "improvement_tip": llm_info.get("improvement_tip", "")
            })
        
        return {
            "accuracy": accuracy, 
            "transcription": original_transcription,  # Original Whisper transcription (matches word scores)
            "translated_transcription": translated_transcription,  # Translated version if detected != target, else None
            "detected_language": detected_lang,
            "target_language": normalized_target,
            "word_pronunciation_scores": word_pronunciation_scores,
            "words_to_practice": llm_data.get("words_to_practice", []),
            "well_pronounced_words": llm_data.get("well_pronounced_words", well_pronounced),
            "feedback": llm_data.get("feedback", "Analysis complete."),
            "tips": llm_data.get("tips", []),
            "mispronounced_count": len(mispronounced_words),
            "confidence_data": [{"word": w["word"], "confidence": w["confidence"]} for w in mispronounced_words],
            "level": level, 
            "user_type": user_type
        }
        
    except Exception as e:
        logger.error(f"Pronunciation error: {e}")
        return {
            "accuracy": 70, "transcription": spoken_text or "", 
            "word_pronunciation_scores": [],
            "words_to_practice": [], "well_pronounced_words": [],
            "feedback": f"Could not analyze pronunciation: {str(e)}",
            "tips": ["Ensure clear audio recording"],
            "mispronounced_count": 0,
            "level": level, "user_type": user_type
        }


async def generate_question_llm(level: str, scenario: str, user_name: str, target_lang: str, model: str = "gpt") -> tuple:
    """generate question and hint"""
    prompt = f"""You are a friendly {BOT_ROLE} helping {user_name} practice {scenario.replace('_', ' ')} conversations.

IMPORTANT RULES:
- YOU (the bot) ALWAYS ask the questions
- The USER ({user_name}) ALWAYS answers/responds
- Play the appropriate role for the scenario (staff, interviewer, shopkeeper, receptionist, etc.)
- The user plays the customer/guest/interviewee role

SCENARIO: {scenario}
USER LEVEL: {level}

LEVEL GUIDELINES (B1 minimum - no A1/A2):
- B1: Can handle familiar situations, routine conversations, basic opinions
- B2: Independent user, can discuss abstract topics, argue a viewpoint
- C1: Fluent, can express complex ideas spontaneously, varied vocabulary
- C2: Native-like mastery, nuanced, sophisticated, precise

Generate a {level}-appropriate conversation starter for {scenario}.

Return JSON:
{{"question": "[your question to the user]", "hint": "[example response the user could give at {level} level]"}}"""
    try:
        raw = await call_llm(prompt, model=model, target_language=target_lang)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", "How can I help you today?"), data.get("hint", "")
    except:
        pass
    return "How can I help you today?", "For example: I would like..."


async def generate_follow_up_llm(user_response: str, target_lang: str, chat_history: list, model: str = "gpt") -> tuple:
    """generate follow-up question and hint - more interactive and friendly"""
    
    recent_chat = chat_history[-6:] if chat_history else []
    chat_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_chat])
    
    prompt = f"""You are a warm, friendly language tutor having a REAL conversation.

CONVERSATION SO FAR:
{chat_context}

User just said: "{user_response}"

IMPORTANT RULES:
1. This is a REAL CHAT - remember everything from the conversation above
2. Your response should FLOW NATURALLY from what was discussed  
3. FIRST react to what they said (be specific - mention THEIR words!)
4. THEN ask a follow-up that RELATES to the conversation
5. Make it feel like chatting with a friend, not an interview
6. NEVER ask random unrelated questions
7. NEVER repeat questions already asked

GOOD EXAMPLES:
- If they ordered pasta: "Ooh pasta lover! 🍝 What sauce do you usually go for?"
- If they said they like movies: "Nice! I love movies too. What genre is your favorite?"
- If they're at a hotel: "Great choice of hotel! Is this your first time visiting?"

BAD EXAMPLES (DON'T DO):
- Generic: "That's nice. What else?" (boring, not specific)
- Random: "What's your favorite sport?" (unrelated to context)

Keep language at {target_lang} conversational level.

Return JSON:
{{
    "question": "[your specific reaction] + [contextual follow-up question]",
    "hint": "Example: [sample answer they could give]"
}}"""
    try:
        raw = await call_llm(prompt, model=model, target_language=target_lang)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", "That's interesting! Tell me more."), data.get("hint", "Share more details.")
    except:
        pass
    return "That's interesting! Tell me more about that.", "You can share more details or ask me something."


async def generate_personalized_feedback(user_type: str, overall_score: float, scores: dict, user_text: str = "",
                                          grammar: dict = None, vocabulary: dict = None,
                                          pronunciation: dict = None, model: str = "gpt", target_language: str = "en") -> dict:
    """Generate personalized feedback using LLM based on actual errors detected"""
    
    
    grammar_errors = grammar.get("errors", []) if grammar else []
    filler_words = grammar.get("filler_words", []) if grammar else []
    word_suggestions = grammar.get("word_suggestions", []) if grammar else []
    vocab_suggestions = vocabulary.get("suggestions", []) if vocabulary else []
    mispronounced = pronunciation.get("words_to_practice", []) if pronunciation else []
    
    
    errors_context = []
    if grammar_errors:
        errors_context.append(f"Grammar errors: {[e.get('you_said', '') + ' → ' + e.get('should_be', '') for e in grammar_errors[:3]]}")
    if filler_words:
        errors_context.append(f"Filler words used: {filler_words[:5]}")
    if word_suggestions:
        errors_context.append(f"Weak words: {[w.get('weak_word', w.get('you_used', '')) for w in word_suggestions[:3]]}")
    if vocab_suggestions:
        errors_context.append(f"Vocabulary improvements: {[v.get('word', '') + ' → ' + v.get('better_word', '') for v in vocab_suggestions[:3]]}")
    if mispronounced:
        errors_context.append(f"Pronunciation to practice: {[w.get('word', '') if isinstance(w, dict) else w for w in mispronounced[:3]]}")
    
    improvement_areas = []
    strengths = []
    perfect_areas = []
    
    
    for area, score in scores.items():
        if score is None:  
            continue
        if score >= 90:
            perfect_areas.append(area)
            strengths.append(area)
        elif score >= 75:
            strengths.append(area)
        elif score < 60:
            improvement_areas.append(area)
    
    emotion = detect_emotion(user_text)
    
    emotion_guidance = ""
    if emotion == "nervous":
        emotion_guidance = "The user seems NERVOUS. Be extra encouraging, gentle, and reassuring."
    elif emotion == "excited":
        emotion_guidance = "The user seems EXCITED. Match their energy and celebrate their effort."
    else:
        emotion_guidance = "The user seems calm/neutral. Provide balanced, constructive feedback."
    
    # Language instruction for response
    lang_lower = target_language.lower() if target_language else "en"
    is_english = lang_lower in ["en", "english"]
    lang_instruction = f"IMPORTANT: Write ALL feedback content (message, perfect_feedback, quick_tip, emotion_response) in {target_language} language. Keep JSON field names in English, only translate the VALUES." if not is_english else ""
    
    feedback_prompt = f"""You are a warm, friendly language buddy (like a supportive friend) providing personalized feedback.

USER PROFILE:
- Type: {user_type}
- Overall Score: {overall_score}/100
- Scores: Grammar={scores.get('grammar', 70)}, Vocabulary={scores.get('vocabulary', 70)}, Pronunciation={scores.get('pronunciation', 70)}, Fluency={scores.get('fluency', 70)}
- Strengths: {strengths if strengths else 'Building foundation'}
- Areas to Improve: {improvement_areas if improvement_areas else 'Minor refinements'}
- Perfect Scores: {perfect_areas if perfect_areas else 'None yet'}
- What they said: "{user_text[:150]}..."
- Detected Emotion: {emotion}

ACTUAL ERRORS DETECTED (use these for specific feedback):
{chr(10).join(errors_context) if errors_context else "No major errors detected!"}

{emotion_guidance}

{lang_instruction}

Generate a FRIENDLY, CASUAL response that feels like chatting with a supportive friend. Return STRICTLY valid JSON:
{{
    "message": "Start with a SHORT, WARM one-liner reaction (like 'Hey, nice job on that!' or 'Ooh, that was pretty good!' or 'Don't worry, you're getting there!' based on score). THEN 1-2 sentences of specific, encouraging feedback about their ACTUAL errors. Be conversational, use casual language!",
    "perfect_feedback": {{
        "pronunciation": "casual praise if pronunciation was perfect, else null",
        "grammar": "casual praise if grammar was perfect, else null",
        "vocabulary": "casual praise if vocabulary was perfect, else null",
        "fluency": "casual praise if fluency was perfect, else null"
    }},
    "quick_tip": "One friendly, actionable tip - say it like a friend would!",
    "emotion_response": "One warm sentence acknowledging how they might be feeling"
}}

TONE EXAMPLES for "message" based on score:
- Score >= 85: "Awesome! 🎉 That was really well said! Your grammar was spot on, and..."
- Score 70-84: "Hey, nice effort! You're definitely improving. I noticed..."
- Score 50-69: "You're getting there! Don't stress, everyone struggles with..."
- Score < 50: "No worries at all! This is tricky stuff. Let's work on..."

RULES:
- ALWAYS start message with a friendly one-liner reaction
- Be like a supportive friend, NOT a formal teacher
- Use casual language (contractions like "you're", "don't", etc.)
- Reference ACTUAL errors, not generic advice
- Adapt tone based on detected emotion: {emotion}"""
    try:
        llm_response = await call_llm(feedback_prompt, mode="strict_json", timeout=20, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            llm_data = json.loads(json_match.group())
            message = llm_data.get("message", "")
            perfect_feedback = llm_data.get("perfect_feedback", {})
            quick_tip = llm_data.get("quick_tip", "")
            emotion_response = llm_data.get("emotion_response", "")
        else:
            raise ValueError("No valid JSON")
    except:
        if overall_score >= 90:
            message = f"Impressive! You scored {overall_score}% with strong {', '.join(strengths[:2])}."
        elif overall_score >= 70:
            message = f"You're at {overall_score}%. Your {strengths[0] if strengths else 'overall communication'} is solid." + \
                     (f" Work on {improvement_areas[0]} to level up." if improvement_areas else "")
        else:
            message = f"You're building at {overall_score}%. Focus on {improvement_areas[0] if improvement_areas else 'fluency'} first."
        perfect_feedback = {area: f"Excellent {area}!" for area in perfect_areas}
        quick_tip = f"Practice your {improvement_areas[0] if improvement_areas else 'speaking'} daily."
        emotion_response = ""
    
    return {
        "user_type": user_type, "message": message, "improvement_areas": improvement_areas,
        "strengths": strengths, "perfect_areas": perfect_areas,
        "perfect_feedback": perfect_feedback, "quick_tip": quick_tip,
        "emotion": emotion, "emotion_response": emotion_response
    }


async def generate_session_summary_llm(user_name: str, scenario: str, final_scores: dict, 
                                       chat_history: list, total_turns: int, average_wpm: int, model: str = "gpt", target_language: str = "en") -> dict:
    """Generate elaborative LLM-based session summary"""
    
    
    strengths = [area for area, score in final_scores.items() if score is not None and score >= 80]
    weaknesses = [area for area, score in final_scores.items() if score is not None and score < 70]
    overall = int(sum(v for v in final_scores.values() if v is not None) / max(1, len([v for v in final_scores.values() if v is not None]))) if final_scores else 0
    
    prompt = f"""You are a supportive language coach providing a comprehensive session summary.

SESSION DATA:
- Student Name: {user_name}
- Scenario Practiced: {scenario}
- Total Turns: {total_turns}
- Final Scores: Grammar={final_scores.get('grammar', 0)}, Vocabulary={final_scores.get('vocabulary', 0)}, Pronunciation={final_scores.get('pronunciation', 0)}, Fluency={final_scores.get('fluency', 0)}
- Overall Score: {overall}/100
- Average WPM: {average_wpm}
- Strengths: {strengths if strengths else 'Building foundation'}
- Areas for Improvement: {weaknesses if weaknesses else 'Minor refinements'}

Recent conversation:
{json.dumps(chat_history[-6:], indent=2)}

Generate an encouraging, comprehensive session summary. Return STRICTLY valid JSON:
{{
    "summary": "3-4 sentence personalized summary of their session performance",
    "key_achievements": ["achievement 1", "achievement 2"],
    "focus_areas": ["specific area to focus on", "another focus area"],
    "next_steps": ["actionable next step 1", "actionable next step 2"],
    "encouragement": "1-2 sentences of motivation for next session"
}}

Be specific to their scenario ({scenario}) and reference their actual progress."""

    try:
        llm_content = await call_llm(prompt, mode="strict_json", timeout=25, model=model, target_language=target_language)
        json_match = re.search(r'\{[\s\S]*\}', llm_content)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "summary": f"{user_name}, you completed {total_turns} turns practicing {scenario} with an overall score of {overall}%.",
        "key_achievements": [f"Strong {strengths[0]}" if strengths else "Completed conversation practice"],
        "focus_areas": weaknesses[:2] if weaknesses else ["Continue building vocabulary"],
        "next_steps": ["Practice daily for 10-15 minutes", f"Focus on {weaknesses[0] if weaknesses else 'pronunciation'}"],
        "encouragement": "Great effort! Consistency is key to language mastery."
    }


@router.post("/practice")
async def practice_fluent_lang(
    name: str = Form(...),
    native_language: str = Form(...),
    target_language: str = Form(default="en"),
    level: Optional[str] = Form(default=None),  
    scenario: Optional[str] = Form(default=None),  
    user_type: str = Form(default="student"),
    audio_file: Optional[UploadFile] = File(default=None),
    text_input: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    skip_retry: bool = Form(default=False),
    model: Optional[str] = Form(default="gpt"),  
):
    """
    fluent language practice api - CONVERSATIONAL ONBOARDING
    
    flow:
    1. first call (no audio/text, no scenario/level): Bot greets and asks for scenario
    2. user provides scenario: Bot asks for level
    3. user provides level: practice begins with first question
    4. subsequent calls: normal practice with analysis
    5. termination phrase: ends session with summary
    """
    try:
        
        audio_path = None
        
        if not session_id or session_id.strip() == "" or session_id == "string":
            session_id = str(uuid.uuid4())
        
        
        session = await db.get_user_session(session_id)
        session_exists = session is not None
        native_language = session.get("native_language", native_language) if session else native_language
        target_language = session.get("target_language", target_language) if session else target_language
        
        # Normalize language names to ISO codes (englishen, telugute)
        native_language = normalize_language_code(native_language, default="en")
        target_language = normalize_language_code(target_language, default="en")
        
        # Update session with normalized codes if they differ
        if session_exists:
            if session.get("native_language") != native_language or session.get("target_language") != target_language:
                session["native_language"] = native_language
                session["target_language"] = target_language
                await db.update_session(session_id, session)
        
        
        if session_exists and session.get("status") == "completed":
            error_msg = await translate_text("This session has ended. Please start a new conversation.", "en", native_language)
            return {"status": "error", "session_id": session_id, "error": error_msg}
        
        
        if not session_exists:
            
            
            if scenario:
                initial_state = "collecting_level"  
            else:
                initial_state = "welcome"  
            
            session = {
                "state": initial_state,
                "name": name, 
                "level": None,  
                "scenario": scenario,  
                "native_language": native_language, 
                "target_language": target_language,
                "user_type": user_type, 
                "chat_history": [],
                "scores": {"grammar": 0, "vocabulary": 0, "pronunciation": 0, "fluency": 0, "total_wpm": 0, "count": 0, "audio_count": 0},
                "current_question": None, "current_hint": None, "turn_number": 0,
                "last_overall_score": None, "retry_count": 0,
                "attempts": [], "turn_history": [],
                "onboarding_retry": 0
            }
            await db.create_session(
                session_id=session_id,
                session_type="fluent",
                data=session,
                user_id=None,
                user_name=name
            )
            
            
            if scenario:
                ask_level = f"Great! We'll practice {scenario.replace('_', ' ')}. What's your English level? Beginner, Intermediate, Advanced, or Proficient?"
                ask_level_target, ask_level_native = await asyncio.gather(
                    translate_text(ask_level, "en", target_language),
                    translate_text(ask_level, "en", native_language)
                )
                
                session["chat_history"].append({"role": "assistant", "content": ask_level})
                await db.update_session(session_id, session)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_level",
                    "session_id": session_id,
                    "scenario": scenario,
                    "message": {"target": ask_level_target, "native": ask_level_native}
                }

        
        
        current_state = session.get("state", "practicing")
        
        
        
        
        if current_state == "welcome" and not audio_file and not text_input:
            greeting = f"Hi {name}! I'm {BOT_NAME} 🙂 What would you like to practice today? For example: ordering food, hotel check-in, casual conversation, or anything else!"
            
            greeting_target, greeting_native = await asyncio.gather(
                translate_text(greeting, "en", target_language),
                translate_text(greeting, "en", native_language)
            )
            
            session["state"] = "collecting_scenario"
            session["chat_history"].append({"role": "assistant", "content": greeting})
            await db.update_session(session_id, session)
            
            return {
                "status": "onboarding",
                "step": "collecting_scenario",
                "session_id": session_id,
                "message": {"target": greeting_target, "native": greeting_native}
            }
        
        
        
        
        if current_state == "collecting_scenario":
            user_text = text_input or ""
            if audio_file:
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                error_msg = await translate_text("No speech detected. Please tell me what you'd like to practice.", "en", native_language)
                return {"status": "error", "session_id": session_id, "error": error_msg}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_scenario_from_text(user_text, model=model)
            
            if extraction.get("success") and extraction.get("scenario"):
                scenario = extraction["scenario"]
                session["scenario"] = scenario
                session["state"] = "collecting_level"
                session["onboarding_retry"] = 0
                
                
                ask_level = f"Great choice! We'll practice {scenario.replace('_', ' ')}. What's your level? Beginner, Intermediate, Advanced, or Proficient?"
                ask_level_native = await translate_text(ask_level, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": ask_level})
                await db.update_session(session_id, session)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_level", 
                    "session_id": session_id,
                    "scenario": scenario,
                    "message": {"target": ask_level, "native": ask_level_native}
                }
            else:
                
                session["onboarding_retry"] = session.get("onboarding_retry", 0) + 1
                retry_msg = "Could you be more specific? For example: ordering food, shopping, hotel check-in, asking directions, or casual chat?"
                retry_native = await translate_text(retry_msg, "en", native_language)
                
                session["chat_history"].append({"role": "assistant", "content": retry_msg})
                await db.update_session(session_id, session)
                
                return {
                    "status": "onboarding",
                    "step": "collecting_scenario",
                    "session_id": session_id,
                    "retry": True,
                    "message": {"target": retry_msg, "native": retry_native}
                }
        
        
        
        
        if current_state == "collecting_level":
            user_text = text_input or ""
            if audio_file:
                user_text = await transcribe_audio_file(audio_file, target_language)
            
            if not user_text.strip():
                error_msg = await translate_text("No speech detected. Please tell me your level.", "en", native_language)
                return {"status": "error", "session_id": session_id, "error": error_msg}
            
            session["chat_history"].append({"role": "user", "content": user_text})
            
            
            extraction = await extract_level_from_text(user_text, model=model)
            level = extraction.get("level", "B1")
            session["level"] = level
            session["state"] = "practicing"
            session["onboarding_retry"] = 0
            
            
            scenario = session.get("scenario", "casual_conversation")
            question, hint = await generate_question_llm(level, scenario, name, target_language, model=model)
            
            
            level_display = LEVEL_DISPLAY.get(level, level)
            start_msg = f"Perfect! Let's start practicing {scenario.replace('_', ' ')} at {level_display} level."
            
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
            
            return {
                "status": "practice_started",
                "session_id": session_id,
                "level": level_display,
                "scenario": scenario,
                "message": {"target": start_target, "native": start_native},
                "next_question": {"target": question, "native": q_native},
                "hint": {"target": hint, "native": h_native}
            }
        
        
        
        
        if skip_retry and not audio_file and not text_input:
            
            session["waiting_retry_decision"] = False
            session["retry_count"] = 0
            session["retry_clarify_count"] = 0  
            scenario = session.get("scenario", "casual_conversation")
            session_user_type = session.get("user_type", user_type)  
            follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, session_user_type, model=model, target_language=target_language)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            
            
            await db.update_session(session_id, session)
            
            # Translate skip messages to native language
            skipped_msg = await translate_text("Skipped", "en", native_language)
            skipped_next_msg = await translate_text("Skipped. Let's try the next question!", "en", native_language)
            
            return {
                "status": "continue", "session_id": session_id, "transcription": "(skipped)",
                "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                "grammar": {"score": 0, "is_correct": True, "you_said": "", "you_should_say": "", "errors": [], "word_suggestions": [], "corrected_sentence": "", "improved_sentence": "", "feedback": skipped_msg},
                "vocabulary": {"score": 0, "overall_level": "skipped", "cefr_distribution": {}, "feedback": skipped_msg, "suggestions": []},
                "pronunciation": {"accuracy": 0, "total_words": 0, "words_to_practice": [], "well_pronounced_words": [], "feedback": skipped_msg, "practice_sentence": "", "tips": []},
                "fluency": {"score": 0, "wpm": 0, "speed_status": "skipped", "original_text": "", "corrected_text": "", "improved_sentence": ""},
                "personalized_feedback": {"user_type": user_type, "message": skipped_next_msg, "improvement_areas": [], "strengths": [], "perfect_areas": [], "perfect_feedback": {}, "quick_tip": ""},
                "overall_score": 0, "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"]
            }
        
        
        if not audio_file and not text_input and current_state == "practicing" and session.get("turn_number", 0) == 0:
            scenario = session.get("scenario", "casual_conversation")
            level = session.get("level", "B1")
            greeting = f"Hey {name}! I am {BOT_NAME}. I am your {BOT_ROLE}. Let's practice {scenario.replace('_', ' ')}!"
            question, hint = await generate_question_llm(level, scenario, name, target_language, model=model)
            
            
            greeting_target, greeting_native, question_native, hint_native = await asyncio.gather(
                translate_text(greeting, "en", target_language),
                translate_text(greeting, "en", native_language),
                translate_text(question, target_language, native_language),
                translate_text(hint, target_language, native_language)
            )
            
            session["current_question"] = question
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": question})
            
            
            await db.update_session(session_id, session)
            
            return {
                "status": "conversation_started", "session_id": session_id,
                "greeting": {"target": greeting_target, "native": greeting_native},
                "next_question": {"target": question, "native": question_native},
                "hint": {"target": hint, "native": hint_native}
            }
        
        user_text = text_input or ""
        audio_analysis = None
        audio_path = None
        is_audio_input = audio_file is not None  

        
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                shutil.copyfileobj(audio_file.file, tmp)
                temp_upload = tmp.name
            
            try:
                
                def convert_audio():
                    audio = AudioSegment.from_file(temp_upload)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    converted_path = temp_upload.replace('.tmp', '_converted.wav')
                    audio.export(converted_path, format="wav")
                    return converted_path
                
                audio_path = await asyncio.to_thread(convert_audio)
                os.unlink(temp_upload)  
            except:
                
                audio_path = temp_upload
            finally:
                
                
                if audio_path != temp_upload and os.path.exists(temp_upload):
                    try:
                        os.unlink(temp_upload)
                    except:
                        pass
            
            
            session_level_for_audio = session.get("level", "B1")
            audio_analysis = await asyncio.to_thread(analyze_speaking_advanced, audio_path, session_level_for_audio, None, target_language)
            if audio_analysis.get("success") and audio_analysis.get("transcription"):
                user_text = audio_analysis.get("transcription")
            elif not user_text:
                # Fallback to speech_to_text - convert language name to ISO code
                try:
                    languages_data = load_language_mapping()
                    iso_code = languages_data.get(target_language.lower(), target_language.lower())
                    user_text = await asyncio.to_thread(speech_to_text, audio_path, iso_code)
                except:
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
                count = max(1, session["scores"]["count"])
                audio_count = session["scores"].get("audio_count", 0)
                if not audio_count and (
                    session["scores"].get("pronunciation", 0) > 0 or session["scores"].get("fluency", 0) > 0
                ):
                    audio_count = count
                pronunciation_avg = int(session["scores"]["pronunciation"] / audio_count) if audio_count > 0 else 0
                fluency_avg = int(session["scores"]["fluency"] / audio_count) if audio_count > 0 else 0
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": pronunciation_avg if audio_count > 0 else None,
                    "fluency": fluency_avg if audio_count > 0 else None
                }
                
                if audio_count > 0:
                    overall = int(sum(v for v in final_scores.values() if v is not None) / 4)
                else:
                    overall = int((final_scores["grammar"] + final_scores["vocabulary"]) / 2)
                average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
                
                improvement_areas = [area for area, score in final_scores.items() if score is not None and score < 70]
                strengths = [area for area, score in final_scores.items() if score is not None and score >= 80]
                
                
                final_feedback_data = {
                    "final_scores": final_scores,
                    "overall_score": overall,
                    "average_wpm": average_wpm,
                    "strengths": strengths,
                    "improvement_areas": improvement_areas,
                    "total_turns": session.get("turn_number", 0)
                }
                
                
                await db.complete_session(session_id, final_feedback=final_feedback_data, overall_score=overall)
                
                return {
                    "status": "conversation_ended",
                    "session_id": session_id,
                    "final_scores": final_scores,
                    "overall_score": overall,
                    "passing_score": PASSING_SCORE,
                    "average_wpm": average_wpm,
                    "wpm_status": "slow" if average_wpm < 110 else "normal" if average_wpm <= 160 else "fast",
                    "strengths": strengths,
                    "improvement_areas": improvement_areas,
                    "total_turns": session.get("turn_number", 0),
                    "message": {"target": "Session ended. Great practice!", "native": await translate_text("Session ended. Great practice!", "en", native_language)}
                }
            
            retry_keywords = ["yes", "retry", "practice", "again", "try", "redo", "repeat", "once more", "one more"]
            skip_keywords = ["no", "skip", "next", "move", "forward", "pass", "don't want", "not now", "let's move", "move on", "go ahead"]
            
            wants_retry = any(keyword in user_choice for keyword in retry_keywords)
            wants_skip = any(keyword in user_choice for keyword in skip_keywords)
            
            if wants_retry:
                
                session["waiting_retry_decision"] = False
                session["retry_clarify_count"] = 0
                session["is_retry_attempt"] = True  
                current_q = session.get("current_question", "")
                current_h = session.get("current_hint", "")
                session["chat_history"].append({"role": "assistant", "content": current_q})
                await db.update_session(session_id, session)
                
                
                retry_msg = await generate_retry_encouragement(
                    scenario=session.get("scenario", "conversation"),
                    retry_count=session.get("retry_count", 0),
                    previous_score=session.get("last_overall_score", 50),
                    user_name=session.get("name", "there"),
                    model=model,
                    target_language=target_language
                )
                q_native, h_native, retry_msg_native = await asyncio.gather(
                    translate_text(current_q, target_language, native_language),
                    translate_text(current_h, target_language, native_language),
                    translate_text(retry_msg, target_language, native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "next_question": {"target": current_q, "native": q_native},
                    "hint": {"target": current_h, "native": h_native},
                    "message": {"target": retry_msg, "native": retry_msg_native},
                    "turn_number": session.get("turn_number", 0)
                }
            elif wants_skip:
                
                session["waiting_retry_decision"] = False
                session["retry_clarify_count"] = 0
                session["retry_count"] = 0
                session["is_retry_attempt"] = False  
                session["last_overall_score"] = None  
                scenario = session.get("scenario", "casual_conversation")
                follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, user_type, model=model, target_language=target_language)
                session["current_question"] = follow_up
                session["current_hint"] = hint
                session["chat_history"].append({"role": "assistant", "content": follow_up})
                
                await db.update_session(session_id, session)
                
                
                skip_msg = await generate_skip_message(
                    scenario=scenario,
                    user_name=session.get("name", "there"),
                    model=model,
                    target_language=target_language
                )
                skip_msg_native, follow_up_native, hint_native = await asyncio.gather(
                    translate_text(skip_msg, target_language, native_language),
                    translate_text(follow_up, target_language, native_language),
                    translate_text(hint, target_language, native_language)
                )
                
                return {
                    "status": "continue",
                    "session_id": session_id,
                    "message": {"target": skip_msg, "native": skip_msg_native},
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
                    scenario = session.get("scenario", "casual_conversation")
                    follow_up, hint = await generate_context_aware_follow_up("", session["chat_history"], scenario, user_type, model=model, target_language=target_language)
                    session["current_question"] = follow_up
                    session["current_hint"] = hint
                    session["chat_history"].append({"role": "assistant", "content": follow_up})
                    
                    await db.update_session(session_id, session)
                    
                    return {
                        "status": "auto_skipped",
                        "session_id": session_id,
                        "message": {"target": "Moving to the next question.", "native": await translate_text("Moving to the next question.", "en", native_language)},
                        "next_question": {"target": follow_up, "native": await translate_text(follow_up, target_language, native_language)},
                        "hint": {"target": hint, "native": await translate_text(hint, target_language, native_language)},
                        "turn_number": session["turn_number"]
                    }
                else:
                    
                    level = session.get("level", "B1")
                    scenario = session.get("scenario", "casual_conversation")
                    
                    
                    if is_audio_input:
                        grammar, vocabulary, pronunciation = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, user_type=user_type, model=model, target_language=target_language),
                            analyze_vocab_llm(user_text, level=level, user_type=user_type, model=model, target_language=target_language),
                            analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=level, user_type=user_type, model=model, target_language=target_language)
                        )
                        
                        try:
                            audio_for_duration = AudioSegment.from_file(audio_path)
                            audio_duration = len(audio_for_duration) / 1000
                        except:
                            word_count = len(user_text.split())
                            audio_duration = max(1, word_count / 2.5)
                        fluency = await analyze_fluency_metrics(user_text, audio_duration)
                    else:
                        
                        grammar, vocabulary = await asyncio.gather(
                            analyze_grammar_llm(user_text, level=level, user_type=user_type, model=model, target_language=target_language),
                            analyze_vocab_llm(user_text, level=level, user_type=user_type, model=model, target_language=target_language)
                        )
                        pronunciation = None
                        fluency = None
                    
                    
                    if is_audio_input:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": pronunciation.get("score", pronunciation.get("accuracy", 70)) if pronunciation else 0,
                            "fluency": fluency.get("score", 70) if fluency else 0
                        }
                        
                        overall_score = int(scores["pronunciation"] * 0.30 + scores["grammar"] * 0.30 + scores["vocabulary"] * 0.20 + scores["fluency"] * 0.20)
                    else:
                        scores = {
                            "grammar": grammar.get("score", 70),
                            "vocabulary": vocabulary.get("score", 70),
                            "pronunciation": None,
                            "fluency": None
                        }
                        
                        overall_score = int(scores["grammar"] * 0.50 + scores["vocabulary"] * 0.50)
                    
                    personalized_feedback = await generate_personalized_feedback(
                        user_type, overall_score, scores, user_text,
                        grammar=grammar, vocabulary=vocabulary, pronunciation=pronunciation, model=model,
                        target_language=session.get("target_language", "en")
                    )
                    
                    if clarify_count == 1:
                        clarify_msg = "I didn't quite catch that. Say 'yes' to try again, or 'no' to skip to the next question."
                    else:
                        clarify_msg = "Please say 'yes' to retry the same question, or 'no' to move on."
                    
                    await db.update_session(session_id, session)
                    
                    
                    if is_audio_input and pronunciation:
                        (grammar_t, vocab_t, pron_t, feedback_t, clarify_native) = await asyncio.gather(
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                            translate_text(clarify_msg, "en", native_language)
                        )
                    else:
                        (grammar_t, vocab_t, feedback_t, clarify_native) = await asyncio.gather(
                            translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                            translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                            translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS),
                            translate_text(clarify_msg, "en", native_language)
                        )
                        pron_t = None
                    
                    return {
                        "status": "clarify_retry",
                        "session_id": session_id,
                        "transcription": user_text,
                        "message": {"target": clarify_msg, "native": clarify_native},
                        "grammar": grammar_t,
                        "vocabulary": vocab_t,
                        "pronunciation": pron_t,
                        "fluency": fluency,
                        "personalized_feedback": feedback_t,
                        "overall_score": overall_score,
                        "clarify_count": clarify_count,
                        "turn_number": session.get("turn_number", 0)
                    }

        
        cleaned_text = user_text.lower().strip().rstrip('.,!?')
        is_termination = cleaned_text in TERMINATION_PHRASES
        
        if is_termination:
            count = max(1, session["scores"]["count"])
            audio_count = max(1, session["scores"].get("audio_count", 0))  
            
            
            has_audio_turns = session["scores"].get("audio_count", 0) > 0  
            
            if has_audio_turns:
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": int(session["scores"]["pronunciation"] / audio_count),  
                    "fluency": int(session["scores"]["fluency"] / audio_count)  
                }
                
                overall = int(
                    final_scores["pronunciation"] * 0.30 +
                    final_scores["grammar"] * 0.30 +
                    final_scores["vocabulary"] * 0.20 +
                    final_scores["fluency"] * 0.20
                )
                average_wpm = int(session["scores"].get("total_wpm", 0) / audio_count) if audio_count > 0 else 0
            else:
                
                final_scores = {
                    "grammar": int(session["scores"]["grammar"] / count),
                    "vocabulary": int(session["scores"]["vocabulary"] / count),
                    "pronunciation": None,
                    "fluency": None
                }
                
                overall = int(
                    final_scores["grammar"] * 0.50 +
                    final_scores["vocabulary"] * 0.50
                )
                average_wpm = 0

            
            
            if average_wpm < 110:
                wpm_status = "Slow - Try to speak a bit faster"
            elif average_wpm > 160:
                wpm_status = "Fast - Try to slow down for clarity"
            else:
                wpm_status = "Natural - Great speaking pace!"
            
            improvement_areas = []
            strengths = []
            for area, score in final_scores.items():
                if score is None:  
                    continue
                if score >= 85:
                    strengths.append(area)
                elif score < 70:
                    improvement_areas.append(area)
            
            
            turn_history_summary = []
            for i, attempt in enumerate(session.get("attempts", []), 1):
                
                pron_data = attempt.get("pronunciation") or {}
                fluency_data = attempt.get("fluency") or {}
                turn_history_summary.append({
                    "turn": i,
                    "grammar": attempt.get("grammar", {}).get("score", 0),
                    "vocabulary": attempt.get("vocabulary", {}).get("score", 0),
                    "pronunciation": pron_data.get("accuracy", 0) if pron_data else 0,
                    "fluency": fluency_data.get("score", 0) if fluency_data else 0,
                    "wpm": fluency_data.get("wpm", 0) if fluency_data else 0,
                    "overall": attempt.get("overall_score", 0),
                    "transcription": attempt.get("transcription", "")[:50]
                })
            
            final_feedback_prompt = f"""You are {session.get('name', 'friend')}'s warm, encouraging language coach. Generate a HIGHLY PERSONALIZED and INTERACTIVE final session summary.

SESSION DATA:
- Student Name: {session['name']}
- User Type: {session.get('user_type', 'student')}
- Level: {session.get('level', 'B1')}
- Scenario: {session.get('scenario', 'conversation')}
- Total Turns: {session.get('turn_number', 0)}
- Final Scores: Grammar={final_scores['grammar']}%, Vocabulary={final_scores['vocabulary']}%, Pronunciation={final_scores['pronunciation'] if final_scores['pronunciation'] is not None else 'N/A'}%, Fluency={final_scores['fluency'] if final_scores['fluency'] is not None else 'N/A'}%
- Overall Score: {overall}/100
- Average WPM: {average_wpm}
- Strengths: {strengths if strengths else 'Building foundation'}
- Areas to Improve: {improvement_areas if improvement_areas else 'Minor refinements'}

TURN-BY-TURN PERFORMANCE (analyze progression):
{json.dumps(turn_history_summary, indent=2)}

CONVERSATION EXCERPTS:
{session.get('chat_history', [])[-6:]}

REQUIREMENTS:
1. Be WARM and CONVERSATIONAL - address {session['name']} directly
2. Reference SPECIFIC phrases they said during the session
3. Explain WHY scores changed between turns (e.g., "Your grammar jumped from 80% to 95% in Turn 3 when you used complex sentences")
4. Give ACTIONABLE, specific tips (not generic advice)
5. Celebrate their wins enthusiastically!

Return STRICTLY valid JSON:
{{
    "detailed_feedback": {{
        "grammar": {{"score": {final_scores['grammar']}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences explaining grammar performance with specific examples from their responses, mention what they did well and one specific thing to improve", "trend": "improved/declined/stable"}},
        "vocabulary": {{"score": {final_scores['vocabulary']}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about vocabulary richness, mention specific words they used well or could upgrade", "trend": "improved/declined/stable"}},
        "pronunciation": {{"score": {final_scores['pronunciation'] if final_scores['pronunciation'] is not None else '"N/A"'}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about clarity and specific words they pronounced well or struggled with", "trend": "improved/declined/stable"}},
        "fluency": {{"score": {final_scores['fluency'] if final_scores['fluency'] is not None else '"N/A"'}, "status": "Excellent/Good/Needs Work", "feedback": "2-3 sentences about speaking pace ({average_wpm} WPM), hesitations, and natural flow", "trend": "improved/declined/stable"}}
    }},
    "turn_comparison": {{
        "best_turn": <number>,
        "worst_turn": <number>,
        "biggest_improvement": "Specific improvement like 'Grammar jumped from X% to Y% when you started using complex sentences'",
        "area_with_most_growth": "<area name>"
    }},
    "overall_analysis": "3-4 warm, personalized sentences comparing {session['name']}'s start vs end performance. Reference specific things they said. Example: 'Hey {session['name']}! You started a bit nervous in Turn 1, but by Turn 3 you were nailing those complex sentences! Your vocabulary really shined when you described...'",
    "suggestions": ["Very specific actionable suggestion with example exercise", "Another practical tip they can do today"],
    "tip": "One fun, memorable tip personalized to their weakest area - make it encouraging!"
}}"""

            try:
                llm_final_content = await call_llm(final_feedback_prompt, mode="strict_json", timeout=30, model=model)
                json_match = re.search(r'\{[\s\S]*\}', llm_final_content)
                if json_match:
                    final_data = json.loads(json_match.group())
                    detailed_feedback = final_data.get("detailed_feedback", {})
                    turn_comparison = final_data.get("turn_comparison", {})
                    analysis_text = final_data.get("overall_analysis", "")
                    final_message = final_data.get("final_message", "")
                    suggestions = final_data.get("suggestions", [])
                    tip = final_data.get("tip", "")
                else:
                    raise ValueError("No valid JSON")
            except:
                
                first_turn = turn_history_summary[0] if turn_history_summary else {}
                last_turn = turn_history_summary[-1] if turn_history_summary else {}
                
                
                grammar_change = last_turn.get("grammar", 0) - first_turn.get("grammar", 0)
                vocab_change = last_turn.get("vocabulary", 0) - first_turn.get("vocabulary", 0)
                
                detailed_feedback = {
                    "grammar": {
                        "score": final_scores["grammar"], 
                        "status": "Excellent" if final_scores["grammar"] >= 85 else "Good" if final_scores["grammar"] >= 70 else "Needs Work", 
                        "feedback": f"Great job, {session['name']}! Your grammar averaged {final_scores['grammar']}% across {session.get('turn_number', 0)} turns. " + 
                                   (f"You improved by {grammar_change}% from start to finish!" if grammar_change > 0 else "Keep practicing sentence structure for even better results."),
                        "trend": "improved" if grammar_change > 0 else "declined" if grammar_change < 0 else "stable"
                    },
                    "vocabulary": {
                        "score": final_scores["vocabulary"], 
                        "status": "Excellent" if final_scores["vocabulary"] >= 85 else "Good" if final_scores["vocabulary"] >= 70 else "Needs Work", 
                        "feedback": f"Your vocabulary scored {final_scores['vocabulary']}%! " +
                                   ("You used rich, varied words throughout - impressive!" if final_scores["vocabulary"] >= 85 else "Try incorporating more B2/C1 level words to sound more natural."),
                        "trend": "improved" if vocab_change > 0 else "declined" if vocab_change < 0 else "stable"
                    },
                }
                
                if final_scores["pronunciation"] is not None:
                    detailed_feedback["pronunciation"] = {
                        "score": final_scores["pronunciation"], 
                        "status": "Excellent" if final_scores["pronunciation"] >= 85 else "Good" if final_scores["pronunciation"] >= 70 else "Needs Work", 
                        "feedback": f"Pronunciation at {final_scores['pronunciation']}%! " +
                                   ("Your clarity was excellent - native speakers would understand you easily!" if final_scores["pronunciation"] >= 85 else 
                                    "Focus on clearly pronouncing consonant sounds and word endings for better clarity."),
                        "trend": "stable"
                    }
                if final_scores["fluency"] is not None:
                    speed_desc = "perfect pace" if 110 <= average_wpm <= 160 else "a bit slow - try to relax and speak faster" if average_wpm < 110 else "quite fast - try pausing between ideas"
                    detailed_feedback["fluency"] = {
                        "score": final_scores["fluency"], 
                        "status": "Excellent" if final_scores["fluency"] >= 85 else "Good" if final_scores["fluency"] >= 70 else "Needs Work", 
                        "feedback": f"You spoke at {average_wpm} words per minute - {speed_desc}! " +
                                   ("Great natural rhythm!" if final_scores["fluency"] >= 85 else "Practice speaking in longer phrases without hesitation."),
                        "trend": "stable"
                    }
                
                best_turn = max(range(len(turn_history_summary)), key=lambda i: turn_history_summary[i].get("overall", 0)) + 1 if turn_history_summary else 1
                worst_turn = min(range(len(turn_history_summary)), key=lambda i: turn_history_summary[i].get("overall", 0)) + 1 if turn_history_summary else 1
                
                turn_comparison = {
                    "best_turn": best_turn, 
                    "worst_turn": worst_turn, 
                    "biggest_improvement": f"Your best performance was in Turn {best_turn} with {turn_history_summary[best_turn-1].get('overall', 0) if turn_history_summary else 0}%!" if turn_history_summary else "N/A", 
                    "area_with_most_growth": strengths[0] if strengths else "overall"
                }
                
                analysis_text = f"Hey {session['name']}! 🎉 You completed {session.get('turn_number', 0)} practice turns with an overall score of {overall}%. " + \
                               (f"Your strongest areas were {', '.join(strengths)} - amazing work! " if strengths else "You're building a solid foundation. ") + \
                               (f"For next time, let's focus on improving your {', '.join(improvement_areas)} - you've got this!" if improvement_areas else "Keep up the great momentum!")
                
                final_message = f"Awesome session, {session['name']}! You scored {overall}% - be proud of your progress!"
                
                suggestions = [
                    f"Practice {improvement_areas[0]} by recording yourself and comparing with native speakers" if improvement_areas else "Challenge yourself with longer, more complex sentences",
                    f"Try shadowing exercises - listen to English content and repeat immediately" if average_wpm < 110 else "Read aloud for 10 minutes daily to maintain your great pace"
                ]
                
                tip = f"Quick tip for {session['name']}: " + (f"Focus on {improvement_areas[0]} by practicing tongue twisters!" if improvement_areas and improvement_areas[0] == "pronunciation" else 
                      f"Boost your {improvement_areas[0] if improvement_areas else 'speaking'} by having 5-minute English conversations with yourself daily - it really works!")
            
            
            final_feedback_data = {
                "final_scores": final_scores,
                "overall_score": overall,
                "average_wpm": average_wpm,
                "detailed_feedback": detailed_feedback,
                "turn_comparison": turn_comparison,
                "strengths": strengths,
                "improvement_areas": improvement_areas,
                "overall_analysis": analysis_text,
                "suggestions": suggestions,
                "tip": tip,
                "turn_history": turn_history_summary
            }
            await db.complete_session(session_id, final_feedback=final_feedback_data, overall_score=overall)
            
            # Translate end-of-session strings to native language
            final_message_native = await translate_text(final_message, "en", native_language)
            analysis_native = await translate_text(analysis_text, "en", native_language)
            tip_native = await translate_text(tip, "en", native_language)
            suggestions_native = [await translate_text(s, "en", native_language) for s in suggestions]
            
            return {
                "status": "conversation_ended", "session_id": session_id,
                "final_scores": final_scores, "overall_score": overall, "passing_score": PASSING_SCORE,
                "average_wpm": average_wpm, "wpm_status": wpm_status,
                "detailed_feedback": detailed_feedback, "turn_comparison": turn_comparison,
                "strengths": strengths, "improvement_areas": improvement_areas,
                "overall_analysis": analysis_native, "suggestions": suggestions_native,
                "total_turns": session.get("turn_number", 0), "message": final_message_native, "tip": tip_native,
                "turn_history": turn_history_summary
            }

        
        session_level = session.get("level", "B1")
        session_user_type = session.get("user_type", user_type)
        
        
        if is_audio_input:
            grammar, vocabulary, pronunciation = await asyncio.gather(
                analyze_grammar_llm(user_text, level=session_level, user_type=session_user_type, model=model, target_language=session.get("target_language", "en")),
                analyze_vocab_llm(user_text, user_type=session_user_type, level=session_level, model=model, target_language=session.get("target_language", "en")),
                analyze_pronunciation_llm(audio_path=audio_path, spoken_text=user_text, level=session_level, user_type=session_user_type, model=model, target_language=session.get("target_language", "en"))
            )
        else:
            
            grammar, vocabulary = await asyncio.gather(
                analyze_grammar_llm(user_text, level=session_level, user_type=session_user_type, model=model, target_language=session.get("target_language", "en")),
                analyze_vocab_llm(user_text, user_type=session_user_type, level=session_level, model=model, target_language=session.get("target_language", "en"))
            )
            pronunciation = None
        
        
        if is_audio_input:
            if audio_analysis and audio_analysis.get("success"):
                fluency_data = audio_analysis.get("fluency", {})
                fluency = {
                    "score": fluency_data.get("score", 70), "wpm": fluency_data.get("wpm", 100),
                    "speed_status": "slow" if fluency_data.get("wpm", 100) < 100 else "normal" if fluency_data.get("wpm", 100) < 160 else "fast",
                    "pause_count": fluency_data.get("pause_count", 0),
                    "hesitation_count": len(grammar.get("filler_words", []))
                }
            else:
                word_count = len(user_text.split())
                audio_duration_seconds = 5
                if audio_path:
                    try:
                        audio_for_duration = AudioSegment.from_file(audio_path)
                        audio_duration_seconds = len(audio_for_duration) / 1000
                    except:
                        audio_duration_seconds = 5
                
                fluency = await analyze_fluency_metrics(user_text, audio_duration_seconds)
            
            fluency["original_text"] = user_text
            fluency["corrected_text"] = grammar.get("corrected_sentence", user_text)
            fluency["improved_sentence"] = grammar.get("improved_sentence", user_text)
            fluency["filler_words_removed"] = grammar.get("filler_words", [])
        else:
            fluency = None  
        
        
        if is_audio_input:
            scores = {
                "grammar": grammar.get("score", 70), "vocabulary": vocabulary.get("score", 70),
                "pronunciation": pronunciation.get("accuracy", 70) if pronunciation else 0, "fluency": fluency.get("score", 70) if fluency else 0
            }
            
            overall_score = int(scores["pronunciation"] * 0.30 + scores["grammar"] * 0.30 + scores["vocabulary"] * 0.20 + scores["fluency"] * 0.20)
        else:
            scores = {
                "grammar": grammar.get("score", 70), "vocabulary": vocabulary.get("score", 70),
                "pronunciation": None, "fluency": None
            }
            
            overall_score = int(scores["grammar"] * 0.50 + scores["vocabulary"] * 0.50)
        
        
        session["scores"]["grammar"] += scores["grammar"]
        session["scores"]["vocabulary"] += scores["vocabulary"]
        if is_audio_input:
            session["scores"]["pronunciation"] += scores["pronunciation"]
            session["scores"]["fluency"] += scores["fluency"]
            session["scores"]["total_wpm"] += fluency.get("wpm", 100) if fluency else 100
            session["scores"]["audio_count"] = session["scores"].get("audio_count", 0) + 1  
        session["scores"]["count"] += 1
        session["turn_number"] += 1


        personalized_feedback = await generate_personalized_feedback(
            user_type, overall_score, scores, user_text,
            grammar=grammar, vocabulary=vocabulary, pronunciation=pronunciation, model=model,
            target_language=session.get("target_language", "en")
        )
        
        native_language = session.get("native_language", "hi")
        target_language = session.get("target_language", "en")
        
        
        if is_audio_input and pronunciation:
            grammar_translated, vocab_translated, pron_translated, fluency_translated, feedback_translated = await asyncio.gather(
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(pronunciation, target_language, native_language, PRON_FIELDS),
                translate_analysis(fluency, target_language, native_language, FLUENCY_FIELDS) if fluency else asyncio.coroutine(lambda: fluency)(),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
            )
        else:
            grammar_translated, vocab_translated, feedback_translated = await asyncio.gather(
                translate_analysis(grammar, target_language, native_language, GRAMMAR_FIELDS),
                translate_analysis(vocabulary, target_language, native_language, VOCAB_FIELDS),
                translate_analysis(personalized_feedback, target_language, native_language, PERSONAL_FIELDS)
            )
            pron_translated = None
            fluency_translated = fluency

        
        current_attempt = {
            "pronunciation": pron_translated,
            "grammar": grammar_translated,
            "vocabulary": vocab_translated,
            "fluency": fluency_translated,
            "personalized_feedback": feedback_translated,
            "overall_score": overall_score,
            "average_score": overall_score,
            "transcription": user_text,
            "turn_number": session["turn_number"]
        }
        session["attempts"].append(current_attempt)
        
        
        is_retrying = session.get("retry_count", 0) > 0 or session.get("is_retry_attempt", False)
        session["is_retry_attempt"] = False  
        improvement = await compare_attempts(session["attempts"], level=session_level, user_type=session_user_type, target_language=target_language) if is_retrying else {}
        
        prev_score = session.get("last_overall_score")
        if is_retrying and prev_score is not None:
            diff = overall_score - prev_score
            improvement["overall_previous_score"] = prev_score
            improvement["overall_current_score"] = overall_score
            improvement["overall_difference"] = diff
            improvement["overall_improved"] = diff > 0
            if diff > 0:
                msg = f"You improved from {prev_score}% to {overall_score}%! (+{diff}%)"
            elif diff < 0:
                msg = f"Score changed from {prev_score}% to {overall_score}% ({diff}%)"
            else:
                msg = f"Score unchanged at {overall_score}%"
            improvement["overall_message"] = await translate_text(msg, "en", native_language)
        
        # Update stored attempt with improvement data
        if session["attempts"]:
            session["attempts"][-1]["improvement"] = improvement
        
        session["last_overall_score"] = overall_score
        should_retry = overall_score < PASSING_SCORE and not skip_retry
        
        if should_retry:
            session["retry_count"] = session.get("retry_count", 0) + 1
        else:
            session["retry_count"] = 0
        
        if should_retry:
            current_q = session.get("current_question", "")
            current_h = session.get("current_hint", "")
            
            
            retry_prompt_en = "I see there's room for improvement. Would you like to retry? Say 'yes' to try again or 'no' to skip."
            retry_prompt_target = await translate_text(retry_prompt_en, "en", target_language) if target_language != "en" else retry_prompt_en
            
            
            q_native, h_native, retry_prompt_native = await asyncio.gather(
                translate_text(current_q, target_language, native_language),
                translate_text(current_h, target_language, native_language),
                translate_text(retry_prompt_en, "en", native_language)
            )
            
            
            session["waiting_retry_decision"] = True
            await db.update_session(session_id, session)
            
            return {
                "status": "feedback", "session_id": session_id, "transcription": user_text,
                "grammar": grammar_translated, "vocabulary": vocab_translated, 
                "pronunciation": pron_translated, "fluency": fluency_translated,
                "personalized_feedback": feedback_translated, "overall_score": overall_score,
                "passing_score": PASSING_SCORE, "should_retry": True,
                "retry_prompt": {"target": retry_prompt_target, "native": retry_prompt_native},
                "retry_count": session.get("retry_count", 1), "improvement": improvement, "turn_number": session["turn_number"],
                "next_question": {"target": current_q, "native": q_native},
                "hint": {"target": current_h, "native": h_native}
            }

        else:
            
            session["last_overall_score"] = None
            session["is_retry_attempt"] = False
            follow_up, hint = await generate_follow_up_llm(user_text, target_language, session["chat_history"], model=model)
            session["current_question"] = follow_up
            session["current_hint"] = hint
            session["chat_history"].append({"role": "assistant", "content": follow_up})
            
            
            follow_up_native, hint_native = await asyncio.gather(
                translate_text(follow_up, target_language, native_language),
                translate_text(hint, target_language, native_language)
            )
            
            await db.update_session(session_id, session, overall_score=overall_score)
            
            return {
                "status": "continue", "session_id": session_id, "transcription": user_text,
                "grammar": grammar_translated, "vocabulary": vocab_translated,
                "pronunciation": pron_translated, "fluency": fluency_translated,
                "personalized_feedback": feedback_translated, "overall_score": overall_score,
                "passing_score": PASSING_SCORE, "should_retry": False, "turn_number": session["turn_number"],
                "improvement": improvement,  
                "next_question": {"target": follow_up, "native": follow_up_native},
                "hint": {"target": hint, "native": hint_native}
            }
    
    except Exception as e:
        logger.exception(f"Error in practice_fluent_lang: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


@router.get("/sessions")
async def list_fluent_sessions():
    """list active fluent sessions from database"""
    sessions_list = await db.list_sessions(session_type="fluent")
    return {"active_sessions": len(sessions_list), "sessions": sessions_list}


@router.get("/sessions/{session_id}")
async def get_fluent_session(session_id: str):
    """get fluent session data from database"""
    session_data = await db.get_user_session(session_id)
    if session_data:
        return {"status": "success", "session_id": session_id, "data": session_data}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/feedback/{session_id}")
async def get_fluent_feedback(session_id: str):
    """
    Get detailed per-turn feedback for a fluent session.
    
    Returns structured grammar, vocabulary, pronunciation, and fluency feedback for each turn.
    """
    feedback = await db.get_session_feedback(session_id)
    if not feedback:
        raise HTTPException(status_code=404, detail="Session not found")
    if feedback["session_type"] != "fluent":
        raise HTTPException(status_code=400, detail="Not a fluent session")
    return feedback
this def analyze_speaking_advanced(audio_path: str, target_cefr: str = "B1", transcription: str = None, target_lang: str = "en") -> dict:
    """
    Advanced speaking analysis: fluency, pronunciation, 
    grammar errors, filler words, and CEFR vocabulary assessment.
    
    Args:
        audio_path: Path to the audio file for audio-based metrics (fluency, pronunciation)
        target_cefr: Target CEFR level for vocabulary assessment
        transcription: Optional - if provided, skips Whisper transcription and uses this text.
                       Use transcribe_audio_file() from fluent_api_v2 to get transcription.
        target_lang: Target language code (e.g., "en", "hi", "es") - will translate if detected language differs
    """
    result = {
        "success": False,
        "error": None,
        "transcription": "",
        "fluency": {},
        "pronunciation": {},
        "grammar_assessment": {
            "errors": [],
            "filler_words": [],
            "corrected_text": "",
            "filler_count": 0
        },
        "vocabulary_analysis": {
            "cefr_level": "",
            "vocabulary_score": 0,
            "suggestions": []
        },
        "overall_score": 0.0,
        "llm_feedback": ""
    }
    
    try:
        # Normalize target language code
        normalized_target = target_lang.lower() if target_lang else "en"
        lang_mapping = load_language_mapping()
        if normalized_target in lang_mapping:
            normalized_target = lang_mapping[normalized_target]
        
        # Use provided transcription or do Whisper transcription as fallback
        if transcription:
            # Use the provided transcription (from transcribe_audio_file)
            result["transcription"] = transcription
        else:
            # Fallback: do transcription here with auto-detection
            model = _get_whisper_model()
            segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
            segments = list(segments)
            transcription = " ".join([s.text.strip() for s in segments]).strip()
            
            # Detect language and translate if needed
            detected_lang = info.language if info else "en"
            if transcription and detected_lang != normalized_target:
                try:
                    from deep_translator import GoogleTranslator
                    translated = GoogleTranslator(source=detected_lang, target=normalized_target).translate(transcription)
                    if translated:
                        transcription = translated
                except Exception as e:
                    logger.debug(f"Translation to {normalized_target} failed: {e}")
            
            result["transcription"] = transcription
        
        if not transcription:
            result["error"] = "No speech detected"
            return _to_python_type(result)

        
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(transcription.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        
        intervals = librosa.effects.split(y, top_db=30)
        pauses = []
        if len(intervals) > 1:
            for i in range(1, len(intervals)):
                p_dur = (intervals[i][0] - intervals[i-1][1]) / sr
                if p_dur > 0.3: pauses.append(p_dur)
        
        wpm_score = 100
        if wpm < IDEAL_WPM_MIN: wpm_score = max(0, 100 - (IDEAL_WPM_MIN - wpm) * 2)
        elif wpm > IDEAL_WPM_MAX: wpm_score = max(0, 100 - (wpm - IDEAL_WPM_MAX) * 2)
        
        pause_score = max(0, 100 - len(pauses) * 5 - sum(p for p in pauses if p > 2) * 10)
        fluency_score = (wpm_score * 0.5 + pause_score * 0.5)
        
        result["fluency"] = {
            "wpm": round(wpm, 1),
            "pause_count": len(pauses),
            "score": round(fluency_score, 1)
        }
        
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
        pitch_std = np.std(pitch_vals) if pitch_vals else 0
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        clarity_score = min(100, np.mean(spectral_centroids) / 30)
        intonation_score = min(100, max(0, 85 + (pitch_std - 20) * 0.2)) if 20 < pitch_std < 100 else (60 if pitch_std < 20 else 75)
        
        pron_score = (intonation_score * 0.4 + clarity_score * 0.6)
        result["pronunciation"] = {
            "clarity": round(clarity_score, 1),
            "intonation": round(intonation_score, 1),
            "score": round(pron_score, 1)
        }

        
        prompt = f"""Analyze the following spoken transcript for a language learner targeting CEFR level {target_cefr}.
        
        TRANSCRIPT: "{transcription}"
        
        INSTRUCTIONS:
        1. Identify GRAMMAR ERRORS and provide corrections.
        2. Identify FILLER WORDS (like "um", "ah", "like", "you know", "er").
        3. Provide a FULL CORRECTED VERSION of the text.
        4. Analyze VOCABULARY:
           - Is it appropriate for {target_cefr}?
           - Suggest 3 better/more advanced words.
        5. Provide an overall linguistic feedback.

        Respond ONLY in this JSON format:
        {{
            "grammar_assessment": {{
                "errors": [
                    {{"error": "incorrect phrase", "correction": "correct phrase", "rule": "why it was wrong"}}
                ],
                "filler_words": ["um", "like"],
                "corrected_text": "Complete corrected transcription here"
            }},
            "vocabulary_analysis": {{
                "detected_cefr": "B1",
                "vocabulary_score": 85,
                "suggestions": [
                    {{"original": "good", "advanced": "exceptional", "context": "describing weather"}}
                ]
            }},
            "linguistic_score": 80,
            "feedback": "Overall assessment of language usage."
        }}
        """
        
        llm_response = call_gpt(prompt, "You are an expert language examiner. Focus on grammar, filler words, and CEFR vocabulary.")
        
        import json
        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start != -1:
                parsed = json.loads(llm_response[json_start:json_end])
                
                
                result["grammar_assessment"] = parsed.get("grammar_assessment", {})
                result["grammar_assessment"]["filler_count"] = len(result["grammar_assessment"].get("filler_words", []))
                
                result["vocabulary_analysis"] = parsed.get("vocabulary_analysis", {})
                result["llm_feedback"] = parsed.get("feedback", "")
                
                
                ling_score = parsed.get("linguistic_score", 0)
                result["overall_score"] = round((fluency_score * 0.3 + pron_score * 0.3 + ling_score * 0.4), 1)
        except Exception as e:
            result["llm_feedback"] = f"AI Analysis failed to parse: {str(e)}"
            result["overall_score"] = round((fluency_score + pron_score) / 2, 1)

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        
    return _to_python_type(result)
in ai fleunt 
def speech_to_text(audio_path, lang_code="en", return_raw=False):
    """Converts speech to text for the given language, retrying if no speech is detected."""

    while True:  # Keep asking until valid speech is detected
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

        if max(speech_array) < 0.01:  # Check if audio is silent
            print("🔇 No speech detected. Please speak again...")
            record_audio(audio_path)  # Re-record audio
            continue  # Retry transcription

        whisper_lang = WHISPER_LANG_MAP.get(lang_code, "english")

        try:
            # Use faster_whisper model for transcription
            segments, _ = model.transcribe(
                audio_path,
                language=lang_code,   # pass ISO code ("en", "fr", etc.)
                beam_size=5
            )
            raw_transcription = " ".join([seg.text for seg in segments]).strip()
        except Exception as e:
            print(f"⚠️ Whisper transcription failed: {e}")
            raw_transcription = ""

        print(f"📝 Raw Transcription ({whisper_lang}): {raw_transcription}")

        if not raw_transcription.strip():  # If empty, ask again
            print("⚠️ No valid speech detected. Please try again...")
            record_audio(audio_path)  # Re-record audio
            continue  # Retry transcription

        # --- Return cases ---
        if return_raw:
            # Just return the raw transcription without redundant translation
            return remove_repeated_words(raw_transcription.strip())

        # Return transcription as-is (caller should use transcribe_audio_file for proper detection/translation)
        return remove_repeated_words(raw_transcription.strip().lower())
