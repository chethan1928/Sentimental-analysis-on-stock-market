 
"""
Language Analysis Functions for FastAPI
Two functions: analyze_speaking() and analyze_writing()
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import whisper
import librosa

try:
    from g2p_en import G2p
    import jellyfish
    G2P_AVAILABLE = True
    g2p = G2p()
except ImportError:
    G2P_AVAILABLE = False
    g2p = None

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


IDEAL_WPM_MIN = 120
IDEAL_WPM_MAX = 160


def _to_python_type(value):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: _to_python_type(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_to_python_type(v) for v in value]
    return value

_whisper_model = None

def _get_whisper_model():
    """Load Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _get_llm_client():
    """Get Azure OpenAI client"""
    if not OPENAI_AVAILABLE:
        return None
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    
    if not endpoint or not api_key:
        return None
    
    if "/openai/" in endpoint:
        endpoint = endpoint.split("/openai/")[0]
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )


def _call_llm(prompt: str, system_prompt: str = "") -> Optional[str]:
    """Call Azure OpenAI LLM"""
    client = _get_llm_client()
    if not client:
        return None
    
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    if not deployment:
        return None
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt or "You are an expert language assessment evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"


def analyze_speaking(audio_path: str, expected_text: str) -> dict:
    """
    Analyze speaking/pronunciation from audio file.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, M4A)
        expected_text: The text that should have been spoken
    
    Returns:
        dict with transcription, pronunciation, fluency, comparison metrics
    """
    result = {
        "success": False,
        "error": None,
        "transcription": "",
        "pronunciation": {},
        "fluency": {},
        "comparison": {},
        "overall_score": 0.0
    }
    
    try:
        model = _get_whisper_model()
        whisper_result = model.transcribe(audio_path, word_timestamps=True, language="en")
        transcription = whisper_result["text"].strip()
        result["transcription"] = transcription
        
        words_with_timing = []
        if "segments" in whisper_result:
            for segment in whisper_result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        words_with_timing.append({
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"]
                        })
        
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        word_count = len(transcription.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        intervals = librosa.effects.split(y, top_db=30)
        pauses = []
        if len(intervals) > 1:
            for i in range(1, len(intervals)):
                pause_duration = (intervals[i][0] - intervals[i-1][1]) / sr
                if pause_duration > 0.3:
                    pauses.append(pause_duration)
        
        
        wpm_score = 100
        if wpm < IDEAL_WPM_MIN:
            wpm_score = max(0, 100 - (IDEAL_WPM_MIN - wpm) * 2)
        elif wpm > IDEAL_WPM_MAX:
            wpm_score = max(0, 100 - (wpm - IDEAL_WPM_MAX) * 2)
        
        pause_score = max(0, 100 - len(pauses) * 5 - sum(p for p in pauses if p > 2) * 10)
        overall_fluency = (wpm_score * 0.5 + pause_score * 0.5)
        
        result["fluency"] = {
            "duration_seconds": round(duration, 2),
            "word_count": word_count,
            "words_per_minute": round(wpm, 1),
            "wpm_score": round(wpm_score, 1),
            "pause_count": len(pauses),
            "avg_pause_duration": round(sum(pauses) / len(pauses), 2) if pauses else 0,
            "pause_score": round(pause_score, 1),
            "overall_score": round(overall_fluency, 1)
        }
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_std = np.std(pitch_values) if pitch_values else 0
        pitch_mean = np.mean(pitch_values) if pitch_values else 0
        
        if pitch_std < 20:
            intonation_score = 60
        elif pitch_std > 100:
            intonation_score = 70
        else:
            intonation_score = 85 + (pitch_std - 20) * 0.2
        intonation_score = min(100, max(0, intonation_score))
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        clarity_score = min(100, np.mean(spectral_centroids) / 30)
        
        rms = librosa.feature.rms(y=y)[0]
        volume_consistency = 100 - (np.std(rms) / np.mean(rms) * 100) if np.mean(rms) > 0 else 50
        volume_consistency = min(100, max(0, volume_consistency))
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        rhythm_consistency = 70
        if len(beats) > 2:
            beat_intervals = np.diff(beats)
            if np.mean(beat_intervals) > 0:
                rhythm_consistency = 100 - (np.std(beat_intervals) / np.mean(beat_intervals) * 50)
        rhythm_consistency = min(100, max(0, rhythm_consistency))
        
        phoneme_accuracy = 75.0
        if G2P_AVAILABLE and expected_text and transcription:
            try:
                expected_phonemes = g2p(expected_text.lower())
                spoken_phonemes = g2p(transcription.lower())
                
                expected_set = set([p for p in expected_phonemes if p.isalpha()])
                spoken_set = set([p for p in spoken_phonemes if p.isalpha()])
                
                if expected_set:
                    matched = expected_set & spoken_set
                    phoneme_accuracy = (len(matched) / len(expected_set)) * 100
            except:
                phoneme_accuracy = 75.0
        
        overall_pronunciation = (
            intonation_score * 0.2 + 
            clarity_score * 0.2 + 
            volume_consistency * 0.15 + 
            rhythm_consistency * 0.15 +
            phoneme_accuracy * 0.3
        )
        
        result["pronunciation"] = {
            "phoneme_accuracy": round(phoneme_accuracy, 1),
            "intonation_score": round(intonation_score, 1),
            "pitch_variation": round(pitch_std, 1),
            "pitch_mean": round(pitch_mean, 1),
            "clarity_score": round(min(100, clarity_score), 1),
            "volume_consistency": round(volume_consistency, 1),
            "rhythm_consistency": round(rhythm_consistency, 1),
            "tempo": round(float(tempo), 1),
            "overall_score": round(min(100, overall_pronunciation), 1)
        }
        
        if expected_text and transcription:
            def normalize(text):
                return ' '.join(text.lower().split())
            
            def get_words(text):
                return [w for w in text.lower().split() if w.isalnum() or w.replace("'", "").isalnum()]
            
            spoken_norm = normalize(transcription)
            expected_norm = normalize(expected_text)
            
            spoken_words = get_words(transcription)
            expected_words = get_words(expected_text)
            
            spoken_set = set(spoken_words)
            expected_set = set(expected_words)
            
            matched = spoken_set & expected_set
            missing = expected_set - spoken_set
            extra = spoken_set - expected_set
            
            word_match_pct = (len(matched) / len(expected_words) * 100) if expected_words else 100
            
            if G2P_AVAILABLE:
                try:
                    similarity = jellyfish.jaro_winkler_similarity(spoken_norm, expected_norm) * 100
                except:
                    similarity = word_match_pct
            else:
                similarity = word_match_pct
            
            order_score = 100
            if spoken_words and expected_words:
                matches_in_order = 0
                exp_idx = 0
                for word in spoken_words:
                    if exp_idx < len(expected_words) and word == expected_words[exp_idx]:
                        matches_in_order += 1
                        exp_idx += 1
                    elif word in expected_words[exp_idx:]:
                        try:
                            exp_idx = expected_words.index(word, exp_idx) + 1
                            matches_in_order += 1
                        except:
                            pass
                order_score = (matches_in_order / len(expected_words) * 100) if expected_words else 100
            
            completeness = (len(matched) / len(expected_words) * 100) if expected_words else 100
            overall_accuracy = (word_match_pct * 0.4 + similarity * 0.3 + order_score * 0.2 + completeness * 0.1)
            
            result["comparison"] = {
                "word_match_percentage": round(word_match_pct, 1),
                "similarity": round(similarity, 1),
                "order_accuracy": round(order_score, 1),
                "completeness": round(completeness, 1),
                "words_matched": len(matched),
                "words_expected": len(expected_words),
                "words_spoken": len(spoken_words),
                "missing_words": list(missing)[:10],
                "extra_words": list(extra)[:10],
                "overall_accuracy": round(overall_accuracy, 1)
            }
            
            result["overall_score"] = round(
                (result["pronunciation"]["overall_score"] * 0.4 +
                 result["fluency"]["overall_score"] * 0.3 +
                 result["comparison"]["overall_accuracy"] * 0.3), 1
            )
            
            # Get LLM feedback for speaking analysis
            llm_prompt = f"""You are a language assessment expert. Compare what the student SPOKE vs what they SHOULD have said.

Expected Text (Correct Answer):
{expected_text}

What Student Said (Transcription):
{transcription}

Analysis Metrics:
- Pronunciation Score: {result["pronunciation"]["overall_score"]}/100
- Fluency Score: {result["fluency"]["overall_score"]}/100
- Word Match: {word_match_pct:.1f}%
- Words Per Minute: {result["fluency"]["words_per_minute"]}
- Missing Words: {', '.join(list(missing)[:5]) if missing else 'None'}
- Extra Words: {', '.join(list(extra)[:5]) if extra else 'None'}

Provide a detailed analysis:

1. Accuracy Assessment (1-2 sentences)
   - How closely does the spoken text match the expected text?

2. Word-by-Word Analysis
   - Correct words
   - Missing words (should have said but didn't)
   - Extra words (said but shouldn't have)
   - Substituted words (said differently)

3. Pronunciation and Fluency Feedback
   - Any words that might have been mispronounced
   - Flow and natural speech patterns

4. Improvement Suggestions (2-3 bullet points)
   - Specific tips to improve

5. Score: X/100 (numerical score based on accuracy)

Be encouraging but honest. Keep response concise."""

            llm_feedback = _call_llm(llm_prompt, "You are an expert language assessment evaluator providing helpful, concise feedback.")
            result["llm_feedback"] = llm_feedback if llm_feedback else "LLM feedback not available"
            
        else:
            result["comparison"] = {"available": False}
            result["overall_score"] = round(
                (result["pronunciation"]["overall_score"] * 0.5 +
                 result["fluency"]["overall_score"] * 0.5), 1
            )
            result["llm_feedback"] = "Provide expected text to get AI feedback"
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False
    
    return _to_python_type(result)


def analyze_writing(user_text: str, topic: str, expected_cefr: str = "B1") -> dict:
    """
    Analyze writing for topic relevance and CEFR level.
    
    Args:
        user_text: The text written by the user
        topic: The topic/prompt the user was asked to write about
        expected_cefr: Expected CEFR level (A1, A2, B1, B2, C1, C2)
    
    Returns:
        dict with topic_relevance, cefr_assessment, writing_quality, feedback
    """
    result = {
        "success": False,
        "error": None,
        "topic_relevance": {},
        "cefr_assessment": {},
        "word_analysis": {},
        "fluency": {},
        "writing_quality": {},
        "feedback": {},
        "overall_score": 0.0
    }
    
    if not user_text or not topic:
        result["error"] = "Both user_text and topic are required"
        return _to_python_type(result)
    
    expected_cefr = expected_cefr.upper()
    if expected_cefr not in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        expected_cefr = "B1"
    
    prompt = f"""Analyze this writing sample. Respond ONLY with valid JSON.

**TOPIC:** {topic}
**TEXT:** {user_text}
**EXPECTED CEFR:** {expected_cefr}

JSON Response:
{{
    "topic_relevance": {{
        "score": <0-100>,
        "is_on_topic": <true/false>
    }},
    "cefr_assessment": {{
        "level_match": <true/false>,
        "vocabulary_level": "<A1/A2/B1/B2/C1/C2>",
        "grammar_level": "<A1/A2/B1/B2/C1/C2>",
        "too_simple_words": ["word1"],
        "too_complex_words": ["word1"],
        "suggested_vocabulary": ["better_word1"],
        "explanation": "Why vocabulary matches or doesn't match {expected_cefr}"
    }},
    "word_analysis": {{
        "total_words": <number>,
        "misspelled_words": [{{"word": "wrong", "correction": "correct"}}],
        "grammar_errors": [{{"error": "text", "correction": "fixed"}}]
    }},
    "fluency": {{
        "flow_score": <0-100>,
        "connectors_used": ["and", "but"]
    }},
    "writing_quality": {{
        "grammar_score": <0-100>,
        "spelling_score": <0-100>,
        "punctuation_score": <0-100>,
        "vocabulary_score": <0-100>,
        "overall_score": <0-100>
    }},
    "feedback": {{
        "strengths": ["strength1"],
        "improvements": ["area1"],
        "suggestions": ["tip1"],
        "corrected_text": "Full corrected text"
    }},
    "overall_score": <0-100>
}}

CEFR Guidelines:
- A1: Basic (I, you, like, good)
- A2: Simple (weekend, friend, usually)
- B1: Abstract (opinion, experience)
- B2: Sophisticated (significant, consequently)
- C1: Advanced (unprecedented, nuanced)
- C2: Near-native (quintessential, paradigm)

Find ALL errors. JSON only."""

    llm_response = _call_llm(prompt, "You are an expert CEFR language assessor. Respond only with valid JSON.")
    
    if not llm_response or llm_response.startswith("LLM Error"):
        result["error"] = llm_response or "LLM not available"
        
        word_count = len(user_text.split())
        sentence_count = user_text.count('.') + user_text.count('!') + user_text.count('?')
        avg_word_length = sum(len(w) for w in user_text.split()) / max(word_count, 1)
        
        if avg_word_length < 4 and word_count < 50:
            detected_level = "A1"
        elif avg_word_length < 5 and word_count < 100:
            detected_level = "A2"
        elif avg_word_length < 5.5:
            detected_level = "B1"
        elif avg_word_length < 6:
            detected_level = "B2"
        elif avg_word_length < 6.5:
            detected_level = "C1"
        else:
            detected_level = "C2"
        
        basic_score = min(100, word_count * 2)
        
        result["topic_relevance"] = {
            "score": 70,
            "is_on_topic": True,
            "feedback": "Unable to fully assess - LLM unavailable"
        }
        result["cefr_assessment"] = {
            "detected_level": detected_level,
            "expected_level": expected_cefr,
            "level_match": detected_level == expected_cefr,
            "explanation": "Basic heuristic assessment (LLM unavailable)"
        }
        result["writing_quality"] = {
            "grammar_score": basic_score,
            "vocabulary_score": basic_score,
            "coherence_score": basic_score,
            "task_achievement_score": basic_score,
            "overall_score": basic_score
        }
        result["feedback"] = {
            "strengths": ["Text was provided"],
            "improvements": ["LLM analysis unavailable for detailed feedback"],
            "suggestions": ["Configure Azure OpenAI for full assessment"]
        }
        result["overall_score"] = basic_score
        result["success"] = True
        return _to_python_type(result)
    
    try:
        import json
        
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = llm_response[json_start:json_end]
            parsed = json.loads(json_str)
            
            result["topic_relevance"] = parsed.get("topic_relevance", {})
            result["cefr_assessment"] = parsed.get("cefr_assessment", {})
            result["word_analysis"] = parsed.get("word_analysis", {})
            result["fluency"] = parsed.get("fluency", {})
            result["writing_quality"] = parsed.get("writing_quality", {})
            result["feedback"] = parsed.get("feedback", {})
            result["overall_score"] = parsed.get("overall_score", 0)
            result["success"] = True
        else:
            result["error"] = "Could not parse LLM response as JSON"
            result["raw_response"] = llm_response
            
    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {str(e)}"
        result["raw_response"] = llm_response
    except Exception as e:
        result["error"] = str(e)
    
    return _to_python_type(result)


if __name__ == "__main__":
    print("Testing analyze_writing...")
    writing_result = analyze_writing(
        user_text="I like to play football with my friends. We play every weekend in the park. It is very fun.",
        topic="Write about your favorite hobby",
        expected_cefr="A2"
    )
    print(f"Writing Analysis Success: {writing_result['success']}")
    if writing_result['success']:
        print(f"Overall Score: {writing_result['overall_score']}")
        print(f"CEFR Detected: {writing_result['cefr_assessment'].get('detected_level', 'N/A')}")
    else:
        print(f"Error: {writing_result['error']}")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from analysis_functions import analyze_speaking, analyze_writing

app = FastAPI(
    title="Language Analysis API" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WritingRequest(BaseModel):
    user_text: str
    topic: str
    expected_cefr: str = "B1"


class WritingResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    topic_relevance: dict = {}
    cefr_assessment: dict = {}
    word_analysis: dict = {}
    fluency: dict = {}
    writing_quality: dict = {}
    feedback: dict = {}
    overall_score: float = 0.0


class SpeakingResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    transcription: str = ""
    pronunciation: dict = {}
    fluency: dict = {}
    comparison: dict = {}
    llm_feedback: str = ""
    overall_score: float = 0.0


 
@app.post("/speaking", response_model=SpeakingResponse)
async def speaking_analysis(
    audio: UploadFile = File(..., description="Audio file (MP3, WAV, M4A)"),
    expected_text: str = Form(..., description="The text that should have been spoken")
):
    """ 
    """
    allowed_types = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a"]
    
    suffix = os.path.splitext(audio.filename)[1].lower()
    if suffix not in [".mp3", ".wav", ".m4a"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use MP3, WAV, or M4A")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = analyze_speaking(tmp_path, expected_text)
        
        os.unlink(tmp_path)
        
        return SpeakingResponse(**result)
        
    except Exception as e:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/writing", response_model=WritingResponse)
async def writing_analysis(request: WritingRequest):
    """
    Analyze writing for topic relevance and CEFR level.
    
     """
    if not request.user_text or not request.topic:
        raise HTTPException(status_code=400, detail="user_text and topic are required")
    
    try:
        result = analyze_writing(
            user_text=request.user_text,
            topic=request.topic,
            expected_cefr=request.expected_cefr
        )
        
        return WritingResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



 pip install fastapi uvicorn python-multipart python-dotenv openai openai-whisper librosa numpy g2p_en jellyfish
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

