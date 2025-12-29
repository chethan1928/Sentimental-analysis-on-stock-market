"""
Language Test Analysis - Simplified Version
Analyzes audio and writing for pronunciation, fluency, and accuracy.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Audio Processing
import whisper
import librosa

# Text Analysis
try:
    from g2p_en import G2p
    import jellyfish
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False

# Azure OpenAI
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration
IDEAL_WPM_MIN = 120
IDEAL_WPM_MAX = 160
FILLER_WORDS = ["um", "uh", "er", "ah", "like", "you know", "basically", "actually", "so", "well"]


# ============================================
# LLM COMPARISON FUNCTIONS
# ============================================
def get_llm_comparison(spoken_text: str, expected_text: str, mode: str = "audio") -> str:
    """Use LLM to provide detailed comparison analysis"""
    if not OPENAI_AVAILABLE:
        return "OpenAI library not installed. Run: pip install openai"
    
    # Get credentials from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    
    if not endpoint or not api_key or not deployment:
        return "Azure OpenAI credentials not configured. Check .env file."
    
    # Extract base endpoint if full URL is provided
    # e.g., "https://xxx.openai.azure.com/openai/..." -> "https://xxx.openai.azure.com"
    if "/openai/" in endpoint:
        endpoint = endpoint.split("/openai/")[0]
    
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        if mode == "audio":
            prompt = f"""You are a language assessment expert. Compare what the student SPOKE vs what they SHOULD have said.

Expected Text (Correct Answer):
{expected_text}

What Student Said (Transcription):
{spoken_text}

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
        else:
            prompt = f"""You are a writing assessment expert. Compare the student's written text with the expected/correct text.

Expected Text (Correct Version):
{expected_text}

Student's Written Text:
{spoken_text}

Provide a detailed analysis:

1. Accuracy Assessment (1-2 sentences)
   - How closely does the written text match the expected text?

2. Word-by-Word Comparison
   - Correct words/phrases
   - Missing words/phrases
   - Extra words/phrases
   - Substituted or changed words

3. Writing Quality
   - Spelling errors
   - Grammar issues
   - Punctuation problems

4. Improvement Suggestions (2-3 bullet points)
   - Specific tips to improve writing accuracy

5. Score: X/100 (numerical score based on accuracy)

Be encouraging but honest. Keep response concise."""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are an expert language assessment evaluator providing helpful, concise feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ============================================
# CACHED MODEL LOADING
# ============================================
@st.cache_resource
def load_whisper_model():
    """Load Whisper model for transcription"""
    return whisper.load_model("base")


# ============================================
# AUDIO ANALYSIS FUNCTIONS
# ============================================
def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Whisper"""
    model = load_whisper_model()
    result = model.transcribe(audio_path, word_timestamps=True, language="en")
    
    words_with_timing = []
    if "segments" in result:
        for segment in result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    words_with_timing.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })
    
    return {
        "text": result["text"].strip(),
        "words": words_with_timing
    }


def analyze_fluency(audio_path: str, transcription: dict) -> dict:
    """Analyze fluency metrics from audio"""
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    
    text = transcription["text"]
    word_count = len(text.split())
    
    # Speaking pace (WPM)
    wpm = (word_count / duration) * 60 if duration > 0 else 0
    
    # Pause detection
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    if len(intervals) > 1:
        for i in range(1, len(intervals)):
            pause_duration = (intervals[i][0] - intervals[i-1][1]) / sr
            if pause_duration > 0.3:
                pauses.append(pause_duration)
    
    # Filler word detection
    filler_count = sum(text.lower().count(f) for f in FILLER_WORDS)
    
    # Score calculations
    wpm_score = 100
    if wpm < IDEAL_WPM_MIN:
        wpm_score = max(0, 100 - (IDEAL_WPM_MIN - wpm) * 2)
    elif wpm > IDEAL_WPM_MAX:
        wpm_score = max(0, 100 - (wpm - IDEAL_WPM_MAX) * 2)
    
    pause_score = max(0, 100 - len(pauses) * 5 - sum(p for p in pauses if p > 2) * 10)
    filler_score = max(0, 100 - filler_count * 10)
    
    overall_fluency = (wpm_score * 0.4 + pause_score * 0.3 + filler_score * 0.3)
    
    return {
        "duration_seconds": round(duration, 2),
        "word_count": word_count,
        "words_per_minute": round(wpm, 1),
        "wpm_score": round(wpm_score, 1),
        "pause_count": len(pauses),
        "pause_score": round(pause_score, 1),
        "filler_word_count": filler_count,
        "filler_score": round(filler_score, 1),
        "overall_fluency_score": round(overall_fluency, 1)
    }


def analyze_pronunciation(audio_path: str, transcription: dict) -> dict:
    """Analyze pronunciation using acoustic features"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    pitch_std = np.std(pitch_values) if pitch_values else 0
    
    # Intonation score
    if pitch_std < 20:
        intonation_score = 60
    elif pitch_std > 100:
        intonation_score = 70
    else:
        intonation_score = 85 + (pitch_std - 20) * 0.2
    intonation_score = min(100, max(0, intonation_score))
    
    # Clarity score
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    clarity_score = min(100, np.mean(spectral_centroids) / 30)
    
    # Volume consistency
    rms = librosa.feature.rms(y=y)[0]
    volume_consistency = 100 - (np.std(rms) / np.mean(rms) * 100) if np.mean(rms) > 0 else 50
    volume_consistency = min(100, max(0, volume_consistency))
    
    # Rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    rhythm_consistency = 70
    if len(beats) > 2:
        beat_intervals = np.diff(beats)
        if np.mean(beat_intervals) > 0:
            rhythm_consistency = 100 - (np.std(beat_intervals) / np.mean(beat_intervals) * 50)
    rhythm_consistency = min(100, max(0, rhythm_consistency))
    
    # Overall pronunciation score
    overall = (intonation_score * 0.3 + clarity_score * 0.3 + 
               volume_consistency * 0.2 + rhythm_consistency * 0.2)
    
    return {
        "intonation_score": round(intonation_score, 1),
        "clarity_score": round(min(100, clarity_score), 1),
        "volume_consistency": round(volume_consistency, 1),
        "rhythm_consistency": round(rhythm_consistency, 1),
        "overall_pronunciation_score": round(min(100, overall), 1)
    }


def compare_text(spoken_text: str, expected_text: str) -> dict:
    """Compare spoken/written text with expected text"""
    if not expected_text or not spoken_text:
        return {"comparison_available": False}
    
    def normalize(text):
        return ' '.join(text.lower().split())
    
    def get_words(text):
        return [w for w in text.lower().split() if w.isalnum() or w.replace("'", "").isalnum()]
    
    spoken_norm = normalize(spoken_text)
    expected_norm = normalize(expected_text)
    
    spoken_words = get_words(spoken_text)
    expected_words = get_words(expected_text)
    
    # Word matching
    spoken_set = set(spoken_words)
    expected_set = set(expected_words)
    
    matched = spoken_set & expected_set
    missing = expected_set - spoken_set
    extra = spoken_set - expected_set
    
    # Calculate percentages
    word_match_pct = (len(matched) / len(expected_words) * 100) if expected_words else 100
    
    # Similarity score
    if G2P_AVAILABLE:
        try:
            similarity = jellyfish.jaro_winkler_similarity(spoken_norm, expected_norm) * 100
        except:
            similarity = word_match_pct
    else:
        similarity = word_match_pct
    
    # Order accuracy
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
    
    # Completeness
    completeness = (len(matched) / len(expected_words) * 100) if expected_words else 100
    
    # Overall accuracy
    overall = (word_match_pct * 0.4 + similarity * 0.3 + order_score * 0.2 + completeness * 0.1)
    
    return {
        "comparison_available": True,
        "word_match_percentage": round(word_match_pct, 1),
        "similarity": round(similarity, 1),
        "order_accuracy": round(order_score, 1),
        "completeness": round(completeness, 1),
        "overall_accuracy": round(overall, 1),
        "words_matched": len(matched),
        "words_expected": len(expected_words),
        "words_spoken": len(spoken_words),
        "missing_words": list(missing)[:10],
        "extra_words": list(extra)[:10],
        "spoken_text": spoken_text,
        "expected_text": expected_text
    }


# ============================================
# METRIC EXPLANATIONS
# ============================================
METRIC_EXPLANATIONS = {
    # Fluency Metrics
    "words_per_minute": "Speaking speed measured in words per minute. Ideal range is 120-160 WPM for clear speech.",
    "wpm_score": "Score based on speaking pace. 100 = ideal speed (120-160 WPM). Lower scores indicate too fast or too slow speech.",
    "pause_count": "Number of significant pauses (over 0.3 seconds) detected in speech.",
    "pause_score": "Score based on pauses. Fewer and shorter pauses = higher score. Long pauses (>2s) reduce score more.",
    "filler_word_count": "Count of filler words like 'um', 'uh', 'like', 'you know', etc.",
    "filler_score": "Score based on filler words. Each filler word reduces score by 10 points.",
    "overall_fluency_score": "Combined fluency score: 40% speaking pace + 30% pause quality + 30% filler word avoidance.",
    
    # Pronunciation Metrics
    "intonation_score": "Measures pitch variation in speech. Natural speech has moderate variation (not monotone, not erratic).",
    "clarity_score": "Measures how clearly words are articulated based on spectral analysis of the audio.",
    "volume_consistency": "Measures how consistent the speaking volume is. Consistent volume = higher score.",
    "rhythm_consistency": "Measures regularity of speech rhythm. Steady rhythm = higher score.",
    "overall_pronunciation_score": "Combined pronunciation score: 30% intonation + 30% clarity + 20% volume + 20% rhythm.",
    
    # Comparison Metrics
    "word_match_percentage": "Percentage of expected words that were correctly spoken/written.",
    "similarity": "Text similarity using Jaro-Winkler algorithm. Considers character-level similarity.",
    "order_accuracy": "How well the word order matches the expected text. Out-of-order words reduce score.",
    "completeness": "Percentage of expected words that were included in the response.",
    "overall_accuracy": "Combined accuracy score: 40% word match + 30% similarity + 20% order + 10% completeness."
}


# ============================================
# STREAMLIT UI
# ============================================
def main():
    st.set_page_config(
        page_title="Language Test Analyzer",
        layout="wide"
    )
    
    st.title("Language Test Analyzer")
    st.caption("Analyze audio and writing for pronunciation, fluency, and accuracy")

    
    # Mode Selection
    mode = st.radio(
        "Select Test Mode:",
        ["Audio Analysis (Speech to Text)", "Writing Analysis (Text Comparison)"],
        horizontal=True
    )
    
    st.divider()
    
    # ==========================================
    # AUDIO ANALYSIS MODE
    # ==========================================
    if "Audio" in mode:
        st.header("Audio Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            audio_file = st.file_uploader(
                "Upload Audio Recording",
                type=["mp3", "wav", "m4a"]
            )
            
            expected_text = st.text_area(
                "Expected Text (for comparison)",
                placeholder="Enter the text that should have been spoken...",
                height=150
            )
        
        with col2:
            st.caption("Upload audio and optionally enter expected text for comparison analysis.")
        
        if st.button("Analyze Audio", type="primary", use_container_width=True):
            if not audio_file:
                st.error("Please upload an audio file!")
                return
            
            with st.spinner("Processing audio..."):
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                
                try:
                    # Transcribe
                    transcription = transcribe_audio(tmp_path)
                    
                    # Analyze fluency
                    fluency = analyze_fluency(tmp_path, transcription)
                    
                    # Analyze pronunciation
                    pronunciation = analyze_pronunciation(tmp_path, transcription)
                    
                    # Compare with expected text
                    comparison = compare_text(transcription["text"], expected_text)
                    
                    # Get AI analysis automatically if credentials are configured and expected text provided
                    llm_result = None
                    if comparison.get("comparison_available"):
                        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
                        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
                        if endpoint and api_key and deployment:
                            llm_result = get_llm_comparison(
                                transcription["text"], 
                                expected_text, 
                                mode="audio"
                            )
                    
                    st.success("Analysis complete!")
                    
                    # Display Results
                    st.divider()
                    st.header("Results")
                    
                    # Transcription
                    st.subheader("Transcription")
                    st.info(transcription["text"] if transcription["text"] else "No speech detected")
                    
                    # Score Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pronunciation", f"{pronunciation['overall_pronunciation_score']}/100")
                    with col2:
                        st.metric("Fluency", f"{fluency['overall_fluency_score']}/100")
                    with col3:
                        if comparison.get("comparison_available"):
                            st.metric("Accuracy", f"{comparison['overall_accuracy']}/100")
                        else:
                            overall = (pronunciation['overall_pronunciation_score'] + fluency['overall_fluency_score']) / 2
                            st.metric("Overall", f"{overall:.1f}/100")
                    
                    # Detailed tabs
                    tabs = st.tabs(["Comparison", "Pronunciation Details", "Fluency Details"])
                    
                    with tabs[0]:
                        if comparison.get("comparison_available"):
                            st.subheader("Text Comparison")
                            
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**What was spoken:**")
                                st.warning(comparison["spoken_text"])
                            with c2:
                                st.markdown("**Expected text:**")
                                st.success(comparison["expected_text"])
                            
                            st.divider()
                            
                            # Metrics with explanations
                            st.subheader("Comparison Metrics")
                            
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                st.metric("Word Match", f"{comparison['word_match_percentage']}%")
                                st.caption(METRIC_EXPLANATIONS["word_match_percentage"])
                            with m2:
                                st.metric("Similarity", f"{comparison['similarity']}%")
                                st.caption(METRIC_EXPLANATIONS["similarity"])
                            with m3:
                                st.metric("Order Accuracy", f"{comparison['order_accuracy']}%")
                                st.caption(METRIC_EXPLANATIONS["order_accuracy"])
                            with m4:
                                st.metric("Completeness", f"{comparison['completeness']}%")
                                st.caption(METRIC_EXPLANATIONS["completeness"])
                            
                            st.divider()
                            
                            if comparison["missing_words"]:
                                st.warning(f"Missing words: {', '.join(comparison['missing_words'])}")
                            if comparison["extra_words"]:
                                st.info(f"Extra words: {', '.join(comparison['extra_words'])}")
                            
                            st.progress(int(comparison['overall_accuracy']) / 100)
                            st.markdown(f"**Overall Accuracy: {comparison['overall_accuracy']}/100**")
                            st.caption(METRIC_EXPLANATIONS["overall_accuracy"])
                            
                            # AI Analysis - shown automatically
                            st.divider()
                            st.subheader("AI Analysis")
                            if llm_result:
                                st.markdown(llm_result)
                            else:
                                st.info("AI analysis will appear after configuring Azure OpenAI in .env file")
                        else:
                            st.info("Enter expected text above to compare with transcription")
                    
                    with tabs[1]:
                        st.subheader("Pronunciation Metrics")
                        
                        p1, p2 = st.columns(2)
                        with p1:
                            st.metric("Intonation", f"{pronunciation['intonation_score']}/100")
                            st.caption(METRIC_EXPLANATIONS["intonation_score"])
                            
                            st.metric("Clarity", f"{pronunciation['clarity_score']}/100")
                            st.caption(METRIC_EXPLANATIONS["clarity_score"])
                        with p2:
                            st.metric("Volume Consistency", f"{pronunciation['volume_consistency']}/100")
                            st.caption(METRIC_EXPLANATIONS["volume_consistency"])
                            
                            st.metric("Rhythm", f"{pronunciation['rhythm_consistency']}/100")
                            st.caption(METRIC_EXPLANATIONS["rhythm_consistency"])
                        
                        st.divider()
                        st.markdown(f"**Overall Pronunciation Score: {pronunciation['overall_pronunciation_score']}/100**")
                        st.caption(METRIC_EXPLANATIONS["overall_pronunciation_score"])
                    
                    with tabs[2]:
                        st.subheader("Fluency Metrics")
                        
                        f1, f2 = st.columns(2)
                        with f1:
                            st.metric("Words Per Minute", f"{fluency['words_per_minute']}")
                            st.caption(METRIC_EXPLANATIONS["words_per_minute"])
                            
                            st.metric("WPM Score", f"{fluency['wpm_score']}/100")
                            st.caption(METRIC_EXPLANATIONS["wpm_score"])
                            
                            st.metric("Duration", f"{fluency['duration_seconds']}s")
                        with f2:
                            st.metric("Pause Count", f"{fluency['pause_count']}")
                            st.caption(METRIC_EXPLANATIONS["pause_count"])
                            
                            st.metric("Pause Score", f"{fluency['pause_score']}/100")
                            st.caption(METRIC_EXPLANATIONS["pause_score"])
                            
                            st.metric("Filler Words", f"{fluency['filler_word_count']}")
                            st.caption(METRIC_EXPLANATIONS["filler_word_count"])
                        
                        st.divider()
                        st.markdown(f"**Overall Fluency Score: {fluency['overall_fluency_score']}/100**")
                        st.caption(METRIC_EXPLANATIONS["overall_fluency_score"])
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    os.unlink(tmp_path)
    
    # ==========================================
    # WRITING ANALYSIS MODE
    # ==========================================
    else:
        st.header("Writing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Expected Text")
            expected_text = st.text_area(
                "Enter the correct/expected text:",
                placeholder="Enter the original or expected text here...",
                height=200,
                key="writing_expected"
            )
        
        with col2:
            st.subheader("User's Text")
            user_text = st.text_area(
                "Enter the user's written text:",
                placeholder="Enter what the user wrote here...",
                height=200,
                key="writing_user"
            )
        
        if st.button("Compare Texts", type="primary", use_container_width=True):
            if not expected_text or not user_text:
                st.error("Please enter both expected and user texts!")
                return
            
            with st.spinner("Comparing texts..."):
                comparison = compare_text(user_text, expected_text)
                
                # Get AI analysis automatically if credentials are configured
                llm_result = None
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
                deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
                if endpoint and api_key and deployment:
                    llm_result = get_llm_comparison(
                        user_text, 
                        expected_text, 
                        mode="writing"
                    )
                
                st.success("Comparison complete!")
                
                # Display Results
                st.divider()
                st.header("Results")
                
                # Side by side comparison
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Expected Text:**")
                    st.success(expected_text)
                with c2:
                    st.markdown("**User's Text:**")
                    st.warning(user_text)
                
                st.divider()
                
                # Scores with explanations
                st.subheader("Comparison Scores")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Overall", f"{comparison['overall_accuracy']}/100")
                    st.caption(METRIC_EXPLANATIONS["overall_accuracy"])
                with m2:
                    st.metric("Word Match", f"{comparison['word_match_percentage']}%")
                    st.caption(METRIC_EXPLANATIONS["word_match_percentage"])
                with m3:
                    st.metric("Similarity", f"{comparison['similarity']}%")
                    st.caption(METRIC_EXPLANATIONS["similarity"])
                with m4:
                    st.metric("Order", f"{comparison['order_accuracy']}%")
                    st.caption(METRIC_EXPLANATIONS["order_accuracy"])
                with m5:
                    st.metric("Complete", f"{comparison['completeness']}%")
                    st.caption(METRIC_EXPLANATIONS["completeness"])
                
                st.divider()
                
                # Word details
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.info(f"Words Expected: {comparison['words_expected']}")
                with d2:
                    st.info(f"Words Written: {comparison['words_spoken']}")
                with d3:
                    st.success(f"Words Matched: {comparison['words_matched']}")
                
                # Missing and extra words
                if comparison["missing_words"]:
                    st.warning(f"Missing words: {', '.join(comparison['missing_words'])}")
                
                if comparison["extra_words"]:
                    st.info(f"Extra words: {', '.join(comparison['extra_words'])}")
                
                # Overall progress
                st.divider()
                st.subheader("Overall Accuracy")
                st.progress(int(comparison['overall_accuracy']) / 100)
                
                # Grade
                accuracy = comparison['overall_accuracy']
                if accuracy >= 90:
                    grade = "A - Excellent!"
                elif accuracy >= 80:
                    grade = "B - Good"
                elif accuracy >= 70:
                    grade = "C - Average"
                elif accuracy >= 60:
                    grade = "D - Needs Improvement"
                else:
                    grade = "F - Poor"
                
                st.markdown(f"### Grade: {grade}")
                
                # AI Analysis - shown automatically
                st.divider()
                st.subheader("AI Analysis")
                if llm_result:
                    st.markdown(llm_result)
                else:
                    st.info("AI analysis will appear after configuring Azure OpenAI in .env file")


if __name__ == "__main__":
    main()
