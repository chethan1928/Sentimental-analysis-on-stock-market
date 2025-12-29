
def analyze_speaking_advanced(audio_path: str, target_cefr: str = "B1") -> dict:
    """
    Advanced speaking analysis: transcription, fluency, pronunciation, 
    grammar errors, filler words, and CEFR vocabulary assessment.
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
        # 1. Transcription
        model = _get_whisper_model()
        segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True, language="en")
        segments = list(segments)
        transcription = " ".join([s.text.strip() for s in segments]).strip()
        result["transcription"] = transcription
        
        if not transcription:
            result["error"] = "No speech detected"
            return _to_python_type(result)

        # 2. Acoustic Analysis (Fluency & Pronunciation)
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(transcription.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        # Fluency calculation
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
        
        # Pronunciation calculation
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

        # 3. LLM Deep Analysis (Grammar, Fillers, Vocab)
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
                
                # Merge into result
                result["grammar_assessment"] = parsed.get("grammar_assessment", {})
                result["grammar_assessment"]["filler_count"] = len(result["grammar_assessment"].get("filler_words", []))
                
                result["vocabulary_analysis"] = parsed.get("vocabulary_analysis", {})
                result["llm_feedback"] = parsed.get("feedback", "")
                
                # Overall Score: Balanced average of Fluency, Pronunciation, and linguistic quality
                ling_score = parsed.get("linguistic_score", 0)
                result["overall_score"] = round((fluency_score * 0.3 + pron_score * 0.3 + ling_score * 0.4), 1)
        except Exception as e:
            result["llm_feedback"] = f"AI Analysis failed to parse: {str(e)}"
            result["overall_score"] = round((fluency_score + pron_score) / 2, 1)

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        
    return _to_python_type(result)
