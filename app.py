async def transcribe_audio_file(audio_file: UploadFile, target_lang: str = "en") -> str:
    """Helper function to transcribe audio file using Whisper - eliminates code duplication"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        shutil.copyfileobj(audio_file.file, tmp)
        temp_upload = tmp.name
    
    audio_path = None
    try:
        audio = AudioSegment.from_file(temp_upload)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_path = temp_upload.replace('.tmp', '_converted.wav')
        audio.export(audio_path, format="wav")

        languages_data = load_language_mapping()
        valid_whisper_codes = set(languages_data.values()) if languages_data else {"en"}
        
        if target_lang.lower() in languages_data:
            whisper_lang = languages_data[target_lang.lower()]
        elif target_lang in valid_whisper_codes:
            whisper_lang = target_lang
        else:
            whisper_lang = "en"
        
        segments, _ = await asyncio.to_thread(_whisper_model.transcribe, audio_path, language=whisper_lang)
        user_text = " ".join([seg.text for seg in segments]).strip()
        
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
