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
