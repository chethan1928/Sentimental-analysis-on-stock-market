   async def get_pronunciation_vocab(self, set_number: int = None) -> list:
        """Fetch pronunciation vocabulary from database, optionally filtered by set"""
        await self.init_db()
        async with async_session() as sess:
            if set_number:
                result = await sess.execute(
                    text("SELECT word, meaning_en, sentences FROM pronunciation_vocab WHERE set_number = :set_num"),
                    {"set_num": set_number}
                )
            else:
                result = await sess.execute(
                    text("SELECT word, meaning_en, sentences FROM pronunciation_vocab")
                )
            rows = result.fetchall()
            vocab_list = []
            for r in rows:
                # Handle sentences - may be list or JSON string (possibly doubly-serialized)
                sentences_raw = r[2]
                if isinstance(sentences_raw, list):
                    sentences = sentences_raw
                elif isinstance(sentences_raw, str):
                    try:
                        parsed = json.loads(sentences_raw)
                        # Handle doubly-serialized JSON string
                        if isinstance(parsed, str):
                            parsed = json.loads(parsed)
                        sentences = parsed if isinstance(parsed, list) else []
                    except (json.JSONDecodeError, TypeError):
                        sentences = []
                else:
                    sentences = []
                
                vocab_list.append({
                    "word": r[0],
                    "meaning_en": r[1] or "",
                    "sentences": sentences
                })
            return vocab_list
