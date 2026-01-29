 
import asyncio
import json
import re
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from openai import AzureOpenAI
 
 
)

app = FastAPI(title="Test Discussion Summary API")

# Schema matching otherpc_faq.py
class Conversation(BaseModel):
    conversation: str
    category: Optional[str] = "General"
    topic: Optional[str] = "Discussion"

# No remove_symbols function - we keep # symbols for grammar error highlighting

async def extract_key_points_llm(text):
    """Use LLM to extract key points instead of transformers"""
    prompt = f"Summarize the key discussion points in 2-3 sentences:\n\n{text}"
    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
        model=AZURE_OPENAI_DEPLOYMENT
    )
    return response.choices[0].message.content

async def generate_response_from_groq_AI_Sugesstion(prompt):
    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.85,
        max_tokens=30000,
        model=AZURE_OPENAI_DEPLOYMENT
    )
    return response.choices[0].message.content

async def provide_ai_suggestions(conversation_so_far):
    suggestion_prompt = f"""
    {{
      "role": "Expert GD Evaluator",
      "task": "Analyze the human user's performance in the provided group discussion data.",
      "conversation_data": "{conversation_so_far}",
      "instructions": [
        "Strictly follow the provided conversation data. Do not invent or change the user's input.",
        "Identify the human participant labeled as 'User Input' or by their name.",
        "Ignore all data labeled as 'responds' (Aakash, Arpita, Satish, etc.).",
        "Provide a detailed evaluation of only the human's turns.",
        "Follow the 6-point structure exactly as defined below.",
        "IMPORTANT: Always use 'you' and 'your' in ALL feedback. Never use the user's name (like Vijeta) or third person (like 'the user'). Write feedback as if speaking directly to the person."
      ],
      "output_structure": {{
        "1. Performance Snapshot": {{
          "feedback": "Provide an overall summary of your participation in the conversation. Evaluate how active you were, whether your ideas were relevant to the topic, and how well you contributed to the discussion."
        }},
        "2. Content & Ideas": {{
          "Turn 1": {{
            "Input": "[Exact User text]",
            "Feedback": "[Quality of idea]",
            "Improved": "[Better version]"
          }},
          "Turn n": {{
            "Input": "[Exact User text]",
            "Feedback": "[Quality of idea]",
            "Improved": "[Better version]"
          }}
        }},
        "3. Communication Skills": {{
          "Turn 1": {{
            "Input": "[Exact User text]",
            "Feedback": "Assess how clear and coherent your communication was in this turn. Did you use filler words or repeat points unnecessarily?",
            "Improved": "Suggest ways to improve sentence clarity, avoid fillers, and improve communication flow."
          }},
          "Turn n": {{
            "Input": "[Exact User text]",
            "Feedback": "Assess how clear and coherent your communication was in this turn. Did you use filler words or repeat points unnecessarily?",
            "Improved": "Suggest ways to improve sentence clarity, avoid fillers, and improve communication flow."
          }}
        }},
        "4. Language & Grammar": {{
          "Turn 1": {{
            "Input": "[Exact User text for turn 1]",
            "Mistakes": [
              {{
                "Quote": "I #goed# to the store.",
                "Error": "Verb Tense",
                "Correct": "I #went# to the store."
              }}
            ],
            "Corrected Sentence": "[Full sentence with all grammar errors fixed]"
          }},
          "Turn n": {{
            "Input": "[Exact User text for turn n]",
            "Mistakes": [
              {{
                "Quote": "She #don't# like it.",
                "Error": "Subject-Verb Agreement",
                "Correct": "She #doesn't# like it."
              }},
              {{
                "Quote": "I #goes# there #everyday#.",
                "Error": "Subject-Verb Agreement, Adverb",
                "Correct": "I #go# there #every day#."
              }}
            ],
            "Corrected Sentence": "[Full sentence with all grammar errors fixed]"
          }},
          "ANALYZE_FOR": {{
            "1. FILLER WORDS": "Detect ALL of these if present: um, uh, uhh, er, err, ah, ahh, like, you know, I mean, basically, actually, literally, so, well, kind of, sort of",
            "2. GRAMMAR ERRORS": "Check for verb tense, subject-verb agreement, articles, prepositions, word order, pronouns, plurals, comparatives",
            "3. WORD SUGGESTIONS": "Suggest stronger alternatives for weak/basic words (e.g., good -> excellent, bad -> challenging)"
          }},
          "STRICTLY_REQUIRED": {{
            "1. Quote field": "MUST contain the exact phrase/sentence the user said with the #wrong# word wrapped in # symbols",
            "2. Correct field": "MUST contain the corrected phrase/sentence with the #correct# word wrapped in # symbols",
            "3. Each Turn": "MUST list ALL grammar/filler mistakes found in that turn under Mistakes array",
            "4. Corrected Sentence": "MUST show the full corrected version of the user's input for that turn"
          }},
          "CRITICAL": {{
            "ALWAYS wrap wrong words with # in Quote": "e.g., I #goed# to store",
            "ALWAYS wrap correct words with # in Correct": "e.g., I #went# to the store",
            "Group mistakes by Turn": "Each Turn should have its own Mistakes array"
          }}
        }},
        "5. Participation & Interaction": {{
          "Turn 1": {{
            "Input": "[Exact User text]",
            "Feedback": "How they responded to bots/flow of GD",
            "Improved": "Better interactive phrase"
          }},
          "Turn n": {{
            "Input": "[Exact User text]",
            "Feedback": "How they responded to bots/flow of GD",
            "Improved": "Better interactive phrase"
          }}
        }},
        "6. Growth Plan": {{
          "Your Strengths": [
            "List of strengths based on the conversation"
          ],
          "Your Mistakes": [
            "List of real mistakes captured from the specific turns above"
          ],
          "Action Tips": [
            "Specific actionable steps to improve for the next session"
          ]
        }}
      }},
      "strict_rule": "COMPLETELY IGNORE: spelling errors, punctuation mistakes, capitalization errors. Focus ONLY on: verb tense, subject-verb agreement, articles, prepositions, word order, filler words (um, uh, like, you know, I mean, basically, actually, literally)."
    }}
    """
    
    suggestion = await generate_response_from_groq_AI_Sugesstion(suggestion_prompt)
    # Return as-is to keep # symbols for grammar error highlighting
    return suggestion

@app.post("/discussion_summary/")
async def discussion_summary(data: Conversation):
    """Same endpoint structure as otherpc_faq.py - no auth required"""
    conversation_so_far = data.conversation
    
    # Extract key points using LLM
    key_points = await extract_key_points_llm(conversation_so_far)
    speak_sentence = f"Key points of the conversation so far: {key_points}"
    
    # Get AI suggestions
    ai_response = await provide_ai_suggestions(conversation_so_far)
    # Note: ai_response is NOT added to speak_sentence to keep transcript clean
    
    # Parse response
    parsed_response = None
    try:
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if json_match:
            parsed_response = json.loads(json_match.group())
    except:
        pass
    
    # Mock audio URL for testing
    audio_url = "http://localhost:8182/test_audio/sample.mp3"
    
    # Build response (exact same structure as otherpc_faq.py)
    if parsed_response:
        response = {
            "speak": audio_url,
            "audio": audio_url,
            "transcript": speak_sentence,
            "1. Performance Snapshot": parsed_response.get("1. Performance Snapshot", parsed_response.get("Performance Snapshot", {})),
            "2. Content & Ideas": parsed_response.get("2. Content & Ideas", parsed_response.get("Content & Ideas", {})),
            "3. Communication Skills": parsed_response.get("3. Communication Skills", parsed_response.get("Communication Skills", {})),
            "4. Language & Grammar": parsed_response.get("4. Language & Grammar", parsed_response.get("Language & Grammar", {})),
            "5. Participation & Interaction": parsed_response.get("5. Participation & Interaction", parsed_response.get("Participation & Interaction", {})),
            "6. Growth Plan": parsed_response.get("6. Growth Plan", parsed_response.get("Growth Plan", {}))
        }
    else:
        response = {
            "speak": audio_url,
            "audio": audio_url,
            "transcript": speak_sentence,
            "1. Performance Snapshot": {"feedback": ai_response if isinstance(ai_response, str) else ""},
            "2. Content & Ideas": {},
            "3. Communication Skills": {},
            "4. Language & Grammar": {},
            "5. Participation & Interaction": {},
            "6. Growth Plan": {"Your Strengths": [], "Your Mistakes": [], "Action Tips": []}
        }
    
    return response

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST API running at: http://localhost:8182")
    print("Endpoint: POST /discussion_summary/")
    print("="*60)
    print("\nSample payload:")
    print('{"conversation": "Vijeta input: I goes to school...", "category": "Education", "topic": "Test"}')
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8180)
