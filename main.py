from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from legal_advisor_prompt import LEGAL_ADVISOR_PROMPT
from similar_case_prompt import SIMILAR_CASE_PROMPT
import re
import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCDrAaSPtpPV54cyTQ1ew9dO3OG9LnbsJg")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class CaseRequest(BaseModel):
    service: str
    subject: str
    description: str

class ChatRequest(BaseModel):
    subject: str
    description: str

class ChatFollowupRequest(BaseModel):
    question: str
    subject: str
    description: str

# Helpers
def sanitize_output(text: str) -> str:
    return re.sub(r"[\*\$#@&]", "", text)

def construct_prompt(service, subject, description):
    if service == "legal_advisor":
        return LEGAL_ADVISOR_PROMPT + f"\n\nCase Subject: {subject}\n\nCase Description: {description}"
    else:
        return SIMILAR_CASE_PROMPT + f"\n\nCase Subject: {subject}\n\nCase Description: {description}"

def generate_gpt_response(prompt: str, max_output_tokens=2500) -> str:
    """Generate response using Google Gemini 2.0"""
    try:
        response = genai.generate(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_output_tokens=max_output_tokens,
            candidate_count=1,
            prompt=prompt
        )
        # Gemini returns text in response.result[0].output_text
        return response.candidates[0].output_text
    except Exception as e:
        print("Error generating response:", e)
        return "⚠️ Error generating response."

def generate_quick_questions(case_description: str):
    """Generate 5 quick questions related to the case using Gemini"""
    prompt = f"""
    Based on this legal case description:
    "{case_description}"

    Generate 5 specific questions that the user might want to ask about their case.
    Make each question practical and relevant to their specific situation.
    Format each question as a concise, clear question that would help them understand their rights or next steps.

    Return only the questions, one per line.
    """
    response = generate_gpt_response(prompt, max_output_tokens=500)
    questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Based on')]
    
    # Clean numbering if present
    clean_questions = []
    for q in questions:
        cleaned = q.lstrip('0123456789.-) ').strip()
        if cleaned and not cleaned.startswith('Q'):
            clean_questions.append(cleaned)
    return clean_questions[:5]  # return max 5 questions

# FastAPI Endpoints
@app.post("/generate")
async def generate_response(request: CaseRequest):
    prompt = construct_prompt(request.service, request.subject, request.description)
    message = generate_gpt_response(prompt)
    return {"response": sanitize_output(message)}

@app.post("/suggest_questions")
async def suggest_questions(req: ChatRequest):
    case_description = f"Subject: {req.subject}\nDescription: {req.description}"
    questions = generate_quick_questions(case_description)
    return {"questions": questions}

@app.post("/chat_followup")
async def chat_followup(request: ChatFollowupRequest):
    """Handle follow-up questions in chat format"""
    message = generate_gpt_response(request.question, max_output_tokens=1500)
    return {"response": sanitize_output(message)}
