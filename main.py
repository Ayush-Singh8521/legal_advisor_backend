from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from legal_advisor_prompt import LEGAL_ADVISOR_PROMPT
from similar_case_prompt import SIMILAR_CASE_PROMPT
import re
import requests
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_DEPLOYMENT_ENDPOINT_GPT4 = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4")
OPENAI_DEPLOYMENT_NAME_GPT4 = os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4")
OPENAI_API_KEY_GPT4 = os.getenv("OPENAI_API_KEY_GPT4")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def sanitize_output(text: str) -> str:
    return re.sub(r"[\*\$#@&]", "", text)

def construct_prompt(service, subject, description):
    if service == "legal_advisor":
        return LEGAL_ADVISOR_PROMPT + f"\n\nCase Subject: {subject}\n\nCase Description: {description}"
    else:
        return SIMILAR_CASE_PROMPT + f"\n\nCase Subject: {subject}\n\nCase Description: {description}"

def generate_gpt_response(prompt, max_tokens=2500):
    """Helper function to generate GPT response"""
    url = f"{OPENAI_DEPLOYMENT_ENDPOINT_GPT4}/openai/deployments/{OPENAI_DEPLOYMENT_NAME_GPT4}/chat/completions?api-version={OPENAI_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": OPENAI_API_KEY_GPT4
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert legal assistant specialized in Indian law."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    
    if 'choices' in result and len(result['choices']) > 0:
        return result['choices'][0]['message']['content']
    else:
        raise Exception("Failed to generate response")

def generate_quick_questions(case_description):
    """Generate 5 quick questions related to the case"""
    prompt = f"""
    Based on this legal case description:
    "{case_description}"
    
    Generate 5 specific questions that the user might want to ask about their case. 
    Make each question practical and relevant to their specific situation.
    Format each question as a concise, clear question that would help them understand their rights or next steps.
    
    Return only the questions, one per line.
    """
    
    try:
        response = generate_gpt_response(prompt, max_tokens=500)
        questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Based on')]
        # Clean up questions (remove numbering if present)
        clean_questions = []
        for q in questions:
            # Remove number prefixes like "1.", "1)", etc.
            cleaned = q.lstrip('0123456789.-) ').strip()
            if cleaned and not cleaned.startswith('Q'):
                clean_questions.append(cleaned)
        return clean_questions[:5]  # Return max 5 questions
    except:
        # Default questions if generation fails
        return [
            "What are my legal rights in this situation?",
            "What documents do I need to collect?",
            "How long do I have to take legal action?",
            "What are the possible outcomes of this case?",
            "Should I consult a lawyer immediately?"
        ]

@app.post("/generate")
async def generate_response(request: CaseRequest):
    prompt = construct_prompt(request.service, request.subject, request.description)

    try:
        message = generate_gpt_response(prompt)
    except:
        message = "⚠️ Error generating response."

    return {"response": sanitize_output(message)}

@app.post("/suggest_questions")
async def suggest_questions(req: ChatRequest):
    """Generate quick questions using the new implementation"""
    case_description = f"Subject: {req.subject}\nDescription: {req.description}"
    questions = generate_quick_questions(case_description)
    return {"questions": questions}

@app.post("/chat_followup")
async def chat_followup(request: ChatFollowupRequest):
    """Handle follow-up questions in chat format"""
    
    try:
        message = generate_gpt_response(request.question, max_tokens=1500)
    except:
        message = "⚠️ Error generating response. Please try again."

    return {"response": sanitize_output(message)}