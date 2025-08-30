# -*- coding: utf-8 -*-
"""
FinBuddy: FastAPI backend + single-page frontend (served from the same URL).
- LLM providers (env-configurable): IBM watsonx, Hugging Face, OpenAI, Gemini
- Finance CRUD: Savings, Investments, Taxes, Planning, Settings
- Chat history
- Static SPA (index.html + JS + CSS) served at "/"

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import random
import requests
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ----------------------------
# Config (env-driven)
# ----------------------------
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_BASE_URL = os.getenv("WATSONX_BASE_URL")
WATSONX_MODEL = os.getenv("WATSONX_MODEL", "granite-3.3-instruct")

HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1")

DUMMY_ALWAYS = os.getenv("DUMMY_ALWAYS", "false").lower() in ("1", "true", "yes")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ----------------------------
# Models
# ----------------------------
class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    sender = Column(String)  # "user" | "bot"
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SavingsGoal(Base):
    __tablename__ = "savings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    name = Column(String, nullable=False)
    target_amount = Column(Float, default=0.0)
    saved_amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Investment(Base):
    __tablename__ = "investments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    type = Column(String, default="Stock")
    symbol = Column(String, nullable=True)
    amount_invested = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class TaxRecord(Base):
    __tablename__ = "taxes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    year = Column(Integer, index=True)
    income = Column(Float, default=0.0)
    tax_paid = Column(Float, default=0.0)
    refund_expected = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint('user_id', 'year', name='uq_tax_user_year'),)

class Plan(Base):
    __tablename__ = "planning"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    goal = Column(String, nullable=False)
    target_year = Column(Integer, nullable=True)
    estimated_amount = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserSettings(Base):
    __tablename__ = "settings"
    user_id = Column(String, primary_key=True, index=True)
    name = Column(String, default="You")
    preferred_model = Column(String, default="")   # "", "ibm", "hf", "openai", "gemini"
    theme = Column(String, default="light")
    notifications = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FinBuddy Backend + Frontend (Single URL)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Serve Frontend (Single URL) ----------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = (BASE_DIR / "frontend").resolve()

# Static assets (css/js) will be served at /assets/
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

# favicon (optional)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return {"message": "Frontend not found. Keep /frontend/index.html in place."}
    return FileResponse(index_file)

# ---------- Dummy Responses ----------
DUMMY_RESPONSES = [
    "That's a sensible question — a good place to start is a budget.",
    "Try setting aside a small percentage of your income every month.",
    "Consider opening a high-yield savings account to earn interest.",
    "Automating transfers helps ensure you consistently save.",
    "Reduce discretionary spending for a few months to boost savings.",
    "You could consider a side gig for extra down payment cash.",
    "Prioritize building a 3-6 month emergency fund first.",
    "Split your savings: short-term for the down payment, long-term for retirement.",
    "Check for first-time home buyer assistance programs in your area.",
    "Downsize recurring subscriptions and put the savings into a dedicated fund.",
    "Use the 50/30/20 rule as a simple budgeting framework.",
    "Track your expenses for a month to find cuttable items.",
    "Sell unused items online and add proceeds to your down payment fund.",
    "Consider a target date and calculate how much to save monthly.",
    "If you have high-interest debt, pay that down to free up cash flow.",
    "A Certificate of Deposit (CD) ladder can protect cash while earning interest.",
    "Keep your down payment separate from everyday accounts to avoid spending it.",
    "Look into employer-matched plans first if you're saving for retirement.",
    "Try the 'no-spend' weekend challenge and move that money into savings.",
    "Reduce dining out and coffee-shop purchases for a few months.",
    "Set progressive savings goals — increase contributions yearly.",
    "Use cash-back or rewards responsibly and funnel rewards to savings.",
    "If eligible, a tax refund can be a one-time boost to your down payment.",
    "Consider budgeting apps that categorize spending automatically.",
    "Open a dedicated savings account named 'House Down Payment'.",
    "Compare mortgage programs and required down payments ahead of time.",
    "Consider a lower-cost starter home and upgrade later.",
    "Create a visual progress tracker to stay motivated.",
    "Use windfalls (bonuses, gifts) to fast-track your goal.",
    "Ask family to gift towards down payment rather than material gifts.",
    "Cut transport costs by carpooling or using public transit temporarily.",
    "Set weekly micro-goals to reinforce the habit of saving.",
    "Shop generic brands for groceries and shift savings to your goal.",
    "Negotiate recurring bills (insurance, cable) and reallocate savings.",
    "Consider a temporary budget for 6 months to accelerate progress.",
    "If renting, try negotiating lease renewal or moving to a cheaper unit.",
    "Open a money market account if you need some liquidity with interest.",
    "Explore government-subsidized loan programs for first-time buyers.",
    "Keep track of credit score improvements — better score = better mortgage rates.",
    "Avoid dipping into retirement savings unless you understand penalties.",
    "Calculate total homeownership costs (taxes, insurance, maintenance).",
    "Plan for closing costs on top of the down payment.",
    "Create a 'no-luxury' period and use the freed funds to save.",
    "Set up automatic transfers right after payday.",
    "Use a round-up savings app to add small amounts that compound.",
    "Bundle savings into a weekly transfer instead of monthly for better pace.",
    "Review your budget quarterly and adjust targets as income changes.",
    "Get pre-qualified to understand the loan amount you can get.",
    "When in doubt, consult a local financial advisor for tailored advice."
]
if len(DUMMY_RESPONSES) < 50:
    DUMMY_RESPONSES.extend(["Consider a small, recurring transfer to your savings account."] * (50 - len(DUMMY_RESPONSES)))
elif len(DUMMY_RESPONSES) > 50:
    DUMMY_RESPONSES = DUMMY_RESPONSES[:50]

# ---------- Schemas ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    prefer: Optional[str] = None  # "ibm|hf|openai|gemini"

class ChatResponse(BaseModel):
    reply: str
    source: str
    timestamp: datetime

class SavingsCreate(BaseModel):
    name: str = Field(..., example="House Down Payment")
    target_amount: float = Field(0, ge=0)
    saved_amount: float = Field(0, ge=0)

class SavingsOut(BaseModel):
    id: int
    user_id: str
    name: str
    target_amount: float
    saved_amount: float
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class InvestmentCreate(BaseModel):
    type: str = Field("Stock", example="Stock")
    symbol: Optional[str] = Field(None, example="AAPL")
    amount_invested: float = Field(0, ge=0)
    current_value: float = Field(0, ge=0)
    notes: Optional[str] = None

class InvestmentOut(BaseModel):
    id: int
    user_id: str
    type: str
    symbol: Optional[str]
    amount_invested: float
    current_value: float
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class TaxCreate(BaseModel):
    year: int = Field(..., example=2024)
    income: float = Field(0, ge=0)
    tax_paid: float = Field(0, ge=0)
    refund_expected: float = Field(0, ge=0)

class TaxOut(BaseModel):
    id: int
    user_id: str
    year: int
    income: float
    tax_paid: float
    refund_expected: float
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class PlanCreate(BaseModel):
    goal: str = Field(..., example="Retirement")
    target_year: Optional[int] = Field(None, example=2040)
    estimated_amount: float = Field(0, ge=0)
    notes: Optional[str] = None

class PlanOut(BaseModel):
    id: int
    user_id: str
    goal: str
    target_year: Optional[int]
    estimated_amount: float
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    name: Optional[str] = None
    preferred_model: Optional[str] = None
    theme: Optional[str] = None
    notifications: Optional[bool] = None

class SettingsOut(BaseModel):
    user_id: str
    name: str
    preferred_model: str
    theme: str
    notifications: bool
    updated_at: datetime
    class Config:
        from_attributes = True

# ---------- DB helpers ----------
def save_message(user_id: str, sender: str, message: str):
    db = SessionLocal()
    try:
        msg = ChatMessage(user_id=user_id, sender=sender, message=message, timestamp=datetime.utcnow())
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
    finally:
        db.close()

def get_history(user_id: str, limit: int = 200):
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.user_id == user_id)
            .order_by(ChatMessage.timestamp.asc())
            .limit(limit)
            .all()
        )
        return [{"user_id": r.user_id, "sender": r.sender, "message": r.message, "timestamp": r.timestamp.isoformat()} for r in rows]
    finally:
        db.close()

# ---------- AI callers ----------
def call_ibm_watsonx(prompt: str, timeout: int = 8) -> Optional[str]:
    if not WATSONX_API_KEY or not WATSONX_BASE_URL:
        return None
    path = os.getenv("WATSONX_GENERATE_PATH", "/v1/generate")
    url = WATSONX_BASE_URL.rstrip("/") + path
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {WATSONX_API_KEY}"}
    payload = {"model": WATSONX_MODEL, "input": prompt, "max_output_tokens": 256, "temperature": 0.7}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                if "output" in data and isinstance(data["output"], list) and data["output"]:
                    first = data["output"][0]
                    return first.get("content") or first.get("text") or str(first)
                if "generations" in data and isinstance(data["generations"], list) and data["generations"]:
                    g0 = data["generations"][0]
                    if isinstance(g0, dict) and "text" in g0:
                        return g0["text"]
                if "text" in data:
                    return data["text"]
                return str(data)
        return None
    except Exception:
        return None

def call_huggingface(prompt: str, timeout: int = 8) -> Optional[str]:
    if not HUGGINGFACE_API_KEY or not HF_MODEL:
        return None
    hf_base = os.getenv("HF_INFERENCE_BASE", "https://api-inference.huggingface.co")
    url = hf_base.rstrip("/") + f"/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.7}}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                first = data[0]
                return first.get("generated_text") or str(first)
            if isinstance(data, dict):
                return data.get("generated_text") or str(data)
        return None
    except Exception:
        return None

def call_openai(prompt: str, timeout: int = 8) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        # Use the new OpenAI client pattern if available; fallback to old SDK.
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def call_gemini(prompt: str, timeout: int = 8) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    url = os.getenv("GEMINI_BASE_URL", "https://gemini.api.example.com/v1/generate")
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GEMINI_MODEL, "prompt": prompt, "max_output_tokens": 256, "temperature": 0.7}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("text") or str(data)
        return None
    except Exception:
        return None

def get_dummy_response(prompt: str) -> str:
    return random.choice(DUMMY_RESPONSES)

# ---------- API Endpoints ----------
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    user_id = req.user_id.strip()
    message = req.message.strip()
    if not user_id or not message:
        raise HTTPException(status_code=400, detail="user_id and message are required.")

    save_message(user_id=user_id, sender="user", message=message)

    if DUMMY_ALWAYS:
        reply = get_dummy_response(message)
        save_message(user_id=user_id, sender="bot", message=reply)
        return ChatResponse(reply=reply, source="dummy", timestamp=datetime.utcnow())

    preferred = (req.prefer or "").lower()
    providers = (
        ["hf", "ibm", "openai", "gemini"] if preferred == "hf" else
        ["ibm", "hf", "openai", "gemini"] if preferred == "ibm" else
        ["openai", "ibm", "hf", "gemini"] if preferred == "openai" else
        ["gemini", "ibm", "hf", "openai"] if preferred == "gemini" else
        ["ibm", "hf", "openai", "gemini"]
    )

    reply_text = None
    source = "dummy"
    for p in providers:
        reply_text = (
            call_ibm_watsonx(message) if p == "ibm" else
            call_huggingface(message) if p == "hf" else
            call_openai(message) if p == "openai" else
            call_gemini(message) if p == "gemini" else None
        )
        if reply_text:
            source = p
            break

    if not reply_text:
        reply_text = get_dummy_response(message)
        source = "dummy"

    save_message(user_id=user_id, sender="bot", message=reply_text)
    return ChatResponse(reply=reply_text, source=source, timestamp=datetime.utcnow())

@app.get("/history/{user_id}", tags=["chat"])
def history(user_id: str, limit: int = 200):
    return {"user_id": user_id, "history": get_history(user_id, limit)}

@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "watsonx_configured": bool(WATSONX_API_KEY and WATSONX_BASE_URL),
        "hf_configured": bool(HUGGINGFACE_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY)
    }

def touch(obj):
    if hasattr(obj, "updated_at"):
        setattr(obj, "updated_at", datetime.utcnow())

# Savings
@app.get("/savings/{user_id}", response_model=List[SavingsOut], tags=["savings"])
def list_savings(user_id: str):
    db = SessionLocal(); 
    try:
        rows = db.query(SavingsGoal).filter(SavingsGoal.user_id == user_id).order_by(SavingsGoal.created_at.asc()).all()
        return rows
    finally:
        db.close()

@app.post("/savings/{user_id}", response_model=SavingsOut, tags=["savings"])
def create_savings(user_id: str, body: SavingsCreate):
    db = SessionLocal()
    try:
        row = SavingsGoal(
            user_id=user_id,
            name=body.name.strip(),
            target_amount=body.target_amount,
            saved_amount=body.saved_amount,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.put("/savings/{user_id}/{goal_id}", response_model=SavingsOut, tags=["savings"])
def update_savings(user_id: str, goal_id: int = FPath(..., ge=1), body: SavingsCreate = None):
    db = SessionLocal()
    try:
        row = db.query(SavingsGoal).filter(SavingsGoal.user_id == user_id, SavingsGoal.id == goal_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Savings goal not found.")
        if body is not None:
            row.name = body.name.strip()
            row.target_amount = body.target_amount
            row.saved_amount = body.saved_amount
        touch(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.delete("/savings/{user_id}/{goal_id}", tags=["savings"])
def delete_savings(user_id: str, goal_id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        row = db.query(SavingsGoal).filter(SavingsGoal.user_id == user_id, SavingsGoal.id == goal_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Savings goal not found.")
        db.delete(row); db.commit()
        return {"status": "deleted", "id": goal_id}
    finally:
        db.close()

# Investments
@app.get("/investments/{user_id}", response_model=List[InvestmentOut], tags=["investments"])
def list_investments(user_id: str):
    db = SessionLocal()
    try:
        rows = db.query(Investment).filter(Investment.user_id == user_id).order_by(Investment.created_at.asc()).all()
        return rows
    finally:
        db.close()

@app.post("/investments/{user_id}", response_model=InvestmentOut, tags=["investments"])
def create_investment(user_id: str, body: InvestmentCreate):
    db = SessionLocal()
    try:
        row = Investment(
            user_id=user_id,
            type=(body.type or "Stock"),
            symbol=(body.symbol or None),
            amount_invested=body.amount_invested,
            current_value=body.current_value,
            notes=body.notes,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.put("/investments/{user_id}/{inv_id}", response_model=InvestmentOut, tags=["investments"])
def update_investment(user_id: str, inv_id: int = FPath(..., ge=1), body: InvestmentCreate = None):
    db = SessionLocal()
    try:
        row = db.query(Investment).filter(Investment.user_id == user_id, Investment.id == inv_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Investment not found.")
        if body is not None:
            row.type = (body.type or row.type)
            row.symbol = (body.symbol or None)
            row.amount_invested = body.amount_invested
            row.current_value = body.current_value
            row.notes = body.notes
        touch(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.delete("/investments/{user_id}/{inv_id}", tags=["investments"])
def delete_investment(user_id: str, inv_id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        row = db.query(Investment).filter(Investment.user_id == user_id, Investment.id == inv_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Investment not found.")
        db.delete(row); db.commit()
        return {"status": "deleted", "id": inv_id}
    finally:
        db.close()

# Taxes
@app.get("/taxes/{user_id}", response_model=List[TaxOut], tags=["taxes"])
def list_taxes(user_id: str):
    db = SessionLocal()
    try:
        rows = db.query(TaxRecord).filter(TaxRecord.user_id == user_id).order_by(TaxRecord.year.asc()).all()
        return rows
    finally:
        db.close()

@app.post("/taxes/{user_id}", response_model=TaxOut, tags=["taxes"])
def create_tax(user_id: str, body: TaxCreate):
    db = SessionLocal()
    try:
        existing = db.query(TaxRecord).filter(TaxRecord.user_id == user_id, TaxRecord.year == body.year).first()
        if existing:
            raise HTTPException(status_code=400, detail="Tax record for this year already exists.")
        row = TaxRecord(
            user_id=user_id,
            year=body.year,
            income=body.income,
            tax_paid=body.tax_paid,
            refund_expected=body.refund_expected,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.put("/taxes/{user_id}/{tax_id}", response_model=TaxOut, tags=["taxes"])
def update_tax(user_id: str, tax_id: int = FPath(..., ge=1), body: TaxCreate = None):
    db = SessionLocal()
    try:
        row = db.query(TaxRecord).filter(TaxRecord.user_id == user_id, TaxRecord.id == tax_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Tax record not found.")
        if body is not None:
            if body.year != row.year:
                dup = db.query(TaxRecord).filter(TaxRecord.user_id == user_id, TaxRecord.year == body.year).first()
                if dup:
                    raise HTTPException(status_code=400, detail="Another tax record already exists with that year.")
                row.year = body.year
            row.income = body.income
            row.tax_paid = body.tax_paid
            row.refund_expected = body.refund_expected
        touch(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.delete("/taxes/{user_id}/{tax_id}", tags=["taxes"])
def delete_tax(user_id: str, tax_id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        row = db.query(TaxRecord).filter(TaxRecord.user_id == user_id, TaxRecord.id == tax_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Tax record not found.")
        db.delete(row); db.commit()
        return {"status": "deleted", "id": tax_id}
    finally:
        db.close()

# Planning
@app.get("/planning/{user_id}", response_model=List[PlanOut], tags=["planning"])
def list_plans(user_id: str):
    db = SessionLocal()
    try:
        rows = db.query(Plan).filter(Plan.user_id == user_id).order_by(Plan.created_at.asc()).all()
        return rows
    finally:
        db.close()

@app.post("/planning/{user_id}", response_model=PlanOut, tags=["planning"])
def create_plan(user_id: str, body: PlanCreate):
    db = SessionLocal()
    try:
        row = Plan(
            user_id=user_id,
            goal=body.goal.strip(),
            target_year=body.target_year,
            estimated_amount=body.estimated_amount,
            notes=body.notes,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.put("/planning/{user_id}/{plan_id}", response_model=PlanOut, tags=["planning"])
def update_plan(user_id: str, plan_id: int = FPath(..., ge=1), body: PlanCreate = None):
    db = SessionLocal()
    try:
        row = db.query(Plan).filter(Plan.user_id == user_id, Plan.id == plan_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Plan not found.")
        if body is not None:
            row.goal = body.goal.strip()
            row.target_year = body.target_year
            row.estimated_amount = body.estimated_amount
            row.notes = body.notes
        touch(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.delete("/planning/{user_id}/{plan_id}", tags=["planning"])
def delete_plan(user_id: str, plan_id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        row = db.query(Plan).filter(Plan.user_id == user_id, Plan.id == plan_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Plan not found.")
        db.delete(row); db.commit()
        return {"status": "deleted", "id": plan_id}
    finally:
        db.close()

# Settings
@app.get("/settings/{user_id}", response_model=SettingsOut, tags=["settings"])
def get_settings(user_id: str):
    db = SessionLocal()
    try:
        row = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
        if not row:
            row = UserSettings(
                user_id=user_id,
                name="You",
                preferred_model="",
                theme="light",
                notifications=True,
                updated_at=datetime.utcnow()
            )
            db.add(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

@app.put("/settings/{user_id}", response_model=SettingsOut, tags=["settings"])
def update_settings(user_id: str, body: SettingsUpdate):
    db = SessionLocal()
    try:
        row = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
        if not row:
            row = UserSettings(user_id=user_id, updated_at=datetime.utcnow())
            db.add(row); db.flush()
        if body.name is not None: row.name = body.name
        if body.preferred_model is not None: row.preferred_model = body.preferred_model
        if body.theme is not None: row.theme = body.theme
        if body.notifications is not None: row.notifications = body.notifications
        touch(row); db.commit(); db.refresh(row)
        return row
    finally:
        db.close()

# Entrypoint (only needed if "python app.py")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
