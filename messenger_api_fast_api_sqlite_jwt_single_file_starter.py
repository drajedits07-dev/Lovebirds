"""
Messenger API — FastAPI + SQLite + JWT (single-file starter)

Features:
- User auth: register, login (JWT bearer)
- Conversations: 1:1 and group
- Messages: text (incl. emoji), GIF (by URL), media/photo/video/document (file upload), location
- Message listing with pagination
- File uploads to ./uploads with basic MIME/type checks

Run locally:
  pip install fastapi uvicorn sqlalchemy passlib[bcrypt] python-multipart python-jose[cryptography]
  export SECRET_KEY="change-me"  # or set in .env
  uvicorn main:app --reload

Notes:
- This is a minimal starter. Add rate limiting, validation hardening, and production file storage (e.g., S3) before going live.
"""

from __future__ import annotations
import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Literal

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, field_validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# --------------------------
# Config
# --------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./messenger.db")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = 15
ALLOWED_MIME = {
    # images
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    # videos (basic — increase as needed)
    "video/mp4",
    "video/quicktime",
    # docs
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------
# DB setup
# --------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    is_group = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    participants = relationship("Participant", back_populates="conversation", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Participant(Base):
    __tablename__ = "participants"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    joined_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), index=True)

    type = Column(String, nullable=False)  # text, gif, media, location
    text = Column(Text, nullable=True)     # emojis are just text
    gif_url = Column(String, nullable=True)
    media_url = Column(String, nullable=True)  # for uploaded files
    media_mime = Column(String, nullable=True)

    # location
    latitude = Column(String, nullable=True)
    longitude = Column(String, nullable=True)
    location_label = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("User")


Base.metadata.create_all(bind=engine)

# --------------------------
# Auth utilities
# --------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.get(User, int(user_id))
    if user is None:
        raise credentials_exception
    return user

# --------------------------
# Schemas
# --------------------------

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: str

    class Config:
        from_attributes = True


MessageType = Literal["text", "gif", "media", "location"]


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    participant_ids: List[int]
    is_group: bool = False

    @field_validator("participant_ids")
    @classmethod
    def require_at_least_two(cls, v: List[int]) -> List[int]:
        if len(set(v)) < 2:
            raise ValueError("A conversation needs at least two distinct participants")
        return v


class ConversationOut(BaseModel):
    id: int
    title: Optional[str]
    is_group: bool
    participants: List[int]

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: int
    conversation_id: int
    sender_id: int
    type: MessageType
    text: Optional[str] = None
    gif_url: Optional[str] = None
    media_url: Optional[str] = None
    media_mime: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    location_label: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --------------------------
# FastAPI app
# --------------------------

app = FastAPI(title="Messenger API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (simple dev use)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# -------- Auth endpoints --------
@app.post("/auth/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_in.email, name=user_in.name, password_hash=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=TokenOut)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return TokenOut(access_token=token)


# -------- Conversation endpoints --------
@app.post("/conversations", response_model=ConversationOut)
def create_conversation(payload: ConversationCreate, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Ensure current user is in the participants list
    participant_ids = list(dict.fromkeys(payload.participant_ids))  # dedupe
    if current.id not in participant_ids:
        participant_ids.append(current.id)

    conv = Conversation(title=payload.title, is_group=payload.is_group)
    db.add(conv)
    db.flush()  # get ID
    for uid in participant_ids:
        db.add(Participant(conversation_id=conv.id, user_id=uid))
    db.commit()
    db.refresh(conv)
    return ConversationOut(
        id=conv.id,
        title=conv.title,
        is_group=conv.is_group,
        participants=[p.user_id for p in conv.participants],
    )


@app.get("/conversations/{conv_id}", response_model=ConversationOut)
def get_conversation(conv_id: int, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    # TODO: check membership
    return ConversationOut(id=conv.id, title=conv.title, is_group=conv.is_group, participants=[p.user_id for p in conv.participants])


# -------- Message endpoints --------
class SendTextBody(BaseModel):
    text: str


@app.post("/conversations/{conv_id}/messages/text", response_model=MessageOut)
def send_text(conv_id: int, body: SendTextBody, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msg = Message(conversation_id=conv_id, sender_id=current.id, type="text", text=body.text)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


class SendGifBody(BaseModel):
    gif_url: str


@app.post("/conversations/{conv_id}/messages/gif", response_model=MessageOut)
def send_gif(conv_id: int, body: SendGifBody, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msg = Message(conversation_id=conv_id, sender_id=current.id, type="gif", gif_url=body.gif_url)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


@app.post("/conversations/{conv_id}/messages/media", response_model=MessageOut)
def send_media(
    conv_id: int,
    file: UploadFile = File(...),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate size (streaming would be better; here we load into temp)
    contents = file.file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large (>{MAX_UPLOAD_MB}MB)")
    mime = file.content_type or "application/octet-stream"
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported file type: {mime}")

    ext = os.path.splitext(file.filename)[1] or ""
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, stored_name)
    with open(dest_path, "wb") as f:
        f.write(contents)

    url = f"/uploads/{stored_name}"

    conv = db.get(Conversation, conv_id)
    if not conv:
        # cleanup orphaned file
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise HTTPException(404, "Conversation not found")

    msg = Message(
        conversation_id=conv_id,
        sender_id=current.id,
        type="media",
        media_url=url,
        media_mime=mime,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


@app.post("/conversations/{conv_id}/messages/location", response_model=MessageOut)
def send_location(
    conv_id: int,
    latitude: str = Form(...),
    longitude: str = Form(...),
    location_label: Optional[str] = Form(None),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msg = Message(
        conversation_id=conv_id,
        sender_id=current.id,
        type="location",
        latitude=latitude,
        longitude=longitude,
        location_label=location_label,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


@app.get("/conversations/{conv_id}/messages", response_model=List[MessageOut])
def list_messages(
    conv_id: int,
    limit: int = 30,
    before_id: Optional[int] = None,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.id.desc())
    if before_id:
        q = q.filter(Message.id < before_id)
    msgs = q.limit(min(limit, 100)).all()
    return list(reversed(msgs))


# Health
@app.get("/")
def root():
    return {"ok": True, "name": "Messenger API", "version": "0.1.0"}
