"""
Database Schemas for PixFlow 2025

Each Pydantic model represents a collection in MongoDB.
Collection name = lowercase of class name.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime

class Admin(BaseModel):
    full_name: str
    username: str
    email: EmailStr
    password_hash: str
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None

class Event(BaseModel):
    slug: str
    title: str
    description: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    cover_url: Optional[str] = None
    expires_at: datetime
    status: Literal["Active", "Expired"] = "Active"

class Photo(BaseModel):
    event_id: str
    original_url: str
    public_url: str
    downloads: int = 0
    created_at: Optional[datetime] = None
    expires_at: datetime

class Settings(BaseModel):
    default_expiry_days: int = 15
    background_preset: Optional[Literal["Night Party", "Beach Party", "Elegant Gala"]] = None
    background_url: Optional[str] = None
    blur: int = 10
    watermark_url: Optional[str] = None
    wm_opacity: int = 35
    wm_position: Literal["top-left","top-right","bottom-left","bottom-right"] = "bottom-right"
    wm_enabled: bool = True
    gift_mode: bool = False

class Monetization(BaseModel):
    payments_enabled: bool = False
    price_usd: Optional[float] = 0.0
    revolut_id: Optional[str] = None
    ads_enabled: bool = False
    ads_asset_url: Optional[str] = None
    ads_placement: Optional[Literal["Homepage","Event","Sidebar"]] = None

class Message(BaseModel):
    name: str
    email: EmailStr
    body: str
    status: Literal["Unread", "Resolved"] = "Unread"

class Subscriber(BaseModel):
    name: Optional[str] = None
    email: EmailStr
