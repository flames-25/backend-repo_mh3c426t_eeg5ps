import os
import io
import uuid
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
import bcrypt
from PIL import Image, ImageEnhance

from database import db, create_document, get_documents
from schemas import Admin as AdminSchema, Event as EventSchema, Photo as PhotoSchema, Settings as SettingsSchema, Monetization as MonetizationSchema, Message as MessageSchema, Subscriber as SubscriberSchema

# App setup
app = FastAPI(title="PixFlow 2025 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Files and static
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PUBLIC_DIR = os.path.join(UPLOAD_DIR, "public")
PRIVATE_DIR = os.path.join(UPLOAD_DIR, "private")
ASSETS_DIR = os.path.join(UPLOAD_DIR, "assets")
for d in [UPLOAD_DIR, PUBLIC_DIR, PRIVATE_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Session / security
SECRET_KEY = os.getenv("SESSION_SECRET", secrets.token_urlsafe(32))
signer = TimestampSigner(SECRET_KEY)
SESSION_COOKIE = "pixflow_session"
DEFAULT_IDLE_MINUTES = 15
MAX_FILE_MB = 4

# Helpers

def now_utc():
    return datetime.now(timezone.utc)

def sanitize_filename(name: str) -> str:
    keep = [c for c in name if c.isalnum() or c in (' ', '.', '_', '-')] 
    safe = ''.join(keep).strip().replace(' ', '_')
    return safe or uuid.uuid4().hex

def slugify(text: str) -> str:
    base = ''.join(c.lower() if c.isalnum() else '-' for c in text).strip('-')
    return '-'.join([p for p in base.split('-') if p])[:80]

# Database helpers

def collection(name: str):
    return db[name]

# Ensure default settings docs
if collection('settings').count_documents({}) == 0:
    create_document('settings', SettingsSchema().model_dump())
if collection('monetization').count_documents({}) == 0:
    create_document('monetization', MonetizationSchema().model_dump())

# Auth helpers
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    full_name: str
    username: str
    email: EmailStr
    password: str
    confirm: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class NewPasswordRequest(BaseModel):
    token: str
    password: str

RESET_TOKENS = collection('reset_tokens')


def set_session(res: Response, username: str):
    token = signer.sign(username.encode()).decode()
    res.set_cookie(SESSION_COOKIE, token, httponly=True, secure=True, samesite="lax", max_age=DEFAULT_IDLE_MINUTES*60)

def clear_session(res: Response):
    res.delete_cookie(SESSION_COOKIE)

def get_current_admin(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        username = signer.unsign(token, max_age=DEFAULT_IDLE_MINUTES*60).decode()
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session expired")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session")
    admin = collection('admin').find_one({"username": username})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid session")
    return admin

# Public endpoints
@app.get("/")
def read_root():
    return {"name": "PixFlow 2025", "message": "Backend running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected & Working"
            response["database_url"] = "✅ Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:80]}"
    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

@app.get("/public/settings")
def public_settings():
    s = collection('settings').find_one({}, sort=[('_id', 1)])
    m = collection('monetization').find_one({}, sort=[('_id', 1)])
    if s and '_id' in s: s.pop('_id')
    if m and '_id' in m: m.pop('_id')
    return {"settings": s, "monetization": m}

@app.get("/public/events")
def list_live_events():
    now = now_utc()
    events = list(collection('event').find({"expires_at": {"$gt": now}}).sort("date", 1))
    result = []
    for e in events:
        e['_id'] = str(e['_id'])
        expires_at = e.get('expires_at')
        days_left = max(0, (expires_at - now).days)
        result.append({
            "slug": e.get('slug'),
            "title": e.get('title'),
            "date": e.get('date'),
            "location": e.get('location'),
            "cover_url": e.get('cover_url'),
            "days_left": days_left
        })
    ads = collection('monetization').find_one({}) or {}
    return {"events": result, "ads": {"enabled": bool(ads.get('ads_enabled')), "asset_url": ads.get('ads_asset_url'), "placement": ads.get('ads_placement')}}

@app.get("/public/event/{slug}")
def public_event(slug: str):
    now = now_utc()
    e = collection('event').find_one({"slug": slug})
    if not e or e.get('expires_at') <= now:
        raise HTTPException(status_code=404, detail="Event not found")
    e['_id'] = str(e['_id'])
    photos = list(collection('photo').find({"event_id": e['_id'], "expires_at": {"$gt": now}}))
    formatted = []
    for p in photos:
        formatted.append({
            "id": str(p['_id']),
            "public_url": p.get('public_url'),
            "downloads": p.get('downloads', 0)
        })
    # Countdown seconds left
    seconds_left = max(0, int((e['expires_at'] - now).total_seconds()))
    return {"event": {"slug": e['slug'], "title": e['title'], "date": e.get('date'), "location": e.get('location'), "expires_at": e['expires_at'].isoformat(), "seconds_left": seconds_left}, "photos": formatted}

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str

@app.post("/public/contact")
def public_contact(payload: ContactRequest):
    data = MessageSchema(name=payload.name, email=payload.email, body=payload.message)
    create_document('message', data)
    return {"ok": True}

class SubscribeRequest(BaseModel):
    name: Optional[str] = None
    email: EmailStr

@app.post("/public/subscribe")
def public_subscribe(payload: SubscribeRequest):
    # prevent duplicates
    if collection('subscriber').find_one({"email": payload.email}):
        return {"ok": True}
    create_document('subscriber', SubscriberSchema(name=payload.name, email=payload.email))
    return {"ok": True}

@app.get("/public/download/{photo_id}")
def public_download(photo_id: str):
    p = collection('photo').find_one({"_id": db.client.get_default_database().codec_options.document_class({'$oid': photo_id}).get('_id')})
    # The above bson hack is messy, instead use ObjectId import
    from bson import ObjectId
    try:
        oid = ObjectId(photo_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Not found")
    p = collection('photo').find_one({"_id": oid})
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    # increment downloads
    collection('photo').update_one({"_id": oid}, {"$inc": {"downloads": 1}})
    url = p.get('public_url')
    if not url:
        raise HTTPException(status_code=404, detail="File missing")
    return {"url": url}

# Admin auth endpoints
@app.get("/admin/setup-required")
def setup_required():
    exists = collection('admin').count_documents({}) > 0
    return {"setup": not exists}

@app.post("/admin/register")
def admin_register(payload: RegisterRequest, response: Response):
    if collection('admin').count_documents({}) > 0:
        raise HTTPException(status_code=400, detail="Admin already exists")
    if payload.password != payload.confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    salt = bcrypt.gensalt()
    pw_hash = bcrypt.hashpw(payload.password.encode(), salt).decode()
    admin_doc = AdminSchema(full_name=payload.full_name, username=payload.username, email=payload.email, password_hash=pw_hash).model_dump()
    create_document('admin', admin_doc)
    set_session(response, payload.username)
    return {"ok": True}

@app.post("/admin/login")
def admin_login(payload: LoginRequest, response: Response, request: Request):
    admin = collection('admin').find_one({"username": payload.username})
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not bcrypt.checkpw(payload.password.encode(), admin['password_hash'].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # update last login
    collection('admin').update_one({"_id": admin['_id']}, {"$set": {"last_login_at": now_utc(), "last_login_ip": request.client.host if request.client else None}})
    set_session(response, payload.username)
    return {"ok": True}

@app.post("/admin/logout")
def admin_logout(response: Response):
    clear_session(response)
    return {"ok": True}

@app.get("/admin/me")
def admin_me(admin=Depends(get_current_admin)):
    return {"username": admin.get('username'), "email": admin.get('email'), "full_name": admin.get('full_name')}

# Settings / Appearance / Monetization
class UpdateSettings(BaseModel):
    default_expiry_days: Optional[int] = None
    background_preset: Optional[str] = None
    blur: Optional[int] = None
    gift_mode: Optional[bool] = None
    wm_opacity: Optional[int] = None
    wm_position: Optional[str] = None
    wm_enabled: Optional[bool] = None

@app.get("/admin/settings")
def get_settings(admin=Depends(get_current_admin)):
    s = collection('settings').find_one({})
    if s and '_id' in s: s.pop('_id')
    return s

@app.post("/admin/settings")
def update_settings(payload: UpdateSettings, admin=Depends(get_current_admin)):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    collection('settings').update_one({}, {"$set": updates}, upsert=True)
    return {"ok": True}

@app.post("/admin/appearance/background")
def upload_background(file: UploadFile = File(...), blur: int = Form(10), preset: Optional[str] = Form(None), admin=Depends(get_current_admin)):
    if file.size and file.size > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    fname = sanitize_filename(f"bg_{uuid.uuid4().hex}{ext}")
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, 'wb') as f:
        f.write(file.file.read())
    url = f"/uploads/assets/{fname}"
    collection('settings').update_one({}, {"$set": {"background_url": url, "blur": int(max(0, min(30, blur))), "background_preset": preset}}, upsert=True)
    return {"ok": True, "background_url": url}

@app.post("/admin/appearance/watermark")
def upload_watermark(file: UploadFile = File(...), admin=Depends(get_current_admin)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png"]:
        raise HTTPException(status_code=400, detail="Watermark must be PNG with transparency")
    fname = sanitize_filename(f"wm_{uuid.uuid4().hex}{ext}")
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, 'wb') as f:
        f.write(file.file.read())
    url = f"/uploads/assets/{fname}"
    collection('settings').update_one({}, {"$set": {"watermark_url": url}}, upsert=True)
    return {"ok": True, "watermark_url": url}

class UpdateMonetization(BaseModel):
    payments_enabled: Optional[bool] = None
    price_usd: Optional[float] = None
    revolut_id: Optional[str] = None
    ads_enabled: Optional[bool] = None
    ads_placement: Optional[str] = None

@app.get("/admin/monetization")
def get_monetization(admin=Depends(get_current_admin)):
    m = collection('monetization').find_one({})
    if m and '_id' in m: m.pop('_id')
    return m

@app.post("/admin/monetization")
def update_monetization(payload: UpdateMonetization, admin=Depends(get_current_admin)):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    collection('monetization').update_one({}, {"$set": updates}, upsert=True)
    return {"ok": True}

@app.post("/admin/monetization/ads-asset")
def upload_ads_asset(file: UploadFile = File(...), admin=Depends(get_current_admin)):
    if file.size and file.size > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    fname = sanitize_filename(f"ad_{uuid.uuid4().hex}{ext}")
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, 'wb') as f:
        f.write(file.file.read())
    url = f"/uploads/assets/{fname}"
    collection('monetization').update_one({}, {"$set": {"ads_asset_url": url}}, upsert=True)
    return {"ok": True, "asset_url": url}

# Events & Photos
class CreateEventRequest(BaseModel):
    title: str
    description: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    expiry_days: Optional[int] = None

@app.get("/admin/events")
def admin_list_events(admin=Depends(get_current_admin)):
    now = now_utc()
    items = []
    for e in collection('event').find({}).sort("date", -1):
        eid = str(e['_id'])
        status = "Active" if e.get('expires_at') and e['expires_at'] > now else "Expired"
        photos_count = collection('photo').count_documents({"event_id": eid})
        downloads = sum([p.get('downloads', 0) for p in collection('photo').find({"event_id": eid})])
        days_left = max(0, (e['expires_at'] - now).days) if e.get('expires_at') else 0
        items.append({
            "id": eid,
            "slug": e.get('slug'),
            "title": e.get('title'),
            "date": e.get('date'),
            "status": status,
            "days_left": days_left,
            "downloads": downloads,
            "cover_url": e.get('cover_url'),
            "photos": photos_count
        })
    return {"events": items}

@app.post("/admin/events")
def admin_create_event(payload: CreateEventRequest, admin=Depends(get_current_admin)):
    s = collection('settings').find_one({}) or {}
    days = payload.expiry_days if payload.expiry_days is not None else s.get('default_expiry_days', 15)
    exp = now_utc() + timedelta(days=max(1, int(days)))
    slug = slugify(payload.title)
    # ensure unique
    i = 1
    base = slug
    while collection('event').find_one({"slug": slug}):
        slug = f"{base}-{i}"
        i += 1
    data = EventSchema(slug=slug, title=payload.title, description=payload.description, date=payload.date, location=payload.location, cover_url=None, expires_at=exp, status="Active").model_dump()
    eid = create_document('event', data)
    return {"ok": True, "id": eid, "slug": slug}

@app.post("/admin/events/{event_id}/cover")
def upload_event_cover(event_id: str, file: UploadFile = File(...), admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Event not found")
    e = collection('event').find_one({"_id": oid})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    fname = sanitize_filename(f"cover_{event_id}_{uuid.uuid4().hex}{ext}")
    path = os.path.join(PUBLIC_DIR, fname)
    with open(path, 'wb') as f:
        f.write(file.file.read())
    url = f"/uploads/public/{fname}"
    collection('event').update_one({"_id": oid}, {"$set": {"cover_url": url}})
    return {"ok": True, "cover_url": url}

# Watermark utilities

def apply_watermark(image: Image.Image, wm_path: Optional[str], opacity: int = 35, position: str = 'bottom-right') -> Image.Image:
    img = image.convert('RGBA')
    if not wm_path or not os.path.exists(wm_path):
        return img
    wm = Image.open(wm_path).convert('RGBA')
    # scale watermark to 20% width of image
    target_w = max(50, int(img.width * 0.2))
    ratio = target_w / wm.width
    wm = wm.resize((target_w, int(wm.height * ratio)))
    # apply opacity
    alpha = wm.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(max(0.0, min(1.0, opacity/100)))
    wm.putalpha(alpha)
    # position
    margin = 16
    if position == 'top-left':
        pos = (margin, margin)
    elif position == 'top-right':
        pos = (img.width - wm.width - margin, margin)
    elif position == 'bottom-left':
        pos = (margin, img.height - wm.height - margin)
    else:
        pos = (img.width - wm.width - margin, img.height - wm.height - margin)
    img.alpha_composite(wm, dest=pos)
    return img

@app.post("/admin/events/{event_id}/photos")
def upload_event_photos(event_id: str, files: List[UploadFile] = File(...), admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Event not found")
    e = collection('event').find_one({"_id": oid})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    s = collection('settings').find_one({}) or {}
    wm_url = s.get('watermark_url')
    wm_path = os.path.join(BASE_DIR, wm_url.lstrip('/')) if wm_url else None
    wm_enabled = bool(s.get('wm_enabled', True))
    wm_opacity = 15 if s.get('gift_mode') else int(s.get('wm_opacity', 35))
    wm_position = s.get('wm_position', 'bottom-right')

    created = []
    for f in files:
        content = f.file.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        base_name = sanitize_filename(os.path.splitext(f.filename)[0])
        priv_name = f"{base_name}_{uuid.uuid4().hex}{ext}"
        priv_path = os.path.join(PRIVATE_DIR, priv_name)
        with open(priv_path, 'wb') as out:
            out.write(content)
        # Watermark
        try:
            image = Image.open(io.BytesIO(content))
            if wm_enabled:
                image = apply_watermark(image, wm_path, wm_opacity, wm_position)
            # save as webp for public
            pub_name = f"{base_name}_{uuid.uuid4().hex}.webp"
            pub_path = os.path.join(PUBLIC_DIR, pub_name)
            image.convert('RGB').save(pub_path, format='WEBP', quality=90)
        except Exception:
            # fallback copy
            pub_name = f"{base_name}_{uuid.uuid4().hex}{ext}"
            pub_path = os.path.join(PUBLIC_DIR, pub_name)
            with open(pub_path, 'wb') as out:
                out.write(content)
        pub_url = f"/uploads/public/{pub_name}"
        priv_url = f"/uploads/private/{priv_name}"
        photo = PhotoSchema(event_id=str(e['_id']), original_url=priv_url, public_url=pub_url, downloads=0, created_at=now_utc(), expires_at=e['expires_at']).model_dump()
        pid = create_document('photo', photo)
        created.append({"id": pid, "public_url": pub_url})
    return {"ok": True, "photos": created}

@app.get("/admin/events/{event_id}/photos")
def list_event_photos(event_id: str, admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Event not found")
    e = collection('event').find_one({"_id": oid})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    items = []
    for p in collection('photo').find({"event_id": str(oid)}).sort("created_at", -1):
        p['_id'] = str(p['_id'])
        items.append(p)
    return {"photos": items}

class PhotoAction(BaseModel):
    action: str
    value: Optional[float] = None

@app.post("/admin/photos/{photo_id}/action")
def photo_action(photo_id: str, payload: PhotoAction, admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(photo_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Photo not found")
    p = collection('photo').find_one({"_id": oid})
    if not p:
        raise HTTPException(status_code=404, detail="Photo not found")
    # Load original
    path = os.path.join(BASE_DIR, p['original_url'].lstrip('/'))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Original missing")
    image = Image.open(path)
    action = payload.action
    if action == 'rotate':
        image = image.rotate(payload.value or 90, expand=True)
    elif action == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(max(0.1, min(3.0, payload.value or 1.0)))
    elif action == 'crop':
        # value is expected like x,y,w,h in percentages stored as 0-100 sequentially in string
        # For simplicity, ignore if not provided correctly
        pass
    # re-generate public watermarked version
    s = collection('settings').find_one({}) or {}
    wm_url = s.get('watermark_url')
    wm_path = os.path.join(BASE_DIR, wm_url.lstrip('/')) if wm_url else None
    wm_enabled = bool(s.get('wm_enabled', True))
    wm_opacity = 15 if s.get('gift_mode') else int(s.get('wm_opacity', 35))
    wm_position = s.get('wm_position', 'bottom-right')
    if wm_enabled:
        image = apply_watermark(image, wm_path, wm_opacity, wm_position)
    pub_name = f"edit_{uuid.uuid4().hex}.webp"
    pub_path = os.path.join(PUBLIC_DIR, pub_name)
    image.convert('RGB').save(pub_path, format='WEBP', quality=90)
    pub_url = f"/uploads/public/{pub_name}"
    collection('photo').update_one({"_id": oid}, {"$set": {"public_url": pub_url}})
    return {"ok": True, "public_url": pub_url}

@app.delete("/admin/photos/{photo_id}")
def delete_photo(photo_id: str, admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(photo_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Photo not found")
    p = collection('photo').find_one({"_id": oid})
    if not p:
        return {"ok": True}
    collection('photo').delete_one({"_id": oid})
    return {"ok": True}

class ExtendEvent(BaseModel):
    days: int

@app.post("/admin/events/{event_id}/extend")
def extend_event(event_id: str, payload: ExtendEvent, admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Event not found")
    e = collection('event').find_one({"_id": oid})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    new_exp = (e.get('expires_at') or now_utc()) + timedelta(days=max(1, payload.days))
    collection('event').update_one({"_id": oid}, {"$set": {"expires_at": new_exp, "status": "Active"}})
    collection('photo').update_many({"event_id": str(oid)}, {"$set": {"expires_at": new_exp}})
    return {"ok": True, "expires_at": new_exp.isoformat()}

@app.delete("/admin/events/{event_id}")
def delete_event(event_id: str, admin=Depends(get_current_admin)):
    from bson import ObjectId
    try:
        oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Event not found")
    collection('photo').delete_many({"event_id": str(oid)})
    collection('event').delete_one({"_id": oid})
    return {"ok": True}

# Expiry maintenance (simple on-read enforcement will hide expired)
@app.post("/admin/maintenance/expire")
def expire_maintenance(admin=Depends(get_current_admin)):
    now = now_utc()
    # Hide/flag expired events
    collection('event').update_many({"expires_at": {"$lte": now}}, {"$set": {"status": "Expired"}})
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
