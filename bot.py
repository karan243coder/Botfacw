import os
import sys
import logging
import time
import asyncio
import re
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Dict, Any

from PIL import Image
from dotenv import load_dotenv
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from gradio_client import Client, handle_file

# ----------------- Environment & Setup -----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if HF_TOKEN and len(HF_TOKEN.strip()) > 5:
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN.strip()

TEMP_DIR = Path("/tmp/face_bot_cache")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Logging -----------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("UltraRealFaceBot")

# ----------------- State & Caching -----------------
user_sessions: Dict[int, Dict[str, Any]] = {}
CLIENT_CACHE: Dict[str, Client] = {}

AVAILABLE_RATIOS = [
    ("9:16", "📱 9:16 (Story / Reel / Full Portrait)"),
    ("16:9", "🖥️ 16:9 (Landscape / Wallpaper)"),
    ("1:1", "⏹️ 1:1 (Square / Instagram Post)"),
    ("3:4", "📸 3:4 (Standard Portrait)"),
    ("4:3", "🖼️ 4:3 (Standard Photo)"),
]

FLUX_DIMENSIONS = {
    "9:16": (896, 1152),
    "16:9": (1152, 896),
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
}

# ----------------- Koyeb Port 8000 Health Check Server -----------------
class KoyebHealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK - UltraRealFaceBot is healthy")

    def log_message(self, format, *args):
        return

def run_health_server():
    try:
        server_address = ("0.0.0.0", 8000)
        httpd = HTTPServer(server_address, KoyebHealthCheckHandler)
        logger.info("Koyeb TCP/HTTP Health Check Server active on 0.0.0.0:8000")
        httpd.serve_forever()
    except Exception as e:
        logger.warning(f"Health server error: {e}")

def start_koyeb_health_server():
    t = threading.Thread(target=run_health_server, daemon=True)
    t.start()

# ----------------- Cleanup & Image Tools -----------------
def cleanup_old_files():
    """Auto cleanup temporary images older than 20 minutes to stay within Koyeb RAM limits."""
    now = time.time()
    try:
        for p in TEMP_DIR.glob("*"):
            if p.is_file() and (now - p.stat().st_mtime > 1200):
                p.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"File cleanup error: {e}")

def create_high_res_face_crop(image_path: str, user_id: int) -> str:
    """Isolate face region with high pixel density for facial micro-features."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            if h > w:
                crop_box = (0, 0, w, int(h * 0.70))
            else:
                crop_box = (int(w * 0.10), 0, int(w * 0.90), h)

            cropped = img.crop(crop_box)
            crop_file = TEMP_DIR / f"crop_{user_id}_{int(time.time())}.jpg"
            cropped.save(crop_file, format="JPEG", quality=98)
            logger.info(f"High-res face crop created: {crop_file} ({cropped.size})")
            return str(crop_file)
    except Exception as e:
        logger.warning(f"Face crop fallback: {e}")
        return image_path

def apply_aspect_ratio(image_path: str, ratio_str: str) -> str:
    """Exact aspect ratio cropping engine using PIL."""
    ratio_map = {
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "1:1": 1.0,
        "3:4": 3 / 4,
        "4:3": 4 / 3,
    }
    target_ratio = ratio_map.get(ratio_str, 9 / 16)

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            current_ratio = w / h

            if abs(current_ratio - target_ratio) < 0.02:
                return image_path

            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                offset = (w - new_w) // 2
                cropped = img.crop((offset, 0, offset + new_w, h))
            else:
                new_h = int(w / target_ratio)
                offset = (h - new_h) // 2
                cropped = img.crop((0, offset, w, offset + new_h))

            output_ratio_path = str(Path(image_path).with_name(f"ratio_{Path(image_path).name}.jpg"))
            cropped.save(output_ratio_path, format="JPEG", quality=96)
            return output_ratio_path
    except Exception as e:
        logger.warning(f"Aspect ratio crop warning: {e}")
        return image_path

# ----------------- Hugging Face Space Client Loader -----------------
def get_hf_client(space_id: str, max_retries: int = 3) -> Client:
    """Connect to Hugging Face space using official Space repo ID."""
    if space_id in CLIENT_CACHE:
        return CLIENT_CACHE[space_id]

    token = None
    if HF_TOKEN and len(HF_TOKEN.strip()) > 5:
        token = HF_TOKEN.strip()

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to Hugging Face Space: '{space_id}' (Attempt {attempt}/{max_retries})...")
            if token:
                client = Client(space_id, token=token)
            else:
                client = Client(space_id)
            CLIENT_CACHE[space_id] = client
            logger.info(f"Successfully connected and cached: '{space_id}'")
            return client
        except Exception as e:
            logger.warning(f"Attempt {attempt} for '{space_id}' encountered: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise

# ----------------- High-Quality Prompt & Natural Language Parser -----------------
def translate_hinglish_to_dslr_prompt(user_prompt: str) -> str:
    """
    Parses conversational Hinglish / Hindi inputs and converts them into
    rich, high-definition photographic diffusion prompts.
    """
    raw = user_prompt.strip()
    p = raw.lower()

    # 1. Framing Analysis
    framing = "medium waist-up shot, upper body framing, shot on 50mm f1.2 lens"
    if any(k in p for k in ["full", "poori", "pura", "sir se pair", "khadi", "head to toe", "legs", "feet", "heels", "standing", "pair tak"]):
        framing = "full body standing shot, head to toe portrait, wide shot, shot on 35mm f1.4 prime lens"
    elif any(k in p for k in ["close up", "face", "chehra", "portrait", "sirf face", "eyes", "aankh"]):
        framing = "close-up beauty portrait shot, sharp focus on facial micro-features and eyes, shot on 85mm f1.2 lens"
    elif any(k in p for k in ["kamar", "waist", "half", "aadha", "upper body"]):
        framing = "medium waist-up shot, upper body framing, shot on 50mm f1.2 lens"

    # 2. Extract Creative Theme / Attire
    elements = []
    
    # Outfit / Costume / Fashion Themes
    if any(k in p for k in ["lingerie", "bra", "panty", "nighty", "nightwear"]):
        elements.append("wearing elegant luxury lace lingerie, sensual fashion portrait")
    elif any(k in p for k in ["bikini", "swimsuit", "swimwear"]):
        elements.append("wearing stylish luxury bikini, beachwear fashion portrait")
    elif any(k in p for k in ["bunny", "cosplay"]):
        elements.append("wearing sleek black velvet bunny cosplay corset dress with bunny ears headband")
    elif any(k in p for k in ["saree", "sari"]):
        elements.append("wearing glamorous royal silk saree with intricate golden zari work and traditional jewelry")
    elif any(k in p for k in ["lehenga", "ghagra"]):
        elements.append("wearing royal designer bridal lehenga with heavy embroidery and jewelry")
    elif any(k in p for k in ["gown", "dress"]):
        elements.append("wearing haute couture luxury evening gown")
    elif any(k in p for k in ["suit", "coat", "formal"]):
        elements.append("wearing tailored luxury designer suit")
    else:
        # Generic clean extraction
        clean_text = re.sub(r'\b(mera|meri|mujhe|bana|banao|kar|do|chahiye|dekh|lo|mat|karna|sir|photo|please)\b', '', p)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text) > 3:
            elements.append(clean_text)
        else:
            elements.append("glamorous fashion portrait, stylish outfit")

    # Environment
    if any(k in p for k in ["bed", "bedroom", "kamra", "room"]):
        elements.append("in luxury master bedroom with silk sheets, warm ambient lighting")
    elif any(k in p for k in ["beach", "samundar", "sea", "ocean"]):
        elements.append("on exotic tropical beach during golden hour sunset, soft ocean breeze")
    elif any(k in p for k in ["rooftop", "chhat", "terrace", "balcony"]):
        elements.append("on luxury penthouse rooftop balcony overlooking city skyline at dusk")
    elif any(k in p for k in ["studio", "lighting"]):
        elements.append("in high-end photography studio with professional softbox lighting")
    else:
        elements.append("luxurious aesthetic setting with soft natural illumination")

    # Pose & Expression
    if any(k in p for k in ["samne", "camera", "looking", "dekh"]):
        elements.append("looking directly into camera lens with confident captivating expression")
    elif any(k in p for k in ["leti", "soi", "lying"]):
        elements.append("reclining gracefully on plush silk bed")
    elif any(k in p for k in ["baithi", "sitting"]):
        elements.append("sitting elegantly with poised posture")
    else:
        elements.append("poised graceful posture, looking at camera")

    subject_description = ", ".join(elements)

    final_prompt = (
        f"hyperrealistic masterpiece DSLR photo of gorgeous Indian woman, "
        f"{framing}, {subject_description}, "
        f"authentic natural skin texture, visible fine skin pores, subsurface scattering, accurate skin tones, "
        f"realistic body anatomy, soft natural shadows, high dynamic range, 8k resolution, raw color photograph, "
        f"shot on Sony A1 with Hasselblad color profile, 85mm f1.2 optical lens, award winning portrait photography"
    )
    return final_prompt

def build_dslr_negative_prompt() -> str:
    return (
        "cartoon, anime, 3d render, cgi, illustration, drawing, painting, smooth plastic skin, "
        "wax doll, airbrushed, fake, porcelain skin, deformed face, bad eyes, disfigured anatomy, "
        "extra limbs, extra fingers, blurry, low quality, artifacts, watermark, logo, text, lowres"
    )

# ----------------- Base Scene Generator -----------------
def generate_flux_base_image(prompt: str, ratio_str: str = "9:16") -> str:
    """Generates high-resolution base scene using FLUX.1."""
    w, h = FLUX_DIMENSIONS.get(ratio_str, (896, 1152))
    encoded = urllib.parse.quote(prompt)
    seed = int(time.time() * 1000) % 1000000
    url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&model=flux&nologo=true&seed={seed}"
    
    with httpx.Client(timeout=50.0) as client:
        r = client.get(url)
        r.raise_for_status()
        out_file = TEMP_DIR / f"base_flux_{int(time.time())}.jpg"
        with open(out_file, "wb") as f:
            f.write(r.content)
        return str(out_file)

# ----------------- High-Fidelity Face Enhancer Swap -----------------
def execute_enhanced_face_swap(source_face_path: str, target_scene_path: str) -> str:
    """Applies high-resolution face identity transfer with GFPGAN face enhancer."""
    try:
        client = get_hf_client("VoidVision/Face_Swap_Uncensored", max_retries=2)
        res = client.predict(
            source_file=handle_file(source_face_path),
            target_file=handle_file(target_scene_path),
            doFaceEnhancer=True,
            api_name="/predict"
        )
        if isinstance(res, dict) and "path" in res:
            return res["path"]
        return str(res)
    except Exception as e:
        logger.warning(f"VoidVision swap failed ({e}), falling back to standard swap...")
        client = get_hf_client("tonyassi/face-swap", max_retries=2)
        res = client.predict(
            src_img=handle_file(source_face_path),
            dest_img=handle_file(target_scene_path),
            api_name="/swap_faces"
        )
        if isinstance(res, dict) and "path" in res:
            return res["path"]
        return str(res)

# ----------------- Multi-Engine Failover Controller -----------------
def generate_photorealistic_image(face_image_path: str, prompt: str, ratio_str: str = "9:16") -> str:
    """
    Multi-tier architecture for high-resolution photorealism:
    1. InstantID (Direct identity diffusion)
    2. PuLID-FLUX (Direct FLUX.1 ID)
    3. FLUX.1 + Enhanced Face Transfer (Zero-quota unlimited fallback)
    """
    logger.info(f"Generating realistic image for: {face_image_path} | prompt: '{prompt}' | ratio: '{ratio_str}'")
    
    w, h = FLUX_DIMENSIONS.get(ratio_str, (896, 1152))
    enhanced_prompt = translate_hinglish_to_dslr_prompt(prompt)
    negative_prompt = build_dslr_negative_prompt()

    logger.info(f"Optimized DSLR Prompt: {enhanced_prompt}")

    # --- Tier 1: InstantX/InstantID ---
    try:
        logger.info("Attempting Tier 1 Engine: InstantX/InstantID...")
        instant_client = get_hf_client("InstantX/InstantID", max_retries=2)
        res = instant_client.predict(
            face_image_path=handle_file(face_image_path),
            pose_image_path=handle_file(face_image_path),
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            style_name="(No style)",
            num_steps=35,
            identitynet_strength_ratio=1.15,
            adapter_strength_ratio=0.95,
            canny_strength=0.0,
            depth_strength=0.0,
            controlnet_selection=[],
            guidance_scale=4.5,
            seed=int(time.time()) % 1000000,
            scheduler="EulerDiscreteScheduler",
            enable_LCM=False,
            enhance_face_region=True,
            api_name="/generate_image"
        )

        if isinstance(res, (list, tuple)) and len(res) > 0:
            raw_path = res[0]
            if isinstance(raw_path, dict) and "path" in raw_path:
                raw_path = raw_path["path"]
            logger.info(f"Tier 1 InstantID success: {raw_path}")
            return apply_aspect_ratio(str(raw_path), ratio_str)
    except Exception as e:
        logger.warning(f"Tier 1 InstantID unavailable ({e}). Moving to Tier 2...")

    # --- Tier 2: yanze/PuLID-FLUX ---
    try:
        logger.info("Attempting Tier 2 Engine: yanze/PuLID-FLUX...")
        flux_client = get_hf_client("yanze/PuLID-FLUX", max_retries=2)
        res = flux_client.predict(
            prompt=enhanced_prompt,
            id_image=handle_file(face_image_path),
            start_step=0,
            guidance=4.0,
            seed=str(int(time.time()) % 1000000),
            true_cfg=1.0,
            width=w,
            height=h,
            num_steps=28,
            id_weight=1.30,
            neg_prompt=negative_prompt,
            timestep_to_start_cfg=1.0,
            max_sequence_length=512,
            api_name="/generate_image"
        )
        if isinstance(res, (list, tuple)) and len(res) > 0:
            raw_path = res[0]
            if isinstance(raw_path, dict) and "path" in raw_path:
                raw_path = raw_path["path"]
            logger.info(f"Tier 2 PuLID-FLUX success: {raw_path}")
            return apply_aspect_ratio(str(raw_path), ratio_str)
    except Exception as e:
        logger.warning(f"Tier 2 PuLID-FLUX unavailable ({e}). Moving to Tier 3...")

    # --- Tier 3: FLUX.1 + Enhanced Face Transfer ---
    try:
        logger.info("Running Tier 3: FLUX.1 + Enhanced Face Transfer Engine...")
        base_scene = generate_flux_base_image(enhanced_prompt, ratio_str)
        enhanced_image = execute_enhanced_face_swap(face_image_path, base_scene)
        logger.info(f"Tier 3 Enhanced Engine success: {enhanced_image}")
        return apply_aspect_ratio(str(enhanced_image), ratio_str)
    except Exception as e:
        logger.error(f"Tier 3 execution error: {e}")
        raise ValueError("Image generation failed on all available AI engines. Please retry.")

# ----------------- Telegram Handlers -----------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {"face_path": None, "crop_path": None, "ratio": "9:16", "last_active": time.time()}

    welcome_text = (
        f"👋 **Namaste {user.first_name}!**\n\n"
        "Main ek **Ultra-Realistic DSLR AI Studio Bot** hu jo "
        "**100% Face Match, Real Skin Pores, DSLR Lighting, aur High Resolution** ke sath images generate karta hai!\n\n"
        "⚡ **Smart Hinglish Support:**\n"
        "Aap seedhe Hinglish me likh sakte hain:\n\n"
        "• *Full Body:* `full body photo khadi hui sexy black dress me`\n"
        "• *Kamar Tak:* `kamar tak photo red saree me jewelry ke sath`\n"
        "• *Cosplay / Creative:* `bunny cosplay dress pehna do luxury bedroom me`\n"
        "• *100% Face Lock:* Aankhein, naak, face structure exact wahi rahenge.\n\n"
        "🚀 **Kaise use karein?**\n"
        "1️⃣ Pehle apni ek **clear Face Photo** bhejo.\n"
        "2️⃣ Uske baad jo bhi style/outfit chahiye Hinglish ya English me likh kar bhej do!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "💡 **Hinglish Prompting Examples:**\n\n"
        "📏 **1. Full Body Shot (Sir se Pair tak):**\n"
        "• `full body photo khadi hui sexy black gown me, high heels, luxury hotel lobby`\n"
        "• `poori photo designer lehenga me khadi hui`\n\n"
        "📐 **2. Kamar Tak (Waist-Up Shot):**\n"
        "• `kamar tak photo sexy red lingerie me, soft studio lighting`\n"
        "• `half body photo silk saree aur necklace me`\n\n"
        "🔍 **3. Close Up (Face Shot):**\n"
        "• `close up portrait photo, sharp eyes, natural skin pores, 8k dslr`\n\n"
        "✨ **Note:** `/ratio` command se aap photo ka size (9:16, 16:9, 1:1) change kar sakte hain."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def ratio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    for ratio_id, ratio_label in AVAILABLE_RATIOS:
        keyboard.append([InlineKeyboardButton(ratio_label, callback_data=f"ratio:{ratio_id}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📐 **Apna pasandeeda Image Aspect Ratio select karein:**", reply_markup=reply_markup, parse_mode="Markdown")

async def ratio_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("ratio:"):
        selected_ratio = data.split("ratio:")[1]
        user_id = query.from_user.id
        if user_id not in user_sessions:
            user_sessions[user_id] = {"face_path": None, "crop_path": None, "ratio": selected_ratio, "last_active": time.time()}
        else:
            user_sessions[user_id]["ratio"] = selected_ratio
            
        await query.edit_message_text(f"✅ **Aspect Ratio set to:** `{selected_ratio}`\nAb apna Hinglish ya English prompt bhejiye!", parse_mode="Markdown")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
        for k in ["face_path", "crop_path"]:
            p = user_sessions[user_id].get(k)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    user_sessions[user_id] = {"face_path": None, "crop_path": None, "ratio": "9:16", "last_active": time.time()}
    await update.message.reply_text("🔄 **Memory Reset Done!**\nAb nayi reference face photo bhejiye.", parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state = user_sessions.get(user_id, {})
    has_face = "✅ Loaded" if user_state.get("face_path") and os.path.exists(user_state.get("face_path")) else "❌ Not Uploaded"
    current_ratio = user_state.get("ratio", "9:16")
    
    status_msg = (
        "📊 **Bot Status:**\n"
        "• **AI Pipeline:** 3-Tier Multi-Engine (InstantID + FLUX.1 + High-Res Face Enhancer)\n"
        "• **Face Lock:** 🟢 100% High-Fidelity & Skin Texture\n"
        f"• **Target Ratio:** `{current_ratio}`\n"
        f"• **Face Status:** {has_face}\n"
        "• **Server Mode:** Koyeb (512MB RAM Optimized)\n"
        "• **System Health:** 🟢 Online & Ready"
    )
    await update.message.reply_text(status_msg, parse_mode="Markdown")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    cleanup_old_files()

    photo_obj = update.message.photo[-1]
    photo_file = await photo_obj.get_file()
    
    user_face_file = TEMP_DIR / f"face_{user_id}_{int(time.time())}.jpg"
    await photo_file.download_to_drive(custom_path=str(user_face_file))
    
    crop_path = create_high_res_face_crop(str(user_face_file), user_id)

    if user_id not in user_sessions:
        user_sessions[user_id] = {"face_path": str(user_face_file), "crop_path": crop_path, "ratio": "9:16", "last_active": time.time()}
    else:
        user_sessions[user_id]["face_path"] = str(user_face_file)
        user_sessions[user_id]["crop_path"] = crop_path
        user_sessions[user_id]["last_active"] = time.time()

    caption = update.message.caption
    if caption and len(caption.strip()) > 2:
        await update.message.reply_text("✅ **Reference Face Saved!**\nDirect prompt detect hua hai, Real DSLR generation start kar raha hu...")
        await generate_image_flow(update, context, caption.strip())
    else:
        await update.message.reply_text(
            "✅ **Face & Micro-Features Successfully Saved!**\n\n"
            "Ab bataiye kaisa photo chahiye? (Seedhe Hinglish me likhein):\n"
            "• *Example:* `kamar tak photo sexy red lingerie me seductive pose`\n"
            "• *Example:* `full body photo designer lehenga me khadi hui`\n\n"
            "📐 Ratio badalne ke liye `/ratio` dabayein.",
            parse_mode="Markdown"
        )

async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.mime_type and doc.mime_type.startswith("image/"):
        doc_file = await doc.get_file()
        user_id = update.effective_user.id
        user_face_file = TEMP_DIR / f"face_{user_id}_{int(time.time())}.jpg"
        await doc_file.download_to_drive(custom_path=str(user_face_file))
        
        crop_path = create_high_res_face_crop(str(user_face_file), user_id)

        if user_id not in user_sessions:
            user_sessions[user_id] = {"face_path": str(user_face_file), "crop_path": crop_path, "ratio": "9:16", "last_active": time.time()}
        else:
            user_sessions[user_id]["face_path"] = str(user_face_file)
            user_sessions[user_id]["crop_path"] = crop_path
            user_sessions[user_id]["last_active"] = time.time()
            
        await update.message.reply_text("✅ **High-Res Image Saved!**\nAb apna Hinglish ya English prompt bhejiye.", parse_mode="Markdown")
    else:
        await update.message.reply_text("⚠️ Kripya JPG ya PNG format ki image file bhejein.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.strip()
    await generate_image_flow(update, context, prompt)

async def generate_image_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    user = update.effective_user
    user_id = user.id
    
    session = user_sessions.get(user_id, {})
    crop_path = session.get("crop_path") or session.get("face_path")
    ratio_str = session.get("ratio", "9:16")

    if not crop_path or not os.path.exists(crop_path):
        await update.message.reply_text(
            "⚠️ **Reference Face Photo Missing!**\n"
            "Pehle apni ek face photo bhejiye, uske baad prompt likhiye.",
            parse_mode="Markdown"
        )
        return

    status_msg = await update.message.reply_text(
        f"⏳ **Generating Ultra-HD DSLR Photo...**\n"
        f"• **Prompt:** _{prompt}_\n"
        f"• **Ratio:** `{ratio_str}`\n"
        "• **Quality:** 📸 100% Face Match + Enhanced Skin Pores\n\n"
        "⚡ *Processing image (20-30 seconds lag sakte hain, please wait...)*",
        parse_mode="Markdown"
    )

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

    start_time = time.time()

    try:
        output_image_path = await asyncio.to_thread(
            generate_photorealistic_image,
            face_image_path=crop_path,
            prompt=prompt,
            ratio_str=ratio_str
        )
        
        elapsed = round(time.time() - start_time, 1)

        if output_image_path and os.path.exists(output_image_path):
            caption_text = (
                f"✨ **Generation Successful!** ({elapsed}s)\n\n"
                f"📝 **Prompt:** {prompt}\n"
                f"📐 **Ratio:** {ratio_str}\n"
                "📸 **Quality:** 8K UHD DSLR Real Skin Texture & Face Match\n\n"
                "💡 *Nayi photo ke liye prompt bhejein, `/ratio` se size badlein, ya `/reset` karein.*"
            )
            with open(output_image_path, "rb") as img_f:
                await update.message.reply_photo(photo=img_f, caption=caption_text, parse_mode="Markdown")
            
            try:
                await status_msg.delete()
            except Exception:
                pass
        else:
            await status_msg.edit_text("❌ **Generation failed:** Model image generate nahi kar paya. Kripya doosri clear face photo try karein.")

    except Exception as e:
        logger.error(f"Prediction error for user {user_id}: {e}", exc_info=True)
        error_str = str(e)
        err_reply = f"❌ **Error:** `{error_str[:150]}`\n\nKripya thodi der baad dobara prompt bhejein."
        await status_msg.edit_text(err_reply, parse_mode="Markdown")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN is not set. Exiting.")
        sys.exit(1)

    print("========================================")
    print("🤖 Starting Ultra-HD DSLR Face Bot")
    print("📦 Pipeline: FLUX.1 + Enhanced Face Restoration")
    print("💎 Quality: 8K Photorealism + Real Skin Texture")
    print(f"💾 Temp Directory: {TEMP_DIR}")
    print("========================================")

    start_koyeb_health_server()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ratio", ratio_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    
    app.add_handler(CallbackQueryHandler(ratio_callback_handler, pattern="^ratio:"))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
