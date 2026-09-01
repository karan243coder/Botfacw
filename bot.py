import os
import sys
import logging
import time
import asyncio
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Dict, Any

from PIL import Image
from dotenv import load_dotenv
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
    ("9:16", "📱 9:16 (Story / Reel / Portrait)"),
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
        self.wfile.write(b"OK - UltraRealFaceBot is running healthy")

    def log_message(self, format, *args):
        # Suppress noisy HTTP health logs
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
    """Auto cleanup temporary images older than 30 minutes to stay within Koyeb 512MB RAM limit."""
    now = time.time()
    try:
        for p in TEMP_DIR.glob("*"):
            if p.is_file() and (now - p.stat().st_mtime > 1800):
                p.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"File cleanup error: {e}")

def create_high_res_face_crop(image_path: str, user_id: int) -> str:
    """
    Crops the upper/center region of the uploaded image to focus
    on facial micro-features (eyes, nose, skin texture) for optimal identityNet locking.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            # If image is tall portrait, isolate top 65% where head/face is located
            if h > w:
                crop_box = (0, 0, w, int(h * 0.65))
            else:
                crop_box = (int(w * 0.15), 0, int(w * 0.85), h)

            cropped = img.crop(crop_box)
            crop_file = TEMP_DIR / f"crop_{user_id}_{int(time.time())}.jpg"
            cropped.save(crop_file, format="JPEG", quality=98)
            logger.info(f"High-res face crop created: {crop_file} ({cropped.size})")
            return str(crop_file)
    except Exception as e:
        logger.warning(f"Face crop fallback to original image: {e}")
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

            # Check if crop is necessary (threshold > 2%)
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
    """
    Loads Gradio Client via official Space Repo ID (e.g. 'InstantX/InstantID', 'yanze/PuLID-FLUX').
    Resolving by Space ID routes through Hugging Face's official API router,
    avoiding Cloudflare/503 errors and waking up sleeping ZeroGPU containers.
    """
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

# ----------------- Hinglish AI Prompt Engine -----------------
def translate_hinglish_to_dslr_prompt(user_prompt: str) -> str:
    """
    Translates raw Hinglish/Hindi user prompts into high-fidelity,
    photorealistic English diffusion prompts with camera optics & real skin physics.
    Zero censorship / Completely unrestricted.
    """
    raw = user_prompt.strip()
    p = raw.lower()

    # 1. Framing Detection
    framing = "medium shot, waist up shot, shot on 50mm lens"
    if any(k in p for k in ["full", "poori", "pura", "sir se pair", "khadi", "head to toe", "legs", "feet", "heels", "standing", "pair tak"]):
        framing = "full body shot, standing full length from head to toe, wide shot, shot on 35mm lens"
    elif any(k in p for k in ["kamar", "waist", "half", "aadha", "upper body", "chest", "chhati"]):
        framing = "medium shot, waist up framing, upper body, shot on 50mm lens"
    elif any(k in p for k in ["close up", "face", "chehra", "portrait", "sirf face", "eyes"]):
        framing = "close-up portrait shot, sharp focus on facial features and eyes, shot on 85mm f1.4 lens"

    # 2. Hinglish Vocabulary Replacements
    replacements = [
        # Framing & Poses
        (r'\bkamar tak\b', 'waist up'),
        (r'\bpoori photo\b', 'full body photo'),
        (r'\bpura photo\b', 'full body photo'),
        (r'\bsir se pair\b', 'head to toe'),
        (r'\bkhadi hui\b', 'standing gracefully'),
        (r'\bkhadi\b', 'standing'),
        (r'\bbaithi hui\b', 'sitting gracefully'),
        (r'\bbaithi\b', 'sitting'),
        (r'\bleti hui\b', 'reclining lying down sensually'),
        (r'\bsoi hui\b', 'lying in bed'),
        (r'\bsamne dekhte huye\b', 'looking directly at camera'),
        (r'\bsamne dekho\b', 'looking directly at camera'),
        (r'\bsamne dekh\b', 'looking directly at camera'),
        (r'\bsamne\b', 'looking at camera'),
        (r'\bhath kamar par\b', 'hands on hips seductive pose'),
        (r'\bmud kar\b', 'looking over shoulder'),

        # Outfits & Modifiers
        (r'\bpehan kar\b', 'wearing'),
        (r'\bpehna do\b', 'wearing'),
        (r'\bpehna\b', 'wearing'),
        (r'\bkapde\b', 'outfit'),
        (r'\bme\b', 'in'),
        (r'\bhawa me baal\b', 'wind-blown flowing natural hair'),
        (r'\bhawa me\b', 'wind blowing'),
        (r'\bbaal khule\b', 'open wavy long black hair'),
        (r'\bsundar\b', 'stunningly gorgeous'),
        (r'\bchhat par\b', 'on modern rooftop terrace'),
        (r'\bkamre me\b', 'in luxury bedroom'),
        (r'\bkamra\b', 'luxury bedroom'),
        (r'\bbed par\b', 'on luxury silk bed'),
        (r'\bsofe par\b', 'on luxury velvet couch'),
        (r'\bpani me\b', 'in crystal clear water'),
        (r'\bsamundar par\b', 'on tropical beach'),
        (r'\bchahiye\b', ''),
        (r'\bkar do\b', ''),
        (r'\bbana do\b', ''),
        (r'\bbanao\b', ''),
        (r'\bphoto\b', 'photograph'),
    ]

    translated_subject = p
    for pattern, repl in replacements:
        translated_subject = re.sub(pattern, repl, translated_subject)
    
    translated_subject = re.sub(r'\s+', ' ', translated_subject).strip()

    final_prompt = (
        f"raw color 8k photograph, exact 100% real life DSLR photography of the person in reference, "
        f"{framing}, {translated_subject}, "
        f"hyperrealistic natural skin texture with visible fine skin pores, authentic skin subsurface scattering, "
        f"100% anatomical body accuracy, natural skin tones and soft shadows, realistic fabric and skin physics, "
        f"shot on Sony Alpha 1, soft studio natural lighting, ultra-sharp optical clarity, masterpiece photography"
    )
    return final_prompt

def build_uncensored_negative_prompt() -> str:
    """Negative prompt focused strictly on preventing CGI / cartoon artifacts."""
    return (
        "cartoon, anime, 3d render, cgi, illustration, drawing, painting, smooth plastic skin, "
        "wax doll, airbrushed, fake, porcelain skin, deformed face, bad eyes, disfigured anatomy, "
        "extra limbs, extra fingers, blurry, low quality, artifacts, watermark, logo, text"
    )

def generate_photorealistic_image(face_image_path: str, prompt: str, ratio_str: str = "9:16") -> str:
    """
    Multi-engine failover pipeline:
    1. Primary: InstantX/InstantID (High Face Fidelity, Fast & Reliable)
    2. Secondary: yanze/PuLID-FLUX (FLUX.1-dev Photorealism)
    """
    logger.info(f"Generating realistic image for: {face_image_path} | prompt: '{prompt}' | ratio: '{ratio_str}'")
    
    w, h = FLUX_DIMENSIONS.get(ratio_str, (896, 1152))
    enhanced_prompt = translate_hinglish_to_dslr_prompt(prompt)
    negative_prompt = build_uncensored_negative_prompt()

    logger.info(f"Hinglish Translated Prompt: {enhanced_prompt}")

    # 1. Primary Engine: InstantX/InstantID
    try:
        logger.info("Calling Primary Engine: InstantX/InstantID...")
        instant_client = get_hf_client("InstantX/InstantID", max_retries=3)
        res = instant_client.predict(
            face_image_path=handle_file(face_image_path),
            pose_image_path=handle_file(face_image_path),
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            style_name="(No style)",
            num_steps=30,
            identitynet_strength_ratio=1.10,
            adapter_strength_ratio=0.90,
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
            logger.info(f"InstantID generated output successfully: {raw_path}")
            return apply_aspect_ratio(str(raw_path), ratio_str)
    except Exception as e:
        logger.warning(f"InstantX/InstantID encountered error: {e}. Failing over to PuLID-FLUX...")

    # 2. Secondary Engine: yanze/PuLID-FLUX
    try:
        logger.info("Calling Secondary Engine: yanze/PuLID-FLUX...")
        flux_client = get_hf_client("yanze/PuLID-FLUX", max_retries=3)
        res = flux_client.predict(
            prompt=enhanced_prompt,
            id_image=handle_file(face_image_path),
            start_step=0,
            guidance=4.0,
            seed=str(int(time.time()) % 1000000),
            true_cfg=1.0,
            width=w,
            height=h,
            num_steps=24,
            id_weight=1.25,
            neg_prompt=negative_prompt,
            timestep_to_start_cfg=1.0,
            max_sequence_length=512,
            api_name="/generate_image"
        )
        if isinstance(res, (list, tuple)) and len(res) > 0:
            raw_path = res[0]
            if isinstance(raw_path, dict) and "path" in raw_path:
                raw_path = raw_path["path"]
            logger.info(f"PuLID-FLUX generated output successfully: {raw_path}")
            return apply_aspect_ratio(str(raw_path), ratio_str)
    except Exception as e:
        logger.error(f"PuLID-FLUX encountered error: {e}")

    raise ValueError("Image generation failed on available AI engines. Please check your prompt or try again shortly.")

# ----------------- Telegram Handlers -----------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {"face_path": None, "crop_path": None, "ratio": "9:16", "last_active": time.time()}

    welcome_text = (
        f"👋 **Namaste {user.first_name}!**\n\n"
        "Main ek **Ultra-Realistic DSLR AI Studio Bot** hu jo Seedream 5.0 & Nano Banana Pro jaisi "
        "**100% Face Match, Real Skin Pores, Full Body / Kamar tak Framing, aur Real Body Texture** ke sath images generate karta hai!\n\n"
        "⚡ **Smart Hinglish Support:**\n"
        "Aap seedhe Hinglish me likh sakte hain — bot automatically samajh kar execute karega!\n\n"
        "• *Full Body:* `full body photo khadi hui bikini me`\n"
        "• *Kamar Tak:* `kamar tak photo sexy lingerie me seductive pose`\n"
        "• *Cosplay / Creative:* `sexy lingerie pehna do luxury bedroom me`\n"
        "• *100% Face Lock:* Aankhein, naak, bindi, lips exact wahi rahenge.\n"
        "• *Zero Restrictions:* Koi prompt block nahi hoga!\n\n"
        "🚀 **Kaise use karein?**\n"
        "1️⃣ Pehle apni ek **clear Face Photo** bhejo.\n"
        "2️⃣ Uske baad jo bhi prompt Hinglish ya English me likhna hai bhej do!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "💡 **Hinglish Prompting Examples:**\n\n"
        "📏 **1. Full Body Shot (Sir se Pair tak):**\n"
        "• `full body photo khadi hui sexy black bunny cosplay dress me, high heels, seductive pose`\n"
        "• `poori photo red silk gown me luxury hotel lobby me khadi`\n\n"
        "📐 **2. Kamar Tak (Waist-Up Shot):**\n"
        "• `kamar tak photo sexy leather top me, samne dekhte huye attractive pose`\n"
        "• `half body photo royal saree aur jewelry me`\n\n"
        "🔍 **3. Close Up (Face Shot):**\n"
        "• `close up face photo, detailed eyes, soft studio lighting`\n\n"
        "✨ **Note:** Aap Hindi/Hinglish me jaise bolenge, bot exact waisa hi frame aur dress banayega!"
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
        "• **AI Engine:** Official Hugging Face Gateway (InstantID + PuLID-FLUX)\n"
        "• **Face Lock:** 🟢 100% High-Fidelity\n"
        "• **Content Policy:** 🔓 100% Unrestricted / Uncensored\n"
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
            "• *Example:* `full body photo sexy bunny cosplay me khadi hui`\n"
            "• *Example:* `kamar tak photo sexy lingerie me alluring pose`\n\n"
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
        f"⏳ **Generating 100% Real Photo...**\n"
        f"• **Prompt:** _{prompt}_\n"
        f"• **Ratio:** `{ratio_str}`\n"
        "• **Quality:** 📸 100% Face Match + Real Body Physics\n\n"
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
                "📸 **Fidelity:** 100% Real Skin Pores & Exact Face Match\n\n"
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
        if "quota" in error_str.lower() or "zerogpu" in error_str.lower():
            err_reply = "⚠️ **Hugging Face Free Quota Exceeded:** Kripya apne Koyeb environment me free `HF_TOKEN` add karein ya thodi der baad try karein."
        elif "queue" in error_str.lower() or "busy" in error_str.lower():
            err_reply = "⚠️ **GPU Queue Busy:** Server par queue chal rahi hai. Kripya 30 seconds baad dobara try karein."
        else:
            err_reply = f"❌ **Error Aaya:** `{error_str[:150]}`\n\nKripya thodi der baad dobara prompt bhejein."
            
        await status_msg.edit_text(err_reply, parse_mode="Markdown")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN is not set. Exiting.")
        sys.exit(1)

    print("========================================")
    print("🤖 Starting Uncensored Real DSLR Face Bot")
    print("📦 Engine: Official Hugging Face Gateway Multi-Failover")
    print("💎 Fidelity: 100% Face Match + Zero Restrictions")
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
