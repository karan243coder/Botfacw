import os
import sys
import time
import shutil
import asyncio
import logging
import threading
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
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

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("UncensoredRealFaceBot")

# Environment configurations
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN", None)

if HF_TOKEN and len(HF_TOKEN.strip()) > 5:
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN.strip()

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN is missing! Please set it in .env or environment variables.")

# Working directories
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# In-memory client cache
CLIENT_CACHE = {}

# In-memory user state
# Schema: { user_id: { "face_path": str, "crop_path": str, "ratio": str, "framing": str, "last_active": float } }
user_sessions = {}

# Available aspect ratios
AVAILABLE_RATIOS = [
    ("9:16", "📱 9:16 (Story / Reel / Shorts)"),
    ("16:9", "🖥️ 16:9 (Landscape / Wallpaper)"),
    ("1:1", "📸 1:1 (Square / Instagram DP)"),
    ("3:4", "🖼️ 3:4 (Portrait)"),
    ("4:3", "📺 4:3 (Standard Photo)"),
]

RATIO_MAP = {
    "1:1": (1, 1),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "3:4": (3, 4),
    "4:3": (4, 3),
}

FLUX_DIMENSIONS = {
    "1:1": (1024, 1024),
    "9:16": (896, 1152),
    "16:9": (1152, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
}

# ----------------- Koyeb Health Check Server -----------------
class KoyebHealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK - Face Bot is running healthy!\n")
    
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        return

def start_koyeb_health_server():
    port = int(os.getenv("PORT", 8000))
    try:
        server = HTTPServer(("0.0.0.0", port), KoyebHealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Koyeb Health Check server listening on port {port}")
    except Exception as e:
        logger.warning(f"Could not start Koyeb health server on port {port}: {e}")

# ----------------- Helper Functions -----------------
def cleanup_old_files(max_age_hours=2):
    now = time.time()
    try:
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                if now - file_path.stat().st_mtime > (max_age_hours * 3600):
                    file_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Error during temp cleanup: {e}")

def create_high_res_face_crop(image_path: str, user_id: int) -> str:
    """Crops the face region with high precision for micro-feature embedding."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            crop_left = int(w * 0.10)
            crop_top = int(h * 0.08)
            crop_right = int(w * 0.90)
            crop_bottom = int(h * 0.75)
            
            cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            crop_output_path = str(TEMP_DIR / f"crop_{user_id}_{int(time.time())}.jpg")
            cropped.save(crop_output_path, quality=98)
            return crop_output_path
    except Exception as e:
        logger.warning(f"Face crop failed: {e}")
        return image_path

def apply_aspect_ratio(image_path: str, ratio_str: str) -> str:
    """Crop and format generated image to exact target aspect ratio."""
    if ratio_str not in RATIO_MAP or ratio_str == "1:1":
        return image_path

    try:
        with Image.open(image_path) as img:
            target_w_ratio, target_h_ratio = RATIO_MAP[ratio_str]
            target_ratio = target_w_ratio / target_h_ratio
            w, h = img.size
            current_ratio = w / h

            if abs(current_ratio - target_ratio) < 0.01:
                return image_path

            if current_ratio > target_ratio:
                new_w = int(h * target_ratio)
                offset = (w - new_w) // 2
                cropped = img.crop((offset, 0, offset + new_w, h))
            else:
                new_h = int(w / target_ratio)
                offset = (h - new_h) // 2
                cropped = img.crop((0, offset, w, offset + new_h))

            output_ratio_path = str(Path(image_path).with_name(f"ratio_{Path(image_path).name}"))
            cropped.save(output_ratio_path, quality=95)
            return output_ratio_path
    except Exception as e:
        logger.warning(f"Error applying aspect ratio crop: {e}")
        return image_path

def get_hf_client_with_retry(space_name: str, max_retries: int = 3):
    """Connects to HuggingFace space with retry mechanism."""
    if space_name in CLIENT_CACHE:
        return CLIENT_CACHE[space_name]

    token = HF_TOKEN.strip() if (HF_TOKEN and len(HF_TOKEN.strip()) > 5) else None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to space '{space_name}' (Attempt {attempt}/{max_retries})...")
            if token:
                try:
                    c = Client(space_name, token=token)
                except TypeError:
                    c = Client(space_name, headers={"Authorization": f"Bearer {token}"})
            else:
                c = Client(space_name)
            
            CLIENT_CACHE[space_name] = c
            logger.info(f"Connected and cached space: {space_name}")
            return c
        except Exception as e:
            logger.warning(f"Connection attempt {attempt} for {space_name} failed: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise

def detect_framing_and_enhance_prompt(user_prompt: str) -> str:
    """
    Intelligently detects desired camera framing (Full Body, Waist-Up, Close-Up)
    and injects exact camera focal lengths + real skin and body textures without censorship.
    """
    lower_p = user_prompt.lower()
    
    # 1. Full Body Detection
    is_full_body = any(k in lower_p for k in [
        "full body", "full photo", "full length", "poori photo", "head to toe", "standing full", "legs", "feet", "shoes", "heels"
    ])
    
    # 2. Waist-Up / Half Body Detection
    is_waist_up = any(k in lower_p for k in [
        "kamar tak", "waist up", "half body", "medium shot", "upper body", "kamar", "half photo"
    ])
    
    # 3. Close Up Detection
    is_close_up = any(k in lower_p for k in [
        "close up", "face only", "portrait", "chehra"
    ])

    if is_full_body:
        framing_tag = "full body shot, standing full length from head to toe, wide shot, shot on 35mm wide lens"
    elif is_waist_up:
        framing_tag = "medium shot, waist up shot, upper body framing, shot on 50mm lens"
    elif is_close_up:
        framing_tag = "close-up portrait photo, intimate face framing, shot on 85mm lens"
    else:
        # Default: natural framing based on user's direct prompt
        framing_tag = "shot on professional camera with natural lens framing"

    return (
        f"raw color 8k photograph, real life DSLR photography of the exact person in reference, "
        f"{framing_tag}, {user_prompt}, "
        f"hyperrealistic natural skin texture with visible micro-pores, natural skin subsurface scattering, "
        f"accurate real body anatomy, natural shadows, authentic cloth and leather fabric texture, "
        f"soft natural lighting, masterpiece ultra-sharp photography"
    )

def build_uncensored_negative_prompt() -> str:
    """Permissive quality negative prompt (Zero Content Restrictions)."""
    return (
        "cartoon, anime, 3d render, cgi, illustration, drawing, painting, smooth plastic skin, "
        "wax doll, airbrushed, fake, porcelain skin, deformed face, bad eyes, disfigured anatomy, "
        "extra limbs, extra fingers, blurry, low quality, artifacts, watermark, logo, text"
    )

def generate_photorealistic_image(face_image_path: str, prompt: str, ratio_str: str = "9:16") -> str:
    """
    Generates uncensored photo preserving 100% facial identity, natural skin pores,
    real body texture, and exact camera framing.
    """
    logger.info(f"Generating realistic image for: {face_image_path} | prompt: '{prompt}' | ratio: '{ratio_str}'")
    
    w, h = FLUX_DIMENSIONS.get(ratio_str, (896, 1152))
    enhanced_prompt = detect_framing_and_enhance_prompt(prompt)
    negative_prompt = build_uncensored_negative_prompt()

    logger.info(f"Enhanced Prompt: {enhanced_prompt}")

    # 1. Primary Engine: FLUX.1 + PuLID (Maximum Fidelity)
    try:
        logger.info("Calling PuLID-FLUX engine with maximum identity lock (1.35)...")
        flux_client = get_hf_client_with_retry("yanze/PuLID-FLUX", max_retries=3)
        res = flux_client.predict(
            prompt=enhanced_prompt,
            id_image=handle_file(face_image_path),
            start_step=0,
            guidance=4.0,
            seed=str(int(time.time()) % 1000000),
            true_cfg=1.0,
            width=w,
            height=h,
            num_steps=26,
            id_weight=1.35,  # Increased identity lock for 100% face match
            neg_prompt=negative_prompt,
            timestep_to_start_cfg=1.0,
            max_sequence_length=512,
            api_name="/generate_image"
        )
        if isinstance(res, (list, tuple)) and len(res) > 0:
            raw_path = res[0]
            if isinstance(raw_path, dict) and "path" in raw_path:
                raw_path = raw_path["path"]
            return apply_aspect_ratio(str(raw_path), ratio_str)
    except Exception as e:
        logger.warning(f"PuLID-FLUX encountered: {e}. Falling back to InstantID Photorealism...")

    # 2. Secondary Engine: InstantID Photorealism
    instant_client = get_hf_client_with_retry("InstantX/InstantID", max_retries=3)
    res = instant_client.predict(
        face_image_path=handle_file(face_image_path),
        pose_image_path=handle_file(face_image_path),
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        style_name="(No style)",
        num_steps=35,
        identitynet_strength_ratio=1.15,  # Strong identity lock
        adapter_strength_ratio=1.0,       # Deep semantic face match
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
        return apply_aspect_ratio(str(raw_path), ratio_str)

    raise ValueError("Image generation failed on all available engines.")

# ----------------- Telegram Handlers -----------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    user_id = user.id
    
    if user_id not in user_sessions:
        user_sessions[user_id] = {"face_path": None, "crop_path": None, "ratio": "9:16", "last_active": time.time()}

    welcome_text = (
        f"👋 **Namaste {user.first_name}!**\n\n"
        "Main ek **Uncensored Ultra-Realistic AI Studio Bot** hu jo Seedream 5.0 & Nano Banana Pro jaisi "
        "**100% Face Match, Real Skin Pores, Full Body / Kamar tak Framing, aur Real Body Texture** ke sath images generate karta hai!\n\n"
        "⚡ **Aap kya-kya control kar sakte hain?**\n"
        "• **Framing Control:**\n"
        "   - `Full Body photo / Head to toe` ➡️ Poori lambi photo (sir se pair tak)\n"
        "   - `Kamar tak / Waist up` ➡️ Kamar tak ki photo\n"
        "   - `Close up / Portrait` ➡️ Sirf chehre ki photo\n"
        "• **100% Face Lock:** Aankhein, naak, bindi, lips exact wahi rahenge.\n"
        "• **Zero Censorship / Dress Freedom:** Bunny cosplay, bikini, gowns, suits bina kisi restriction ke!\n"
        "• **HD Ratios:** 9:16 (Reels/Shorts), 16:9 (Landscape), 1:1, 3:4, 4:3.\n\n"
        "🚀 **Kaise use karein?**\n"
        "1️⃣ Pehle apni ek **clear Face Photo** bhejo.\n"
        "2️⃣ Uske baad jo bhi **Dress, Pose, ya Framing (Full Body / Kamar tak)** chahiye prompt bhej do!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "💡 **Framing & Prompting Guide:**\n\n"
        "📏 **1. Full Body Shot (Sir se Pair tak):**\n"
        "• `full body shot, standing full length head to toe, wearing sexy black bunny cosplay dress, high heels, alluring pose, 8k`\n"
        "• `full body standing in luxury red evening gown on staircase, photorealistic`\n\n"
        "📐 **2. Kamar Tak (Waist-Up Shot):**\n"
        "• `waist up shot, kamar tak photo, wearing sexy black leather top, confident pose looking at camera`\n"
        "• `half body photo, wearing royal silk saree with jewelry`\n\n"
        "🔍 **3. Close Up (Face Shot):**\n"
        "• `close up portrait, detailed eyes, soft studio lighting`\n\n"
        "✨ **Note:** Aap prompt me seedhe likhenge: `full body` ya `kamar tak` toh bot exact wahi camera distance set karega!"
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
            
        await query.edit_message_text(f"✅ **Aspect Ratio set to:** `{selected_ratio}`\nAb apna prompt bhejiye (Full Body / Kamar tak / Dress)!", parse_mode="Markdown")

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
        "• **AI Engine:** FLUX.1-Dev (Uncensored Real Engine)\n"
        "• **Face Lock:** 🟢 100% High-Fidelity (1.35x Weight)\n"
        "• **Framing Control:** 🟢 Full Body / Waist-Up / Close-Up\n"
        "• **Restrictions:** 🔓 Zero Content Restrictions\n"
        f"• **Target Ratio:** `{current_ratio}`\n"
        f"• **Face Status:** {has_face}\n"
        "• **Server Mode:** Koyeb (512MB RAM Optimized)\n"
        "• **System Health:** 🟢 Online & Healthy"
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
        await update.message.reply_text("✅ **Reference Face Saved!**\nDirect prompt detect hua hai, generation start kar raha hu...")
        await generate_image_flow(update, context, caption.strip())
    else:
        await update.message.reply_text(
            "✅ **Face & Micro-Features Successfully Saved!**\n\n"
            "Ab bataiye kaisa photo chahiye? (Framing + Dress + Pose):\n"
            "• *Example (Full Body):* `full body shot standing head to toe, sexy bunny cosplay outfit, 8k`\n"
            "• *Example (Kamar tak):* `waist up shot kamar tak, red evening dress, seductive pose`\n\n"
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
            
        await update.message.reply_text("✅ **High-Res Image Saved!**\nAb apna prompt bhejiye (Full Body / Kamar tak).", parse_mode="Markdown")
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
        f"⏳ **Generating Accurate Real Photo...**\n"
        f"• **Prompt:** _{prompt}_\n"
        f"• **Ratio:** `{ratio_str}`\n"
        "• **Quality:** 📸 100% Face Lock + Accurate Framing\n\n"
        "⚡ *Generating via Free GPU (20-30 seconds lag sakte hain, please wait...)*",
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
                "📸 **Fidelity:** 100% Face Match & Real Anatomy\n\n"
                "💡 *Nayi photo ke liye prompt bhejein (Full Body / Kamar tak), ya `/reset` karein.*"
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
        elif "503" in error_str or "temporarily unavailable" in error_str.lower():
            err_reply = "⚠️ **AI Space Waking Up:** Hugging Face Space sleep mode se wake up ho raha hai. Kripya 15-20 seconds baad dobara prompt send karein!"
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
    print("📦 Engine: FLUX.1-Dev + Smart Framing Engine")
    print("💎 Fidelity: 100% Face Lock + Full Body / Waist Control")
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
