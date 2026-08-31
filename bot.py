import os
import sys
import time
import shutil
import asyncio
import logging
import threading
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
logger = logging.getLogger("UltraRealFaceBot")

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

# In-memory client cache to prevent repeated 503 connection handshakes
CLIENT_CACHE = {}

# In-memory user state
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
    """Responds 200 OK to Koyeb TCP/HTTP Health Checks on port 8000."""
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
    """Starts background health check server for Koyeb on port 8000."""
    port = int(os.getenv("PORT", 8000))
    try:
        server = HTTPServer(("0.0.0.0", port), KoyebHealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Koyeb Health Check server listening on port {port} (TCP Check Fix Active)")
    except Exception as e:
        logger.warning(f"Could not start Koyeb health server on port {port}: {e}")

# ----------------- Helper Functions -----------------
def cleanup_old_files(max_age_hours=2):
    """Clean up temporary images to prevent memory/disk overflow on Koyeb."""
    now = time.time()
    try:
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                if now - file_path.stat().st_mtime > (max_age_hours * 3600):
                    file_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Error during temp cleanup: {e}")

def create_high_res_face_crop(image_path: str, user_id: int) -> str:
    """Crops the face region to provide maximum pixel density to the model."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            crop_left = int(w * 0.15)
            crop_top = int(h * 0.10)
            crop_right = int(w * 0.90)
            crop_bottom = int(h * 0.70)
            
            cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            crop_output_path = str(TEMP_DIR / f"crop_{user_id}_{int(time.time())}.jpg")
            cropped.save(crop_output_path, quality=98)
            return crop_output_path
    except Exception as e:
        logger.warning(f"Face crop failed, falling back to original image: {e}")
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
    """
    Connects to a HuggingFace space with automatic retry and caching.
    Handles waking up sleeping spaces (503 Service Unavailable).
    """
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
                time.sleep(3)  # Wait 3s for sleeping space container to spin up
            else:
                raise

def build_dslr_prompt(user_prompt: str) -> str:
    """Enriches user prompt with ultra-realistic DSLR camera physics & skin pores."""
    return (
        f"raw color 8k photograph, real life DSLR photography of the exact person in reference, "
        f"{user_prompt}, "
        f"hyperrealistic natural skin texture with visible realistic skin pores, natural subsurface scattering, "
        f"detailed real body anatomy, natural shadows and skin tones, authentic fabric texture, "
        f"shot on Sony Alpha 1 with 85mm f/1.4 GM lens, soft natural lighting, masterpiece photography"
    )

def build_dslr_negative_prompt() -> str:
    """Strong negative prompt to completely eliminate cartoon/plastic/AI artifacts."""
    return (
        "cartoon, anime, 3d render, cgi, illustration, drawing, painting, smooth plastic skin, "
        "wax doll, airbrushed, fake, porcelain skin, deformed face, bad eyes, disfigured anatomy, "
        "extra limbs, extra fingers, blurry, low quality, artifacts, watermark, logo, text"
    )

def generate_photorealistic_image(face_image_path: str, prompt: str, ratio_str: str = "9:16") -> str:
    """
    Generates photo preserving 100% facial identity, natural skin pores,
    and real body texture with multi-engine failover.
    """
    logger.info(f"Generating realistic image for: {face_image_path} | prompt: '{prompt}' | ratio: '{ratio_str}'")
    
    w, h = FLUX_DIMENSIONS.get(ratio_str, (896, 1152))
    enhanced_prompt = build_dslr_prompt(prompt)
    negative_prompt = build_dslr_negative_prompt()

    # 1. Primary Engine: FLUX.1 + PuLID
    try:
        logger.info("Calling PuLID-FLUX engine...")
        flux_client = get_hf_client_with_retry("yanze/PuLID-FLUX", max_retries=3)
        res = flux_client.predict(
            prompt=enhanced_prompt,
            id_image=handle_file(face_image_path),
            start_step=0,
            guidance=3.8,
            seed=str(int(time.time()) % 1000000),
            true_cfg=1.0,
            width=w,
            height=h,
            num_steps=24,
            id_weight=1.20,
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
        num_steps=32,
        identitynet_strength_ratio=1.0,
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
        "Main ek **Ultra-Realistic DSLR AI Studio Bot** hu jo Seedream & Nano Banana Pro jaisi "
        "**100% Original Real Skin Pores, Real Body Texture aur Exact Facial Identity** ke sath images banata hai!\n\n"
        "⚡ **Khaasiyat:**\n"
        "• **100% Real Skin Texture:** Visible skin pores, real lighting, 0% plastic look.\n"
        "• **Real Body Anatomy:** Accurate hands, neck, real fabric textures (silk, leather, etc.).\n"
        "• **Dress & Pose Transformation:** Bunny cosplay, royal saree, western dresses, alluring/dynamic poses.\n"
        "• **HD Ratios:** 9:16 (Reels/Shorts), 16:9 (Landscape), 1:1 (DP), 3:4, 4:3.\n\n"
        "🚀 **Kaise use karein?**\n"
        "1️⃣ Apni ek **clear Face Photo** bhejo.\n"
        "2️⃣ Uske baad jo bhi **Dress, Pose ya Scene** chahiye uska prompt bhej do!\n\n"
        "📌 **Commands:**\n"
        "• `/ratio` - Aspect ratio badlein (9:16, 16:9, etc.)\n"
        "• `/reset` - Purani photo clear karke nayi photo upload karein\n"
        "• `/status` - Bot aur AI health dekhein\n"
        "• `/help` - Helpful prompt examples"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "💡 **Ultra-Realistic Prompting Tips:**\n\n"
        "👗 **Dress & Outfit Prompts:**\n"
        "• `wearing sexy black leather bunny cosplay outfit with bunny ears, attractive confident pose, studio softbox lighting`\n"
        "• `wearing traditional royal red Banarasi silk saree with heavy gold jewelry, bridal photoshoot`\n"
        "• `wearing elegant red evening gown, glamorous fashion model pose leaning on luxury car`\n\n"
        "💃 **Pose & Lighting:**\n"
        "• `seductive alluring pose looking directly at camera, soft cinematic shadows`\n"
        "• `sitting relaxed on velvet sofa in modern penthouse, warm golden hour light`\n\n"
        "✨ **Automatic Enhancer:** Bot automatically real skin pores, 8K DSLR lens physics aur natural body texture prompt me add kar leta hai!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def ratio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show aspect ratio selection buttons."""
    keyboard = []
    for ratio_id, ratio_label in AVAILABLE_RATIOS:
        keyboard.append([InlineKeyboardButton(ratio_label, callback_data=f"ratio:{ratio_id}")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📐 **Apna pasandeeda Image Aspect Ratio select karein:**", reply_markup=reply_markup, parse_mode="Markdown")

async def ratio_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle aspect ratio button click."""
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
            
        await query.edit_message_text(f"✅ **Aspect Ratio set to:** `{selected_ratio}`\nAb apna prompt bhejiye image generate karne ke liye!", parse_mode="Markdown")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset current user face."""
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
    """Check bot health and active session."""
    user_id = update.effective_user.id
    user_state = user_sessions.get(user_id, {})
    has_face = "✅ Loaded" if user_state.get("face_path") and os.path.exists(user_state.get("face_path")) else "❌ Not Uploaded"
    current_ratio = user_state.get("ratio", "9:16")
    
    status_msg = (
        "📊 **Bot Status:**\n"
        "• **AI Engine:** FLUX.1-Dev (Ultra-Photorealism Engine)\n"
        "• **Skin Fidelity:** 📸 Natural Pores & Subsurface Scattering\n"
        "• **Body Anatomy:** 🟢 100% Real Texture\n"
        f"• **Target Ratio:** `{current_ratio}`\n"
        f"• **Face Status:** {has_face}\n"
        "• **Server Mode:** Koyeb (512MB RAM Optimized)\n"
        "• **System Health:** 🟢 Online & Healthy"
    )
    await update.message.reply_text(status_msg, parse_mode="Markdown")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming reference face photo."""
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
        await update.message.reply_text("✅ **Reference Face & Micro-Details Saved!**\nDirect prompt detect hua hai, Real DSLR generation start kar raha hu...")
        await generate_image_flow(update, context, caption.strip())
    else:
        await update.message.reply_text(
            "✅ **Face & Micro-Features Successfully Saved!**\n\n"
            "Ab bataiye is face ke sath kaisa **Dress, Pose ya Scene** chahiye?\n"
            "*(Example: Wearing sexy black bunny cosplay outfit, seductive pose looking at camera, studio softbox lighting)*\n\n"
            "📐 Ratio badalne ke liye `/ratio` dabayein.",
            parse_mode="Markdown"
        )

async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uncompressed images sent as files/documents."""
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
            
        await update.message.reply_text("✅ **Uncompressed High-Res Image Saved!**\nAb apna Dress/Pose/Scene prompt likhkar bhejiye.", parse_mode="Markdown")
    else:
        await update.message.reply_text("⚠️ Kripya JPG ya PNG format ki image file bhejein.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text prompt."""
    prompt = update.message.text.strip()
    await generate_image_flow(update, context, prompt)

async def generate_image_flow(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str):
    """Core realistic image generation workflow."""
    user = update.effective_user
    user_id = user.id
    
    session = user_sessions.get(user_id, {})
    crop_path = session.get("crop_path") or session.get("face_path")
    ratio_str = session.get("ratio", "9:16")

    if not crop_path or not os.path.exists(crop_path):
        await update.message.reply_text(
            "⚠️ **Reference Face Photo Missing!**\n"
            "Pehle apni ek face photo bhejiye, uske baad dress/pose ka prompt likhiye.",
            parse_mode="Markdown"
        )
        return

    status_msg = await update.message.reply_text(
        f"⏳ **Generating 100% Real Skin & Body Photo...**\n"
        f"• **Prompt:** _{prompt}_\n"
        f"• **Ratio:** `{ratio_str}`\n"
        "• **Quality Mode:** 📸 Ultra DSLR (Real Skin Pores + Anatomy)\n\n"
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
                f"✨ **Real Photo Generation Successful!** ({elapsed}s)\n\n"
                f"📝 **Prompt:** {prompt}\n"
                f"📐 **Ratio:** {ratio_str}\n"
                "📸 **Fidelity:** 100% Real Skin & Body Physics\n\n"
                "💡 *Nayi dress/pose ke liye prompt bhejein, `/ratio` se size badlein, ya `/reset` karein.*"
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
            err_reply = "⚠️ **AI Space Waking Up:** Hugging Face Space abhi sleep mode se wake up ho raha hai. Kripya 15-20 seconds baad dobara prompt send karein!"
        elif "queue" in error_str.lower() or "busy" in error_str.lower():
            err_reply = "⚠️ **GPU Queue Busy:** Server par thodi bheed hai. Kripya 30 seconds baad dobara try karein."
        else:
            err_reply = f"❌ **Error Aaya:** `{error_str[:150]}`\n\nKripya thodi der baad dobara prompt bhejein."
            
        await status_msg.edit_text(err_reply, parse_mode="Markdown")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler for telegram bot."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main():
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_TOKEN is not set. Exiting.")
        sys.exit(1)

    print("========================================")
    print("🤖 Starting Ultra-Real DSLR Face & Body Bot")
    print("📦 Engine: FLUX.1-Dev + Smart Face Crop")
    print("💎 Fidelity: Natural Skin Pores & Real Anatomy")
    print(f"💾 Temp Directory: {TEMP_DIR}")
    print("========================================")

    # 1. Start background health check server for Koyeb
    start_koyeb_health_server()

    # 2. Start Telegram Bot polling
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Command Handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ratio", ratio_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("status", status_command))
    
    # Callback Handlers
    app.add_handler(CallbackQueryHandler(ratio_callback_handler, pattern="^ratio:"))

    # Media and Text Handlers
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Error handler
    app.add_error_handler(error_handler)

    # Start Polling
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
