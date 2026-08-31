# 🤖 AI Face-Consistency Telegram Bot (Seedream/InstantID Style)

100% Free AI Face-Consistent Image Generation Telegram Bot, optimized for **Koyeb Free Tier (512MB RAM)**.

Ye bot user ki reference face photo ko yaad rakhta hai aur naye prompts ke according exact same face ke sath nayi cinematic images generate karta hai — bina kisi paid API ke (Free HuggingFace ZeroGPU Cloud inference use karke).

---

## ✨ Features

- 🎯 **100% Face Identity Consistency** (InstantID + SDXL ZeroGPU architecture)
- 💰 **100% Free & Zero Cost** (Runs on Koyeb 512MB Free Tier + Free HF GPU backend)
- ⚡ **Lightweight & Fast** (Bot uses only ~50MB RAM on Koyeb)
- 🎨 **Multi-Style Presets** (`/style` command for Cyberpunk Neon, Cinematic Film Noir, Watercolor, Sci-Fi Mars, Line Art, etc.)
- 📸 **Direct Mode (1-Shot)** (Photo ke sath caption me prompt bhej kar direct generate karein)
- 🧹 **Auto-Cleanup** (Temp images automatically clear ho jaati hain, storage full nahi hoga)
- 🔒 **Zero Censorship / No Blocking** (Open-source weights, no strict prompt bans)

---

## 📁 Project Structure

```text
face_consistency_bot/
├── bot.py              # Main Telegram Bot logic & Gradio API integration
├── Dockerfile          # Koyeb container deployment file
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Ignored files (temp images, keys)
└── README.md           # Setup & Deployment Guide
```

---

## 🚀 Step-by-Step Deployment Guide

### Step 1: Telegram Bot Token Create Karein
1. Telegram open karein aur search karein: **`@BotFather`**
2. `/newbot` command send karein.
3. Apne bot ka **Name** aur **Username** enter karein (username ke end me `bot` hona chahiye, e.g. `MyFaceAiBot`).
4. `@BotFather` aapko ek **API Token** dega (e.g. `789234567:AAH...`). Ise copy karke safe rakhein.

---

### Step 2: GitHub Repository Banayein & Code Upload Karein
1. [GitHub.com](https://github.com) par login karein.
2. Top right me **`+`** icon par click karein -> **`New repository`**.
3. Repository name dein (e.g. `face-consistency-bot`), ise **Public** ya **Private** rakhein, aur **Create repository** par click karein.
4. Apne computer me is folder ki saari files ko GitHub repo me upload karein:

```bash
git init
git add .
git commit -m "Initial commit for Face Consistency Bot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
*(Ya fir GitHub website par direct "Upload files" button se saari files upload kar sakte hain)*

---

### Step 3: Koyeb Par Free Me Deploy Karein (512MB Free Tier)

1. [Koyeb.com](https://www.koyeb.com) par jayein aur **Free Account** banayein.
2. Dashboard me **`Create Service`** ya **`Create App`** par click karein.
3. Deployment method me **GitHub** select karein aur apni banayi hui repository choose karein.
4. **Builder:** `Dockerfile` automatically detect ho jayega.
5. **Instance Size:** **`Nano` (Free Tier - 512MB RAM)** select karein.
6. **Environment Variables** section me click karein aur add karein:
   - **Key:** `TELEGRAM_TOKEN`  
     **Value:** (Aapka @BotFather se mila token)
   - *(Optional)* **Key:** `HF_TOKEN`  
     **Value:** (HuggingFace ka free read token from `huggingface.co/settings/tokens` - queue me fast lane ke liye)
7. **Deploy** button par click karein!

Kuch hi seconds me aapka bot **Live / Healthy** ho jayega! 🎉

---

## 🎮 Telegram Commands

| Command | Description |
| :--- | :--- |
| `/start` | Bot ko start karein aur instructions dekhein |
| `/style` | Interactive buttons se visual style choose karein |
| `/reset` | Current reference face ko clear karein aur nayi photo upload karein |
| `/status` | Server aur backend status check karein |
| `/help` | Best consistency ke liye prompt tips dekhein |

---

## 💡 Best Prompting Tips

- **Face ki photo:** Clear selfie, good lighting, front ya 3/4 angle photo upload karein.
- **Prompts me:** Face features describe karne ki zaroorat nahi hai (model automatically reference photo ka face use karega).
  - *Example 1:* `a young handsome warrior in gold armor holding a shining sword, 8k, photorealistic, cinematic lighting`
  - *Example 2:* `cyberpunk detective standing in rain under neon billboards, highly detailed, film grain`
