
import os
import io
import time
import math
import logging
import sqlite3
import asyncio
from collections import defaultdict, deque
from typing import Optional, Tuple, List, Deque

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

import openai
from telegram import (
    Update,
    ChatPermissions,
    Message,
    ParseMode,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    ChatMemberHandler,
    filters,
)

# ----------------- CONFIG / ENV -----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenRouter key or OpenAI key
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")  # e.g. https://openrouter.ai/api/v1
CREATOR_NAME = os.getenv("CREATOR_NAME", "C0D3 BR34K3R")
WATERMARK = os.getenv("WATERMARK", f"— Created by {CREATOR_NAME}")
DEFAULT_WELCOME = os.getenv("DEFAULT_WELCOME", "Welcome! Be kind and follow the rules.")
DB_PATH = os.getenv("DB_PATH", "scorpio.db")

# Leveling / promotion config
XP_PER_MESSAGE = int(os.getenv("XP_PER_MESSAGE", "5"))
PROMOTION_LEVEL = int(os.getenv("PROMOTION_LEVEL", "5"))  # level required to auto-promote

# OpenAI model & settings (fast & affordable)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "400"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.6"))

# Rate limiting config (AI)
AI_COOLDOWN_SECONDS = int(os.getenv("AI_COOLDOWN_SECONDS", "30"))  # per-user cooldown for AI calls
USER_LAST_AI_CALL = {}  # user_id -> last timestamp

# Conversation memory
CONTEXT_HISTORY_SIZE = int(os.getenv("CONTEXT_HISTORY_SIZE", "6"))
CONTEXT_MEMORY = defaultdict(lambda: deque(maxlen=CONTEXT_HISTORY_SIZE))
# keys: (chat_id, user_id) -> deque of dicts like {"role":"user"/"assistant","content":...}

# Anti-flood config
MESSAGE_WINDOW_SECONDS = int(os.getenv("MESSAGE_WINDOW_SECONDS", "10"))
MESSAGE_WINDOW_LIMIT = int(os.getenv("MESSAGE_WINDOW_LIMIT", "5"))  # >5 messages in window triggers warn
USER_MESSAGE_TIMESTAMPS = defaultdict(lambda: deque())
USER_WARNINGS = defaultdict(int)  # user_id -> warn count
AUTO_MUTE_SECONDS_ON_SECOND_WARN = int(os.getenv("AUTO_MUTE_SECONDS_ON_SECOND_WARN", "60"))

# Bad words: prefer file badwords.txt, fallback to env BAD_WORDS (comma-separated)
BAD_WORDS_FILE = os.getenv("BAD_WORDS_FILE", "badwords.txt")
BAD_WORDS_ENV = os.getenv("BAD_WORDS", "")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

# ----------------- LOGGING -----------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("scorpio")

# ----------------- DATABASE -----------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        chat_id INTEGER PRIMARY KEY,
        welcome_text TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS muted (
        chat_id INTEGER,
        user_id INTEGER,
        until_ts INTEGER,
        PRIMARY KEY(chat_id, user_id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS levels (
        chat_id INTEGER,
        user_id INTEGER,
        xp INTEGER DEFAULT 0,
        level INTEGER DEFAULT 0,
        PRIMARY KEY(chat_id, user_id)
    )""")
    con.commit()
    con.close()

def set_welcome(chat_id: int, text: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO settings (chat_id, welcome_text) VALUES (?, ?) ON CONFLICT(chat_id) DO UPDATE SET welcome_text=excluded.welcome_text",
        (chat_id, text),
    )
    con.commit()
    con.close()

def get_welcome(chat_id: int) -> str:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT welcome_text FROM settings WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else DEFAULT_WELCOME

def add_mute(chat_id: int, user_id: int, until_ts: Optional[int]):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO muted (chat_id, user_id, until_ts) VALUES (?, ?, ?)",
                (chat_id, user_id, until_ts))
    con.commit()
    con.close()

def remove_mute(chat_id: int, user_id: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM muted WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    con.commit()
    con.close()

def is_muted(chat_id: int, user_id: int) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT until_ts FROM muted WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    until_ts = row[0]
    if until_ts is None:
        return True
    return time.time() < until_ts

def get_levels_top(chat_id: int, limit: int = 10) -> List[Tuple[int,int,int]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT user_id, xp, level FROM levels WHERE chat_id=? ORDER BY xp DESC LIMIT ?", (chat_id, limit))
    rows = cur.fetchall()
    con.close()
    return rows

# ----------------- LEVELING -----------------
def calculate_level_from_xp(xp: int) -> int:
    return int(math.sqrt(max(0, xp) / 20))

def add_xp(chat_id: int, user_id: int, amount: int = XP_PER_MESSAGE) -> Tuple[bool, int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT xp, level FROM levels WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    row = cur.fetchone()
    if row:
        xp, level = row
        xp += amount
    else:
        xp, level = amount, 0
    new_level = calculate_level_from_xp(xp)
    leveled_up = new_level > level
    cur.execute("INSERT OR REPLACE INTO levels (chat_id, user_id, xp, level) VALUES (?, ?, ?, ?)",
                (chat_id, user_id, xp, new_level))
    con.commit()
    con.close()
    return leveled_up, new_level

def get_level(chat_id: int, user_id: int) -> Tuple[int,int]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT xp, level FROM levels WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    row = cur.fetchone()
    con.close()
    if row:
        return row[1], row[0]
    return 0, 0

# ----------------- BAD-WORD LOADING -----------------
def load_bad_words(filename=BAD_WORDS_FILE):
    words = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    words.add(w.lower())
    except FileNotFoundError:
        # fallback to env list
        if BAD_WORDS_ENV:
            for w in BAD_WORDS_ENV.split(","):
                w = w.strip()
                if w:
                    words.add(w.lower())
    return words

BAD_WORDS = load_bad_words()

# ----------------- OPENAI HELPER (with conversation memory) -----------------
async def ask_openai_with_context(chat_id: int, user_id: int, user_message: str) -> str:
    """
    Builds messages using conversation memory for (chat_id,user_id),
    sends to OpenAI/OpenRouter, stores assistant reply back in memory.
    """
    if not OPENAI_API_KEY:
        return "AI not configured."

    key = (chat_id, user_id)
    history: Deque = CONTEXT_MEMORY[key]

    # append new user message to history
    history.append({"role": "user", "content": user_message})

    # create messages list for OpenAI: add a light system prompt
    messages = [
        {"role": "system", "content": "You are Scorpio, a helpful Telegram group assistant. Keep answers concise and polite."}
    ]
    # include history
    messages.extend(list(history))

    loop = asyncio.get_event_loop()
    def _call():
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.exception("OpenAI error")
            return f"AI error: {e}"
    ai_reply = await loop.run_in_executor(None, _call)

    # store assistant reply in memory
    history.append({"role": "assistant", "content": ai_reply})
    # deque auto-trims to maxlen
    CONTEXT_MEMORY[key] = history
    return ai_reply

# ----------------- AI COOLDOWN -----------------
def can_use_ai(user_id: int) -> Tuple[bool, int]:
    now = time.time()
    last = USER_LAST_AI_CALL.get(user_id, 0)
    diff = now - last
    if diff < AI_COOLDOWN_SECONDS:
        return False, int(AI_COOLDOWN_SECONDS - diff)
    USER_LAST_AI_CALL[user_id] = now
    return True, 0

# ----------------- WELCOME IMAGE -----------------
def make_welcome_image(name: str, welcome_text: str, watermark_text: str) -> io.BytesIO:
    width, height = 900, 300
    img = Image.new("RGB", (width, height), (10, 10, 12))  # dark background
    draw = ImageDraw.Draw(img)

    # fonts (try common fonts; fallback to default)
    try:
        font_main = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_wm = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_wm = ImageFont.load_default()

    # neon accent effect: draw shadow then text
    accent = (0, 255, 255)  # cyan neon
    shadow = (0, 60, 60)
    # Name
    x_name, y_name = 40, 60
    # shadow
    draw.text((x_name+2, y_name+2), f"Welcome, {name}!", font=font_main, fill=shadow)
    draw.text((x_name, y_name), f"Welcome, {name}!", font=font_main, fill=accent)
    # Welcome text
    draw.text((x_name, y_name+70), welcome_text, font=font_sub, fill=(200, 200, 200))
    # Watermark bottom-left
    draw.text((40, height-40), watermark_text, font=font_wm, fill=(120,200,200))
    # Scorpio emoji logo bottom-right
    try:
        draw.text((width-120, height-70), "🦂", font=font_main, fill=accent)
    except:
        pass

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ----------------- HELPERS -----------------
async def require_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    chat = update.effective_chat
    if chat is None or user is None:
        return False
    if chat.type == "private":
        await update.message.reply_text("This command is for groups only!!!.")
        return False
    try:
        member = await chat.get_member(user.id)
        if member.status not in ("administrator", "creator"):
            await update.message.reply_text("You must be an admin to use this command.")
            return False
    except Exception:
        await update.message.reply_text("Failed to check admin status.")
        return False
    return True

async def _resolve_target_from_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.reply_to_message:
        return update.message.reply_to_message.from_user
    if context.args:
        arg = context.args[0]
        if arg.startswith("@"):
            arg = arg[1:]
        try:
            found = await context.bot.get_chat_member(update.effective_chat.id, int(arg))
            return found.user
        except Exception:
            # can't reliably resolve username -> return None
            return None
    return None

# ----------------- HANDLERS -----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Scorpio 🦂 at your service! Use /help to see commands.\n\n{WATERMARK}")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Scorpio 🦂 commands:\n"
        "/help - show this message\n"
        "/ask <question> - ask the AI (works in group or PM)\n"
        "/rank - show your level & XP\n"
        "/top - show top active users\n"
        "/setwelcome <text> - set welcome message (admins only)\n"
        "/getwelcome - show current welcome text (admins only)\n"
        "/kick - reply to a user to kick (admins only)\n"
        "/ban - reply to a user to ban (admins only)\n"
        "/unban <user_id> - unban by id (admins only)\n"
        "/mute - reply to a user to mute (admins only)\n"
        "/unmute - reply to a user to unmute (admins only)\n"
        "/promote - reply to a user to grant moderator-like rights (admins only)\n"
    )
    await update.message.reply_text(txt + f"\n{WATERMARK}")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else ""
    if not txt:
        await update.message.reply_text("Usage: /ask What is ...")
        return

    # cooldown check
    ok, wait = can_use_ai(update.effective_user.id)
    if not ok:
        await update.message.reply_text(f"🕒 Please wait {wait}s before asking again.")
        return

    sent = await update.message.reply_text("Thinking... 🤔")
    ai_answer = await ask_openai_with_context(update.effective_chat.id, update.effective_user.id, txt)
    await sent.edit_text(f"{ai_answer}\n\n{WATERMARK}")

async def kick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    target = await _resolve_target_from_message(update, context)
    if not target:
        await update.message.reply_text("Reply to a user's message to kick them.")
        return
    try:
        await context.bot.kick_chat_member(update.effective_chat.id, target.id)
        await update.message.reply_text(f"{target.full_name} has been kicked.\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed to kick: {e}")

async def ban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    target = await _resolve_target_from_message(update, context)
    if not target:
        await update.message.reply_text("Reply to a user's message to ban them.")
        return
    try:
        await context.bot.ban_chat_member(update.effective_chat.id, target.id)
        await update.message.reply_text(f"{target.full_name} has been banned.\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed to ban: {e}")

async def unban_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    if not context.args:
        await update.message.reply_text("Usage: /unban <user_id>")
        return
    try:
        uid = int(context.args[0])
        await context.bot.unban_chat_member(update.effective_chat.id, uid)
        await update.message.reply_text(f"Unbanned {uid}\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed: {e}")

async def mute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message to mute them.")
        return
    target = update.message.reply_to_message.from_user
    seconds = None
    if context.args:
        try:
            seconds = int(context.args[0])
        except:
            seconds = None
    until_ts = None
    if seconds:
        until_ts = int(time.time() + seconds)
    try:
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until_ts,
        )
        add_mute(update.effective_chat.id, target.id, until_ts)
        await update.message.reply_text(f"{target.full_name} muted.\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed to mute: {e}")

async def unmute_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message to unmute them.")
        return
    target = update.message.reply_to_message.from_user
    try:
        await context.bot.restrict_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target.id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_polls=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
                can_invite_users=True,
            ),
        )
        remove_mute(update.effective_chat.id, target.id)
        await update.message.reply_text(f"{target.full_name} is unmuted.\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed to unmute: {e}")

async def setwelcome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    txt = " ".join(context.args)
    if not txt:
        await update.message.reply_text("Usage: /setwelcome Welcome to ...")
        return
    set_welcome(update.effective_chat.id, txt)
    await update.message.reply_text(f"Welcome message set.\n{WATERMARK}")

async def getwelcome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return
    txt = get_welcome(update.effective_chat.id)
    await update.message.reply_text(f"Current welcome:\n{txt}\n\n{WATERMARK}")

async def promote_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin-only command: reply to a user to give moderator-like rights."""
    if not await require_admin(update, context):
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Reply to a user's message to promote them.")
        return
    target = update.message.reply_to_message.from_user
    try:
        await context.bot.promote_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target.id,
            can_change_info=False,
            can_post_messages=False,
            can_edit_messages=False,
            can_delete_messages=True,
            can_invite_users=True,
            can_restrict_members=True,
            can_pin_messages=False,
            can_promote_members=False,
        )
        await update.message.reply_text(f"{target.full_name} has been promoted.\n{WATERMARK}")
    except Exception as e:
        await update.message.reply_text(f"Failed to promote: {e}")

async def rank_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lvl, xp = get_level(update.effective_chat.id, update.effective_user.id)
    await update.message.reply_text(f"🏅 {update.effective_user.full_name}, Level {lvl} with {xp} XP.\n{WATERMARK}")

async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_levels_top(update.effective_chat.id, limit=10)
    if not rows:
        await update.message.reply_text("No activity recorded yet.")
        return
    text_lines = ["🏆 Top active users:"]
    for i, (user_id, xp, level) in enumerate(rows, start=1):
        try:
            user = await context.bot.get_chat_member(update.effective_chat.id, user_id)
            name = user.user.full_name
        except:
            name = f"user:{user_id}"
        text_lines.append(f"{i}. {name} — Level {level} ({xp} XP)")
    await update.message.reply_text("\n".join(text_lines) + f"\n\n{WATERMARK}")

# ----------------- MESSAGE / AI / XP / ANTI-FLOOD HANDLER -----------------
# For mentions / private chat - AI responds
async def handle_message_for_ai_and_xp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg: Message = update.message
    if not msg:
        return

    chat = update.effective_chat
    user = update.effective_user
    user_id = user.id
    chat_id = chat.id

    # If muted in DB: delete message and ignore
    if chat and is_muted(chat_id, user_id):
        try:
            await msg.delete()
        except:
            pass
        return

    text = (msg.text or "") if msg.text else ""

    # Anti-flood: track timestamps per user
    now = time.time()
    timestamps = USER_MESSAGE_TIMESTAMPS[user_id]
    timestamps.append(now)
    # drop old timestamps outside window
    while timestamps and (now - timestamps[0]) > MESSAGE_WINDOW_SECONDS:
        timestamps.popleft()
    # check threshold
    if len(timestamps) > MESSAGE_WINDOW_LIMIT:
        # increment warning count
        USER_WARNINGS[user_id] += 1
        warns = USER_WARNINGS[user_id]
        try:
            await chat.send_message(f"{user.mention_html()} Please slow down — you're sending messages too quickly. (Warn {warns})\n{WATERMARK}", parse_mode=ParseMode.HTML)
        except:
            pass
        # on second warning -> short mute
        if warns >= 2:
            until_ts = int(time.time() + AUTO_MUTE_SECONDS_ON_SECOND_WARN)
            try:
                await context.bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=user_id,
                    permissions=ChatPermissions(can_send_messages=False),
                    until_date=until_ts,
                )
                add_mute(chat_id, user_id, until_ts)
                await chat.send_message(f"{user.mention_html()} has been muted for {AUTO_MUTE_SECONDS_ON_SECOND_WARN}s due to repeated flooding.\n{WATERMARK}", parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning(f"Auto-mute failed: {e}")
            # reset warnings and timestamps
            USER_WARNINGS[user_id] = 0
            USER_MESSAGE_TIMESTAMPS[user_id].clear()
        return

    # Moderation: basic profanity deletion using BAD_WORDS
    lowered = text.lower() if text else ""
    for w in BAD_WORDS:
        if w and w in lowered:
            try:
                await msg.delete()
            except:
                pass
            try:
                await chat.send_message(f"{user.mention_html()} please avoid using that language.\n{WATERMARK}", parse_mode=ParseMode.HTML)
            except:
                pass
            return

    # Award XP for normal non-command messages (not bot commands)
    if text and not text.startswith("/"):
        leveled_up, new_level = add_xp(chat_id, user_id)
        if leveled_up:
            try:
                await chat.send_message(f"🎉 {user.mention_html()} leveled up to <b>Level {new_level}</b>!", parse_mode=ParseMode.HTML)
            except:
                pass
            # Try promote if threshold reached
            if new_level >= PROMOTION_LEVEL:
                try:
                    await context.bot.promote_chat_member(
                        chat_id=chat_id,
                        user_id=user_id,
                        can_change_info=False,
                        can_post_messages=False,
                        can_edit_messages=False,
                        can_delete_messages=True,
                        can_invite_users=True,
                        can_restrict_members=True,
                        can_pin_messages=False,
                        can_promote_members=False,
                    )
                    await chat.send_message(f"⚡ {user.mention_html()} promoted for reaching Level {new_level}!\n{WATERMARK}", parse_mode=ParseMode.HTML)
                except Exception as e:
                    logger.warning(f"Promotion failed: {e}")

    # Determine if the bot should respond with AI:
    should_respond = False
    # Private -> respond
    if chat.type == "private":
        should_respond = True
    else:
        # mention detection
        if context.bot.username and text and f"@{context.bot.username}" in text:
            should_respond = True
        elif msg.reply_to_message and msg.reply_to_message.from_user and msg.reply_to_message.from_user.is_bot:
            should_respond = True

    if not should_respond:
        return

    # AI cooldown check
    ok, wait = can_use_ai(user_id)
    if not ok:
        try:
            await msg.reply_text(f"🕒 Slow down! Please wait {wait}s before talking to me again.")
        except:
            pass
        return

    # Prepare prompt: strip bot mention
    user_text = text.replace(f"@{context.bot.username}", "").strip() if text else "Hi, how can I help?"
    sent = await msg.reply_text("Let me think... 🤖")
    ai_answer = await ask_openai_with_context(chat_id, user_id, user_text)
    try:
        await sent.edit_text(f"{ai_answer}\n\n{WATERMARK}")
    except:
        try:
            await sent.edit_text(f"{ai_answer}")
        except:
            pass

# ----------------- WELCOME HANDLER -----------------
async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.new_chat_members:
        return
    for new_user in update.message.new_chat_members:
        name = new_user.full_name
        txt = get_welcome(update.effective_chat.id)
        img_buf = make_welcome_image(name, txt, WATERMARK)
        try:
            await update.message.reply_photo(photo=img_buf, caption=f"{name}, {txt}\n\n{WATERMARK}")
        except Exception:
            # fallback to text if sending image fails
            await update.message.reply_text(f"{name}, {txt}\n\n{WATERMARK}")

async def chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # placeholder for future actions (e.g., track promotions/demotions)
    pass

# ----------------- START BOT -----------------
def main():
    init_db()
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set in .env")
        return
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("ask", ask_cmd))
    application.add_handler(CommandHandler("kick", kick_cmd))
    application.add_handler(CommandHandler("ban", ban_cmd))
    application.add_handler(CommandHandler("unban", unban_cmd))
    application.add_handler(CommandHandler("mute", mute_cmd))
    application.add_handler(CommandHandler("unmute", unmute_cmd))
    application.add_handler(CommandHandler("setwelcome", setwelcome_cmd))
    application.add_handler(CommandHandler("getwelcome", getwelcome_cmd))
    application.add_handler(CommandHandler("promote", promote_cmd))
    application.add_handler(CommandHandler("rank", rank_cmd))
    application.add_handler(CommandHandler("top", top_cmd))

    # Welcome handler
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_new_member))

    # All text messages -> moderation, XP, AI
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message_for_ai_and_xp))

    # Chat member updates
    application.add_handler(ChatMemberHandler(chat_member_update, ChatMemberHandler.CHAT_MEMBER))

    logger.info("Starting Scorpio 🦂")
    application.run_polling(allowed_updates=None)

if __name__ == "__main__":
    main()