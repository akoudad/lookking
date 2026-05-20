"""
Lookking Telegram Bot
State machine per chat:
  IDLE → user clicks /start → MODE_PICK
  MODE_PICK → user clicks Place/Job → AWAITING_QUERY
  AWAITING_QUERY → user sends text → run pipeline → RESULTS_SHOWN
  RESULTS_SHOWN → user clicks Done → IDLE
  RESULTS_SHOWN → user clicks Add Info → AWAITING_REFINEMENT
  AWAITING_REFINEMENT → user sends text → merge + run → RESULTS_SHOWN
"""
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.crew_setup import run_lookking
from utils.logger import log_action

load_dotenv(Path(__file__).parent.parent / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
executor = ThreadPoolExecutor(max_workers=3)

# Per-chat state machine
# _state[chat_id] = {
#   "stage": str,          # MODE_PICK | AWAITING_QUERY | RESULTS_SHOWN | AWAITING_REFINEMENT
#   "mode": str | None,    # "places" or "leads"
#   "last_query": str,     # last user query text
#   "last_result": str,    # last pipeline result
# }
_state: dict = {}


WELCOME = (
    "👋 *Welcome to Lookking!*\n\n"
    "I help you find:\n"
    "📍 *Places* — restaurants, spas, gyms, hotels...\n"
    "💼 *Leads* — business clients for your service\n\n"
    "Pick what you want below:"
)


def mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📍 Find a Place", callback_data="mode_places"),
            InlineKeyboardButton("💼 Find Leads", callback_data="mode_leads"),
        ]
    ])


def result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Done", callback_data="done"),
            InlineKeyboardButton("➕ Add Info", callback_data="add_info"),
        ]
    ])


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    _state[chat_id] = {"stage": "MODE_PICK", "mode": None, "last_query": "", "last_result": ""}
    await update.message.reply_text(WELCOME, parse_mode="Markdown", reply_markup=mode_keyboard())
    log_action("TelegramBot", "start", {"user": update.effective_user.username}, "mode picker shown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send /start to begin.\n\n"
        "Flow:\n"
        "1. Pick *Place* or *Leads*\n"
        "2. Describe what you want\n"
        "3. Get top 3 ranked results\n"
        "4. Click *Done* or *Add Info* to refine",
        parse_mode="Markdown",
    )


async def _run_and_send(update: Update, chat_id: int, username: str, full_query: str):
    """Run pipeline and send results with action buttons."""
    thinking = await (update.message or update.callback_query.message).reply_text(
        "🔍 *Lookking...* AI agents searching. ~20s.",
        parse_mode="Markdown",
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, run_lookking, full_query)
    except Exception as e:
        await thinking.delete()
        await (update.message or update.callback_query.message).reply_text(f"❌ Error: {e}")
        return

    await thinking.delete()

    _state[chat_id]["stage"] = "RESULTS_SHOWN"
    _state[chat_id]["last_query"] = full_query
    _state[chat_id]["last_result"] = result

    msg = (
        f"✅ Results for: {full_query[:200]}\n\n"
        f"{result}\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Happy with results?"
    )
    if len(msg) > 4000:
        msg = msg[:3950] + "...\n\nHappy with results?"

    # No parse_mode — result text may contain URLs/underscores that break Markdown
    await (update.message or update.callback_query.message).reply_text(
        msg, reply_markup=result_keyboard(), disable_web_page_preview=True,
    )
    log_action("TelegramBot", "sent_results", {"user": username, "query": full_query}, result[:300])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    username = update.effective_user.username or "user"

    st = _state.get(chat_id)
    if not st or st["stage"] == "IDLE":
        # No active session — guide back to /start
        await update.message.reply_text(
            "Send /start to begin a search.",
            reply_markup=mode_keyboard(),
        )
        _state[chat_id] = {"stage": "MODE_PICK", "mode": None, "last_query": "", "last_result": ""}
        return

    if st["stage"] == "MODE_PICK":
        await update.message.reply_text(
            "Pick a mode first 👇",
            reply_markup=mode_keyboard(),
        )
        return

    if st["stage"] == "AWAITING_QUERY":
        mode = st["mode"] or "places"
        full_query = f"[MODE: {mode}] {text}"
        log_action("TelegramBot", "new_query", {"user": username, "mode": mode}, text)
        await _run_and_send(update, chat_id, username, full_query)
        return

    if st["stage"] == "AWAITING_REFINEMENT":
        mode = st["mode"] or "places"
        prev = st["last_query"]
        # Strip [MODE: ...] prefix from prev to keep it clean, then re-prefix
        clean_prev = prev.split("] ", 1)[-1] if prev.startswith("[MODE:") else prev
        merged = f"[MODE: {mode}] {clean_prev}, refinement: {text}"
        log_action("TelegramBot", "refinement", {"user": username, "prev": clean_prev, "add": text}, merged)
        await _run_and_send(update, chat_id, username, merged)
        return

    if st["stage"] == "RESULTS_SHOWN":
        # User typed instead of clicking — treat as new query
        await update.message.reply_text(
            "Click *Add Info* to refine, or *Done* to finish — then /start for new search.",
            parse_mode="Markdown",
        )
        return


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cb = update.callback_query
    await cb.answer()

    chat_id = update.effective_chat.id
    data = cb.data
    username = update.effective_user.username or "user"

    if data == "mode_places":
        _state[chat_id] = {"stage": "AWAITING_QUERY", "mode": "places", "last_query": "", "last_result": ""}
        await cb.edit_message_reply_markup(reply_markup=None)
        await cb.message.reply_text(
            "📍 *Place mode.*\n\nDescribe what place you want.\n\n"
            "Examples:\n"
            "• `sushi restaurant Rabat open now`\n"
            "• `luxury spa Casablanca`\n"
            "• `cheap barbershop near me`",
            parse_mode="Markdown",
        )
        return

    if data == "mode_leads":
        _state[chat_id] = {"stage": "AWAITING_QUERY", "mode": "leads", "last_query": "", "last_result": ""}
        await cb.edit_message_reply_markup(reply_markup=None)
        await cb.message.reply_text(
            "💼 *Leads mode.*\n\nDescribe your service or the clients you target.\n\n"
            "Examples:\n"
            "• `I offer video editing for restaurants in Rabat`\n"
            "• `web design agency targeting hotels Casablanca`",
            parse_mode="Markdown",
        )
        return

    if data == "done":
        st = _state.get(chat_id, {})
        log_action("TelegramBot", "user_done", {"user": username, "query": st.get("last_query", "")}, "approved")
        _state[chat_id] = {"stage": "IDLE", "mode": None, "last_query": "", "last_result": ""}
        await cb.edit_message_reply_markup(reply_markup=None)
        await cb.message.reply_text(
            "✅ *Saved.* Send /start to begin a new search.",
            parse_mode="Markdown",
        )
        return

    if data == "add_info":
        st = _state.get(chat_id, {})
        if not st.get("last_query"):
            await cb.message.reply_text("No previous search. Send /start.")
            return
        _state[chat_id]["stage"] = "AWAITING_REFINEMENT"
        log_action("TelegramBot", "user_refine", {"user": username, "prev": st["last_query"]}, "asked for add info")
        await cb.edit_message_reply_markup(reply_markup=None)
        await cb.message.reply_text(
            "➕ *What to add or change?*\n\n"
            "Examples: `but in Rabat`, `cheaper`, `with parking`, `open now`",
            parse_mode="Markdown",
        )
        return


def run_bot():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set in .env")

    # Preload DL model on main thread (avoid I/O errors in worker threads)
    print("🧠 Preloading DL model...")
    from tools.dl_scorer_tool import _load_model
    _load_model()
    print("✅ Model loaded.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Lookking Bot is running... Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)
