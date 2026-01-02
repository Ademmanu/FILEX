#!/usr/bin/env python3
"""
FileX Bot - Clean message headers, only part headers at beginning of each part

Updated to:
- Track all sent messages and save first two words and full text
- Keep messages for at least 72 hours (cleanup only removes older than 72h)
- No maximum limit on tracked messages
- Partial match detection and reporting
- Duplicate file repost confirmation (same name & size within 72h)
- Robust deletion flow with inline buttons and per-file-instance deletion (file_id)
- Deletion avoids deleting bot-sent previews/messages less than 72 hours old
- Deletion UI: "Click Cancel" inline button; selection when multiple same filename instances exist
"""

import os
import json
import asyncio
import logging
import re
import uuid
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Deque, Set, Tuple, Any
from collections import deque, defaultdict
from pathlib import Path
import csv
import io

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from aiohttp import web
import aiohttp

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
PORT = int(os.getenv('PORT', 10000))
MAX_QUEUE_SIZE = 5
TELEGRAM_MESSAGE_LIMIT = 4050
CONTENT_LIMIT_FOR_INTERVALS = 60000
SEND_INTERVAL = 3.5
MESSAGE_DELAY = 0.5  # Delay between messages in seconds
QUEUE_INTERVAL = 2 * 60  # 2-minute interval between files in seconds
MAX_PROCESSING_TIME = 30 * 60  # 30 minutes timeout for file processing in seconds

ALLOWED_USERS_STR = os.getenv('ALLOWED_USERS', '6389552329')
ALLOWED_IDS = set()
for id_str in ALLOWED_USERS_STR.split(','):
    id_str = id_str.strip()
    if id_str.lstrip('-').isdigit():
        ALLOWED_IDS.add(int(id_str))

# ==================== ADMIN USERS ====================
ADMIN_USERS_STR = os.getenv('ADMIN_USERS', '6389552329')
ADMIN_IDS = set()
for id_str in ADMIN_USERS_STR.split(','):
    id_str = id_str.strip()
    if id_str.lstrip('-').isdigit():
        ADMIN_IDS.add(int(id_str))

ADMIN_USER_ID = 6389552329

UTC_PLUS_1 = timezone(timedelta(hours=1))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# file_history: chat_id -> list of entries (each entry includes file_id, filename, size, timestamp, status, etc.)
file_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

# Mapping by file_id (unique per uploaded file instance)
file_message_mapping: Dict[str, List[int]] = {}            # file_id -> list of content message ids
file_notification_mapping: Dict[str, int] = {}            # file_id -> acceptance/queued notification message id
file_other_notifications: Dict[str, List[int]] = {}      # file_id -> list of other notifications

# Pending confirmations for duplicate reposts: confirmation_id -> {file_info}
pending_confirmations: Dict[str, Dict[str, Any]] = {}

# ==================== ADMIN PREVIEW SYSTEM ====================
class MessageEntry:
    """Stores full message text, first two words and timestamp for tracking."""
    def __init__(self, chat_id: int, message_id: int, full_text: str, first_two_words: str, timestamp: datetime):
        self.chat_id = chat_id
        self.message_id = message_id
        self.full_text = full_text
        self.first_two_words = first_two_words
        self.timestamp = timestamp
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"MessageEntry(chat={self.chat_id}, id={self.message_id}, words='{self.first_two_words}', time={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"

# Global storage for message tracking (no maximum)
message_tracking: List[MessageEntry] = []
admin_preview_mode: Set[int] = set()  # track which admins are in preview mode

# ==================== ADMIN FUNCTIONS ====================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def check_admin_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is admin"""
    user_id = update.effective_user.id
    
    if is_admin(user_id):
        return True
    
    await send_bot_message(
        chat_id=update.effective_chat.id,
        context=context,
        message="⛔ **Admin Access Required**\n\nThis command is for administrators only.",
        message_thread_id=update.effective_message.message_thread_id
    )
    return False

# ==================== MESSAGE TRACKING FUNCTIONS ====================
def extract_first_two_words(text: str) -> str:
    """Extract first two words from text, handling markdown and special chars"""
    if not text:
        return ""
    clean_text = text
    # Remove URLs
    clean_text = re.sub(r'https?://\S+', ' ', clean_text)
    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    words = clean_text.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    elif len(words) == 1:
        return words[0]
    else:
        return ""

def track_message(chat_id: int, message_id: int, message_text: str):
    """Track full message text and first two words; keep indefinitely (cleanup will remove older than 72h)"""
    try:
        if not message_text or len(message_text.strip()) == 0:
            return
        
        first_two_words = extract_first_two_words(message_text)
        entry = MessageEntry(
            chat_id=chat_id,
            message_id=message_id,
            full_text=message_text,
            first_two_words=first_two_words,
            timestamp=datetime.now(UTC_PLUS_1)
        )
        message_tracking.append(entry)
        
        logger.debug(f"Tracked message: {entry}")
    except Exception as e:
        logger.error(f"Error tracking message: {e}")

def cleanup_old_messages():
    """Remove messages older than 72 hours"""
    global message_tracking
    
    if not message_tracking:
        return
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_count = len(message_tracking)
    
    message_tracking = [entry for entry in message_tracking if entry.timestamp >= cutoff]
    
    removed = initial_count - len(message_tracking)
    if removed > 0:
        logger.info(f"Cleaned up {removed} expired message entries (older than 72h)")
        logger.info(f"Current database: {len(message_tracking)} active entries")

def get_tracking_stats() -> Dict[str, any]:
    """Get statistics about tracked messages"""
    cleanup_old_messages()
    
    if not message_tracking:
        return {
            'total': 0,
            'oldest': None,
            'newest': None,
            'unique_words': 0
        }
    
    oldest = min(entry.timestamp for entry in message_tracking)
    newest = max(entry.timestamp for entry in message_tracking)
    unique_words = len(set(entry.first_two_words for entry in message_tracking))
    
    return {
        'total': len(message_tracking),
        'oldest': oldest,
        'newest': newest,
        'unique_words': unique_words
    }

# Helper: find message timestamp by chat+message_id
def get_message_timestamp(chat_id: int, message_id: int) -> Optional[datetime]:
    for entry in message_tracking:
        if entry.chat_id == chat_id and entry.message_id == message_id:
            return entry.timestamp
    return None

# ==================== PREVIEW PROCESSING FUNCTIONS ====================
def extract_preview_sections(text: str) -> List[str]:
    """Extract all 'Preview:' sections from report text - exact matching"""
    preview_sections = []
    
    pattern = r'📝\s*[Pp]review:\s*(.+?)(?=\n|\r|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        preview_text = match.strip()
        if preview_text:
            preview_sections.append(preview_text)
    
    # If no matches with emoji, try without emoji
    if not preview_sections:
        pattern2 = r'[Pp]review:\s*(.+?)(?=\n|\r|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for match in matches2:
            preview_text = match.strip()
            if preview_text:
                preview_sections.append(preview_text)
    
    return preview_sections

def find_partial_matches_for_preview(preview: str) -> List[Dict[str, Any]]:
    """
    For a single preview string, look for partial matches across saved messages.
    Returns a list of dicts with details for each message that had partial matches:
      {
        'file_message_entry': MessageEntry,
        'matched_words': [...],
        'positions': [...],
      }
    """
    results = []
    preview_words = re.sub(r'\s+', ' ', preview.strip()).split()
    if not preview_words:
        return results
    
    # Go through all tracked messages and look for substring matches
    for entry in message_tracking:
        # split original message into words
        original_words = re.sub(r'\s+', ' ', entry.full_text.strip()).split()
        matched_words = []
        positions = []
        for pw in preview_words:
            pw_norm = pw.strip()
            if not pw_norm:
                continue
            # search across original words for substring match (case-insensitive)
            for idx, ow in enumerate(original_words):
                if pw_norm.lower() in ow.lower():
                    # record matched word and its 1-based position (use first occurrence in this entry)
                    matched_words.append(pw_norm)
                    positions.append(idx + 1)
                    break  # move to next preview word
        if matched_words:
            results.append({
                'entry': entry,
                'matched_words': matched_words,
                'positions': positions
            })
    return results

def check_preview_against_database(preview_texts: List[str]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Check preview texts against tracked messages
    Returns: (full_matches, partial_matches, non_matches)
      - full_matches: list of preview strings that exactly match first two words of some saved message
      - partial_matches: list of dicts with keys:
            'preview': original preview string,
            'matched_words': ['w1','w2',...],
            'positions': ['pos1','pos2',...]
      - non_matches: list of preview strings not found at all
    """
    cleanup_old_messages()
    if not message_tracking:
        return ([], [], preview_texts.copy())
    
    # Build set of tracked first_two_words for full match detection
    tracked_words = {entry.first_two_words for entry in message_tracking if entry.first_two_words}
    
    full_matches = []
    partial_matches = []
    non_matches = []
    
    for preview in preview_texts:
        # exact first-two-words check for full match
        preview_first_two = extract_first_two_words(preview)
        if preview_first_two and preview_first_two in tracked_words:
            full_matches.append(preview)
            continue
        
        # partial matches: check any preview word occurring (substring) in any saved message
        partial_results = find_partial_matches_for_preview(preview)
        if partial_results:
            # Aggregate across entries: combine matched words and positions (use the first matching entry for reporting)
            # For user clarity, if multiple entries match, list all matches from the first matching saved message
            # (This can be extended to include multiple saved messages if needed)
            entry_info = partial_results[0]
            pm = {
                'preview': preview,
                'matched_words': entry_info['matched_words'],
                'positions': entry_info['positions']
            }
            partial_matches.append(pm)
            continue
        
        non_matches.append(preview)
    
    return (full_matches, partial_matches, non_matches)

def format_preview_report_exact(full_matches: List[str], partial_matches: List[Dict[str, Any]], non_matches: List[str], 
                               total_previews: int) -> str:
    """Format the preview report with Matches, Partial matches, and Not found"""
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Preview Analysis Report**")
    report_lines.append("─" * 35)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    match_count = len(full_matches)
    partial_count = len(partial_matches)
    non_count = len(non_matches)
    
    match_percent = (match_count / total_previews * 100) if total_previews > 0 else 0
    partial_percent = (partial_count / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (non_count / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Full matches found: {match_count} ({match_percent:.1f}%)")
    report_lines.append(f"• Partial matches found: {partial_count} ({partial_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {non_count} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if full_matches:
        report_lines.append("✅ **Matches found in database:**")
        for i, match in enumerate(full_matches, 1):
            report_lines.append(f"{i}. {match}")
    
    if partial_matches:
        if full_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Partial matches found:**")
        for i, pm in enumerate(partial_matches, 1):
            preview = pm.get('preview', '')
            matched_words = ", ".join(pm.get('matched_words', []))
            positions = ", ".join(str(p) for p in pm.get('positions', []))
            report_lines.append(f"{i}. Preview: {preview}")
            report_lines.append(f"   Matched words: {matched_words}")
            report_lines.append(f"   Word position: {positions}")
    
    if non_matches:
        if full_matches or partial_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Not found in database:**")
        for i, non_match in enumerate(non_matches, 1):
            report_lines.append(f"{i}. {non_match}")

    report_lines.append("")
    report_lines.append("Use /cancelpreview to cancel.")
                                   
    return "\n".join(report_lines)

# ==================== ADMIN COMMANDS ====================
async def adminpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to start preview mode"""
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    admin_preview_mode.add(user_id)
    
    await send_bot_message(
        chat_id=update.effective_chat.id,
        context=context,
        message="🔍 **Admin Preview Mode Activated**\n\n"
             "Please send the report data with 'Preview:' sections.\n"
             "Example format:\n"
             "```\n"
             "Preview: 97699115546 97699115547\n"
             "Preview: 237620819778 237620819780\n"
             "```\n\n"
             "Use /cancelpreview to cancel.",
        message_thread_id=update.effective_message.message_thread_id,
        parse_mode='Markdown'
    )

async def cancelpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to cancel preview mode"""
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    
    if user_id not in admin_preview_mode:
        await send_bot_message(
            chat_id=update.effective_chat.id,
            context=context,
            message="ℹ️ **Not in preview mode**\n"
                 "Use /adminpreview to start preview mode.",
            message_thread_id=update.effective_message.message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    admin_preview_mode.remove(user_id)
    
    await send_bot_message(
        chat_id=update.effective_chat.id,
        context=context,
        message="🚫 **Preview Mode Cancelled**\n\n"
             "Preview mode has been deactivated.\n\n"
                 "Use /adminpreview to start preview mode again.",
        message_thread_id=update.effective_message.message_thread_id,
        parse_mode='Markdown'
    )

async def adminstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to show tracking statistics"""
    if not await check_admin_authorization(update, context):
        return
    
    stats = get_tracking_stats()
    
    if stats['total'] == 0:
        await send_bot_message(
            chat_id=update.effective_chat.id,
            context=context,
            message="📊 **Message Tracking Statistics**\n\n"
                 "No messages tracked yet.\n"
                 "Database is empty.",
            message_thread_id=update.effective_message.message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    oldest_str = stats['oldest'].strftime('%Y-%m-%d %H:%M:%S')
    newest_str = stats['newest'].strftime('%Y-%m-%d %H:%M:%S')
    
    if stats['oldest']:
        expires_in = stats['oldest'] + timedelta(hours=72) - datetime.now(UTC_PLUS_1)
        expires_hours = max(0, expires_in.total_seconds() / 3600)
    else:
        expires_hours = 0
    
    message = f"📊 **Message Tracking Statistics**\n\n"
    message += f"**Total tracked messages:** {stats['total']}\n"
    message += f"**Unique word pairs:** {stats['unique_words']}\n"
    message += f"**Oldest entry:** {oldest_str}\n"
    message += f"**Newest entry:** {newest_str}\n"
    message += f"**Expires in:** {expires_hours:.1f}h\n"
    message += f"**Active preview sessions:** {len(admin_preview_mode)}\n\n"
    message += f"Messages auto-delete after 72 hours."
    
    await send_bot_message(
        chat_id=update.effective_chat.id,
        context=context,
        message=message,
        message_thread_id=update.effective_message.message_thread_id,
        parse_mode='Markdown'
    )

async def handle_admin_preview_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin messages when in preview mode"""
    user_id = update.effective_user.id
    
    if user_id not in admin_preview_mode:
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    
    message_text = update.message.text
    if not message_text or not message_text.strip():
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="⚠️ **Empty message**\n"
                 "Please send a message with preview content to check.\n\n"
                 "Use /cancelpreview to cancel.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    previews = extract_preview_sections(message_text)
    
    if not previews:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="❌ **No preview content found**\n\n"
                 "I couldn't find any 'Preview:' sections in your message.\n"
                 "Please use the exact format:\n"
                 "```\n"
                 "📝 Preview: [content]\n"
                 "```\n\n"
             "Use /cancelpreview to cancel.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    full_matches, partial_matches, non_matches = check_preview_against_database(previews)
    total_previews = len(previews)
    report_text = format_preview_report_exact(full_matches, partial_matches, non_matches, total_previews)
    
    await send_bot_message(
        chat_id=chat_id,
        context=context,
        message=report_text,
        message_thread_id=message_thread_id,
        parse_mode='Markdown'
    )

# ==================== PERIODIC CLEANUP TASK ====================
async def periodic_cleanup_task():
    """Periodically clean up old messages and stuck tasks"""
    while True:
        try:
            cleanup_old_messages()
            cleanup_stuck_tasks()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

# ==================== FILE HISTORY & QUEUE ====================
def update_file_history(chat_id: int, filename: str, status: str, parts_count: int = 0, messages_count: int = 0, file_id: Optional[str] = None, size: Optional[int] = None):
    """
    Add or update a history entry.
    - If file_id provided, update any existing entry with same file_id (remove and re-add) to keep latest timestamp/status.
    - If no file_id provided, append a new entry (do not remove other entries with same filename).
    """
    if file_id:
        # remove any existing entries with same file_id
        file_history[chat_id] = [entry for entry in file_history.get(chat_id, []) if entry.get('file_id') != file_id]
    
    entry = {
        'file_id': file_id or f"none-{uuid.uuid4().hex}",
        'filename': filename,
        'timestamp': datetime.now(UTC_PLUS_1),
        'status': status,
        'parts_count': parts_count,
        'messages_count': messages_count,
        'size': size or 0
    }
    file_history[chat_id].append(entry)
    
    if len(file_history[chat_id]) > 500:
        file_history[chat_id] = file_history[chat_id][-500:]

def is_authorized(user_id: int, chat_id: int) -> bool:
    return user_id in ALLOWED_IDS or chat_id in ALLOWED_IDS

async def check_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if is_authorized(user_id, chat_id):
        return True
    
    message_thread_id = update.effective_message.message_thread_id
    
    await send_bot_message(
        chat_id=chat_id,
        context=context,
        message="⛔ **Access Denied**\n\nYou are not authorized to use this bot.",
        message_thread_id=message_thread_id
    )
    
    try:
        is_group = chat_id < 0
        entity_type = "Group" if is_group else "User"
        
        await send_bot_message(
            chat_id=ADMIN_USER_ID,
            context=context,
            message=f"⚠️ **Unauthorized Access Attempt**\n\n"
                 f"**Entity Type:** {entity_type}\n"
                 f"**Chat ID:** `{chat_id}`\n"
                 f"**User ID:** `{user_id}`\n"
                 f"**Username:** @{update.effective_user.username if update.effective_user.username else 'N/A'}\n"
                 f"**Full Name:** {update.effective_user.full_name}\n"
                 f"**Time:** {datetime.now(UTC_PLUS_1).strftime('%Y-%m-%d %H:%M:%S')}",
            message_thread_id=None,
            parse_mode='Markdown'
        )
        logger.warning(f"Unauthorized access attempt - Chat: {chat_id}, User: {user_id}")
    except Exception as e:
        logger.error(f"Failed to notify admin about unauthorized access: {e}")
    
    return False

class UserState:
    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.operation = 'all'
        self.queue: Deque = deque(maxlen=MAX_QUEUE_SIZE)
        self.processing = False
        self.current_parts: List[str] = []
        self.current_index = 0
        self.last_send = None
        self.processing_task: Optional[asyncio.Task] = None
        self.cancel_requested = False
        self.waiting_for_filename = False
        self.last_deleted_file = None
        self.processing_start_time = None
        
        self.paused = False
        self.paused_at = None
        self.paused_progress = None
        
    def pause(self):
        self.paused = True
        self.paused_at = datetime.now(UTC_PLUS_1)
        
    def resume(self):
        self.paused = False
        self.paused_at = None
        self.paused_progress = None
        
    def skip(self):
        self.cancel_current_task()
        if self.queue:
            self.queue.popleft()
        
    def cancel_current_task(self):
        self.cancel_requested = True
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
        self.processing = False
        self.current_parts = []
        self.current_index = 0
        self.processing_start_time = None
        
    def clear_queue(self):
        count = len(self.queue)
        self.queue.clear()
        self.cancel_requested = False
        self.processing_start_time = None
        return count
        
    def has_active_tasks(self) -> bool:
        return self.processing or len(self.queue) > 0
    
    def remove_task_by_file_id(self, file_id: str):
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
            if task.get('file_id') != file_id:
                new_queue.append(task)
        self.queue = new_queue

user_states: Dict[int, UserState] = {}

# ==================== THREAD SAFETY LOCKS ====================
user_state_locks: Dict[int, asyncio.Lock] = {}
queue_locks: Dict[int, asyncio.Lock] = {}

def get_user_state(chat_id: int) -> UserState:
    if chat_id not in user_states:
        user_states[chat_id] = UserState(chat_id)
    return user_states[chat_id]

async def get_user_state_safe(chat_id: int) -> UserState:
    """Thread-safe access to user state"""
    async with user_state_locks.setdefault(chat_id, asyncio.Lock()):
        return get_user_state(chat_id)

async def add_to_queue_safe(state: UserState, file_info: dict) -> int:
    """Thread-safe queue addition with proper position calculation"""
    async with queue_locks.setdefault(state.chat_id, asyncio.Lock()):
        state.queue.append(file_info)
        return len(state.queue)

def get_queue_position_safe(state: UserState) -> int:
    """Get accurate queue position accounting for current processing"""
    if not state.queue:
        return 0
    
    position = len(state.queue)
    if state.processing:
        position -= 1  # Currently processing file is not in waiting queue
    return max(0, position)

def cleanup_stuck_tasks():
    """Clean up tasks that have been processing for too long"""
    current_time = datetime.now(UTC_PLUS_1)
    for chat_id, state in list(user_states.items()):
        try:
            if state.processing and state.processing_start_time:
                if (current_time - state.processing_start_time).total_seconds() > MAX_PROCESSING_TIME:
                    logger.warning(f"Cleaning up stuck task for chat {chat_id} after {MAX_PROCESSING_TIME} seconds")
                    state.cancel_current_task()
                    update_file_history(chat_id, "Unknown", "timeout_cancelled")
        except Exception as e:
            logger.error(f"Error cleaning up stuck task for chat {chat_id}: {e}")

def is_supported_file(filename: str) -> bool:
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in ['.txt', '.csv']

def extract_numeric_with_spacing(content: str) -> str:
    spaced = re.sub(r'[^0-9]', ' ', content)
    normalized = re.sub(r'\s+', ' ', spaced).strip()
    return normalized

def extract_alphabet_with_spacing(content: str) -> str:
    spaced = re.sub(r'[^a-zA-Z ]', ' ', content)
    normalized = re.sub(r'\s+', ' ', spaced).strip()
    return normalized

def process_content(content: str, operation: str) -> str:
    if operation == 'number':
        return extract_numeric_with_spacing(content)
    elif operation == 'alphabet':
        return extract_alphabet_with_spacing(content)
    return content

def split_into_telegram_chunks_without_cutting_words(content: str, max_chunk_size: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    if not content:
        return ["[Empty content]"]
    
    if len(content) <= max_chunk_size:
        return [content]
    
    words = []
    current_word = ""
    
    for char in content:
        if char.isspace():
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(char)
        else:
            current_word += char
    
    if current_word:
        words.append(current_word)
    
    chunks = []
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) <= max_chunk_size:
            current_chunk += word
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(word) > max_chunk_size:
                for i in range(0, len(word), max_chunk_size):
                    chunks.append(word[i:i + max_chunk_size])
                current_chunk = ""
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_large_content(content: str, part_size: int = CONTENT_LIMIT_FOR_INTERVALS) -> List[str]:
    if not content:
        return []
    
    if len(content) <= part_size:
        return [content]
    
    parts = []
    current_part = ""
    current_size = 0
    
    lines = content.split('\n')
    
    for line in lines:
        if current_size + len(line) + 1 <= part_size:
            if current_part:
                current_part += '\n' + line
                current_size += len(line) + 1
            else:
                current_part = line
                current_size = len(line)
        else:
            if current_part:
                parts.append(current_part)
            
            if len(line) > part_size:
                words = line.split(' ')
                current_part = ""
                current_size = 0
                
                for word in words:
                    if current_size + len(word) + 1 <= part_size:
                        if current_part:
                            current_part += ' ' + word
                            current_size += len(word) + 1
                        else:
                            current_part = word
                            current_size = len(word)
                    else:
                        if current_part:
                            parts.append(current_part)
                        current_part = word
                        current_size = len(word)
            else:
                current_part = line
                current_size = len(line)
    
    if current_part:
        parts.append(current_part)
    
    return parts

def process_csv_file(file_bytes: bytes, operation: str) -> str:
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        csv_file = io.StringIO(text)
        reader = csv.reader(csv_file)
        
        lines = []
        for row in reader:
            lines.append(' '.join(str(cell) for cell in row))
        
        content = '\n'.join(lines)
        return process_content(content, operation)
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        text = file_bytes.decode('utf-8', errors='ignore')
        return process_content(text, operation)

# -------------------- Centralized send & track helper --------------------
async def send_telegram_message_safe(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                    message: str, message_thread_id: Optional[int] = None, 
                                    retries: int = 5, file_id: Optional[str] = None,  # file_id instead of filename
                                    notification_type: str = 'content',
                                    parse_mode: str = 'Markdown',
                                    disable_notification: bool = True) -> Optional[Any]:
    """
    Send a message and track it. Returns the sent message object or None on failure.
    - file_id: if provided, maps message id to file-specific mappings
    - notification_type: 'content' | 'notification' | 'other'
    """
    for attempt in range(retries):
        try:
            text_to_send = message
            if len(text_to_send) > TELEGRAM_MESSAGE_LIMIT:
                text_to_send = text_to_send[:TELEGRAM_MESSAGE_LIMIT]
            
            sent_message = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=text_to_send,
                disable_notification=disable_notification,
                parse_mode=parse_mode
            )
            
            # Track message for admin preview system
            if sent_message and getattr(sent_message, 'text', None):
                track_message(chat_id, sent_message.message_id, sent_message.text)
            
            # Track mappings per file_id if provided
            if file_id and sent_message:
                if notification_type == 'content':
                    if file_id not in file_message_mapping:
                        file_message_mapping[file_id] = []
                    file_message_mapping[file_id].append(sent_message.message_id)
                elif notification_type == 'notification':
                    file_notification_mapping[file_id] = sent_message.message_id
                elif notification_type == 'other':
                    if file_id not in file_other_notifications:
                        file_other_notifications[file_id] = []
                    file_other_notifications[file_id].append(sent_message.message_id)
            
            return sent_message
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for chat {chat_id}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return None

# Convenience wrapper for non-file messages
async def send_bot_message(chat_id: int, context: ContextTypes.DEFAULT_TYPE, message: str, message_thread_id: Optional[int] = None, parse_mode: str = 'Markdown'):
    return await send_telegram_message_safe(chat_id, context, message, message_thread_id, retries=1, file_id=None, notification_type='other', parse_mode=parse_mode, disable_notification=True)

# -------------------- other helpers --------------------
async def track_other_notification(chat_id: int, file_id: str, message_id: int):
    if file_id not in file_other_notifications:
        file_other_notifications[file_id] = []
    file_other_notifications[file_id].append(message_id)

async def send_chunks_immediately(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                chunks: List[str], file_id: str, filename: str, message_thread_id: Optional[int] = None) -> bool:
    try:
        total_messages_sent = 0
        
        for i, chunk in enumerate(chunks, 1):
            sent = await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, file_id=file_id, notification_type='content')
            if sent:
                total_messages_sent += 1
                
                # Delay between messages
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for `{filename}` (file_id={file_id})")
                return False  # Stop on failure
        
        state = await get_user_state_safe(chat_id)
        state.last_send = datetime.now(UTC_PLUS_1)
        
        if total_messages_sent > 0:
            completion_msg = await send_telegram_message_safe(
                chat_id=chat_id,
                context=context,
                message=f"✅ Completed: `{filename}`\n📊 Sent {total_messages_sent} message{'s' if total_messages_sent > 1 else ''}",
                message_thread_id=message_thread_id,
                file_id=file_id,
                notification_type='other'
            )
            # track_other_notification already done by send_telegram_message_safe for notification_type=other
            update_file_history(chat_id, filename, 'completed', messages_count=total_messages_sent, file_id=file_id, size=len(''.join(chunks)))
            return True
        else:
            logger.error(f"No chunks sent for `{filename}`")
            return False
        
    except Exception as e:
        logger.error(f"Error in send_chunks_immediately: {e}")
        return False

async def send_large_content_part(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                part: str, part_num: int, total_parts: int, 
                                file_id: str, filename: str, message_thread_id: Optional[int] = None) -> int:
    try:
        chunks = split_into_telegram_chunks_without_cutting_words(part, TELEGRAM_MESSAGE_LIMIT)
        total_messages_in_part = 0
        
        if total_parts > 1:
            part_header_msg = await send_telegram_message_safe(
                chat_id=chat_id,
                context=context,
                message=f"📄 `{filename}` - Part {part_num}/{total_parts}",
                message_thread_id=message_thread_id,
                file_id=file_id,
                notification_type='other'
            )
            total_messages_in_part += 1
        
        for i, chunk in enumerate(chunks, 1):
            sent = await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, file_id=file_id, notification_type='content')
            if sent:
                total_messages_in_part += 1
                
                # Delay between messages
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for part {part_num} of `{filename}` (file_id={file_id})")
                return 0  # Return 0 on failure
        
        return total_messages_in_part  # Return actual count
        
    except Exception as e:
        logger.error(f"Error sending large content part: {e}")
        return 0  # Return 0 on error

async def send_with_intervals(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                            parts: List[str], filename: str, state: UserState, 
                            file_id: str, message_thread_id: Optional[int] = None) -> bool:
    try:
        total_parts = len(parts)
        total_messages_sent = 0  # Initialize total messages counter
        
        for i, part in enumerate(parts, 1):
            if state.cancel_requested:
                return False
            
            while state.paused and not state.cancel_requested:
                await asyncio.sleep(1)
            
            if state.cancel_requested:
                return False
            
            state.current_index = i
            state.current_parts = parts
            
            messages_in_part = await send_large_content_part(
                chat_id, context, part, i, total_parts, file_id, filename, message_thread_id
            )
            
            if not messages_in_part:  # Check for 0 (failure) instead of <= 0
                logger.error(f"Failed to send part {i} of `{filename}` (file_id={file_id})")
                return False
            
            total_messages_sent += messages_in_part  # Accumulate actual messages
            
            state.last_send = datetime.now(UTC_PLUS_1)
            
            if i < total_parts:
                await asyncio.sleep(SEND_INTERVAL * 60)
        
        # Use actual messages count, not parts count
        completion_msg = await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"✅ Completed: `{filename}`\n📊 Sent {total_parts} part{'s' if total_parts > 1 else ''} ({total_messages_sent} messages total)",
            message_thread_id=message_thread_id,
            file_id=file_id,
            notification_type='other'
        )
        
        update_file_history(chat_id, filename, 'completed', parts_count=total_parts, messages_count=total_messages_sent, file_id=file_id, size=len(''.join(parts)))
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def cleanup_completed_file(file_id: str, chat_id: int):
    # Remove mappings for a completed file_id
    if file_id in file_message_mapping:
        del file_message_mapping[file_id]
    if file_id in file_notification_mapping:
        del file_notification_mapping[file_id]
    if file_id in file_other_notifications:
        del file_other_notifications[file_id]
    
    logger.info(f"Cleaned up tracking for completed file: {file_id} in chat {chat_id}")

async def process_queue(chat_id: int, context: ContextTypes.DEFAULT_TYPE, message_thread_id: Optional[int] = None):
    state = await get_user_state_safe(chat_id)
    
    if state.processing:
        return
    
    state.processing = True
    state.cancel_requested = False
    state.paused = False
    state.processing_start_time = datetime.now(UTC_PLUS_1)
    
    try:
        while state.queue and not state.cancel_requested:
            # Check for processing timeout
            if (datetime.now(UTC_PLUS_1) - state.processing_start_time).total_seconds() > MAX_PROCESSING_TIME:
                logger.warning(f"Processing timeout for chat {chat_id}")
                state.cancel_current_task()
                break
            
            while state.paused and not state.cancel_requested:
                await asyncio.sleep(1)
            
            if state.cancel_requested:
                break
                
            file_info = state.queue[0]
            filename = file_info['name']
            file_id = file_info.get('file_id')
            file_message_thread_id = file_info.get('message_thread_id', message_thread_id)
            
            if file_info.get('requires_intervals', False):
                update_file_history(chat_id, filename, 'running', parts_count=len(file_info['parts']), file_id=file_id, size=file_info.get('size'))
            else:
                update_file_history(chat_id, filename, 'running', messages_count=len(file_info['chunks']), file_id=file_id, size=file_info.get('size'))
            
            if len(state.queue) > 0 and state.queue[0] == file_info:
                if file_info.get('requires_intervals', False):
                    sending_msg = await send_telegram_message_safe(
                        chat_id=chat_id,
                        context=context,
                        message=f"📤 Sending: `{filename}`\n"
                             f"📊 Total parts: {len(file_info['parts'])}\n"
                             f"⏰ Interval: {SEND_INTERVAL} minutes between parts",
                        message_thread_id=file_message_thread_id,
                        file_id=file_id,
                        notification_type='other'
                    )
                else:
                    sending_msg = await send_telegram_message_safe(
                        chat_id=chat_id,
                        context=context,
                        message=f"📤 Sending: `{filename}`\n"
                             f"📊 Total messages: {len(file_info['chunks'])}",
                        message_thread_id=file_message_thread_id,
                        file_id=file_id,
                        notification_type='other'
                    )
            
            success = False
            if file_info.get('requires_intervals', False):
                success = await send_with_intervals(
                    chat_id, context, 
                    file_info['parts'], 
                    filename,
                    state,
                    file_info['file_id'],
                    file_message_thread_id
                )
            else:
                success = await send_chunks_immediately(
                    chat_id, context,
                    file_info['chunks'],
                    file_info['file_id'],
                    filename,
                    file_message_thread_id
                )
            
            # Always remove the file from queue after processing (success or failure)
            if state.queue and state.queue[0].get('file_id') == file_id:
                processed_file = state.queue.popleft()
                processed_file_id = processed_file.get('file_id')
                processed_filename = processed_file.get('name')
                
                if success and not state.cancel_requested:
                    logger.info(f"Successfully processed `{processed_filename}` (file_id={processed_file_id}) for chat {chat_id}")
                    await cleanup_completed_file(processed_file_id, chat_id)
                else:
                    logger.error(f"Failed to process `{processed_filename}` (file_id={processed_file_id}) for chat {chat_id}")
                    failed_msg = await send_telegram_message_safe(
                        chat_id=chat_id,
                        context=context,
                        message=f"❌ Failed to send: `{processed_filename}`\nPlease try uploading again.",
                        message_thread_id=file_message_thread_id,
                        file_id=processed_file_id,
                        notification_type='other'
                    )
            
            state.current_parts = []
            state.current_index = 0
            
            # Wait 2 minutes before processing next file (if any remain)
            if state.queue and not state.cancel_requested:
                next_file = state.queue[0]['name']
                next_file_msg = await send_telegram_message_safe(
                    chat_id=chat_id,
                    context=context,
                    message=f"⏰ **Queue Interval**\n\n"
                         f"Next file `{next_file}` will start in 2 minutes...",
                    file_id=None,
                    notification_type='other',
                    message_thread_id=message_thread_id
                )
                
                # Wait for QUEUE_INTERVAL seconds
                wait_start = datetime.now(UTC_PLUS_1)
                while (datetime.now(UTC_PLUS_1) - wait_start).seconds < QUEUE_INTERVAL:
                    if state.cancel_requested:
                        try:
                            await context.bot.delete_message(chat_id=chat_id, message_id=next_file_msg.message_id)
                        except Exception:
                            pass
                        break
                    await asyncio.sleep(1)
                
                if not state.cancel_requested:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=next_file_msg.message_id)
                    except Exception:
                        pass
            
            if state.cancel_requested:
                break
                
    except asyncio.CancelledError:
        logger.info(f"Queue processing cancelled for chat {chat_id}")
    except Exception as e:
        logger.error(f"Queue processing error: {e}")
        error_msg = await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"⚠️ Processing error\n{str(e)[:200]}",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )
        if state.queue:
            current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
            # If file_id available, track under it; else just skip
            if state.queue and state.queue[0].get('file_id'):
                await track_other_notification(chat_id, state.queue[0]['file_id'], error_msg.message_id)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

# ==================== COMMANDS & HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    await send_bot_message(
        chat_id=chat_id,
        context=context,
        message="🤖 **FileX Bot**\n\n"
        f"📝 **Current operation:** {state.operation.capitalize()} content\n"
        f"📊 **Queue:** {get_queue_position_safe(state)}/{MAX_QUEUE_SIZE} files\n"
        f"⏰ **Queue interval:** {QUEUE_INTERVAL // 60} minutes between files\n\n"
        "📁 **Supported files:** TXT, CSV\n\n"
        "🛠 **Operations available:**\n"
        "• All content - Send file as-is\n"
        "• Number only - Extract numeric characters (properly spaced)\n"
        "• Alphabet only - Extract alphabetic characters (properly spaced)\n\n"
        "⚙️ **Commands:**\n"
        "/operation - Change processing operation\n"
        "/status - Check current status\n"
        "/stats - Show last 12 hours stats\n"
        "/queue - View file queue\n"
        "/cancel - Cancel current task and clear queue\n"
        "/pause - Pause current task\n"
        "/resume - Resume paused task\n"
        "/skip - Skip to next task\n"
        "/delfilecontent - Delete all content from a file\n\n"
        "📤 **Upload TXT or CSV file to start!**",
        message_thread_id=message_thread_id,
        parse_mode='Markdown'
    )

async def operation_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    operation_descriptions = {
        'all': 'Send all file content as-is',
        'number': 'Extract only numbers (with proper spacing)',
        'alphabet': 'Extract only letters (with proper spacing)'
    }
    
    current_op_text = f"**Current operation:** {state.operation.capitalize()}\n{operation_descriptions[state.operation]}\n"
    
    keyboard = [
        [
            InlineKeyboardButton("✅ All content", callback_data="all"),
            InlineKeyboardButton("🔢 Number only", callback_data="number"),
        ],
        [
            InlineKeyboardButton("🔤 Alphabet only", callback_data="alphabet")
        ]
    ]
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message=f"{current_op_text}\n"
        "🔧 **Select new operation:**\n\n"
        "This will be remembered for all future file uploads.",
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other',
        parse_mode='Markdown'
    )
    # Attach keyboard in a separate message (to keep centralized sending simple)
    await context.bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=" ", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    # Authorization check
    if not is_authorized(user_id, chat_id):
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="⛔ **Access Denied**\n\nYou are not authorized to use this bot.",
            message_thread_id=query.message.message_thread_id
        )
        return
    
    data = query.data or ""
    state = await get_user_state_safe(chat_id)
    
    # Operation buttons
    if data in ('all', 'number', 'alphabet'):
        operation = data
        operation_names = {
            'all': '✅ All content',
            'number': '🔢 Number only',
            'alphabet': '🔤 Alphabet only'
        }
        state.operation = operation
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"✅ **Operation updated:** {operation_names[operation]}\n\n"
            "All future files will be processed with this operation.\n\n"
            "📤 Now upload a TXT or CSV file!",
            message_thread_id=query.message.message_thread_id,
            file_id=None,
            notification_type='other'
        )
        return
    
    # Duplicate repost confirmation callbacks: dup_confirm:<conf_id> or dup_cancel:<conf_id>
    if data.startswith("dup_confirm:"):
        conf_id = data.split(":", 1)[1]
        conf = pending_confirmations.pop(conf_id, None)
        if not conf:
            await send_bot_message(chat_id, context, "❌ Confirmation expired or invalid.", message_thread_id=query.message.message_thread_id)
            return
        # Proceed to queue the stored file_info
        state = await get_user_state_safe(chat_id)
        file_info = conf['file_info']
        queue_size = await add_to_queue_safe(state, file_info)
        queue_position = get_queue_position_safe(state)
        file_id = file_info['file_id']
        filename = file_info['name']
        content_size = file_info.get('size', 0)
        # send accepted notification and start processing if needed
        sent_msg = await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=(f"✅ File accepted: `{filename}`\n"
                     f"Size: {content_size:,} characters\n"
                     f"Operation: {state.operation.capitalize()}\n\n"
                     f"🟢 Starting Your Task"),
            message_thread_id=file_info.get('message_thread_id'),
            file_id=file_id,
            notification_type='notification'
        )
        update_file_history(chat_id, filename, 'accepted', file_id=file_id, size=content_size)
        if not state.processing:
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, file_info.get('message_thread_id')))
        return
    if data.startswith("dup_cancel:"):
        conf_id = data.split(":",1)[1]
        pending_confirmations.pop(conf_id, None)
        await send_bot_message(chat_id, context, "❌ Upload cancelled.", message_thread_id=query.message.message_thread_id)
        return
    
    # Deletion cancel while waiting for filename input: delete_input_cancel
    if data.startswith("delete_input_cancel"):
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await send_bot_message(chat_id, context, "❌ Deletion cancelled.", message_thread_id=query.message.message_thread_id)
        return
    
    # Deletion choices: delete_choice:<file_id>
    if data.startswith("delete_choice:"):
        chosen_file_id = data.split(":", 1)[1]
        # perform deletion for this file_id
        await perform_deletion_by_file_id(chat_id, chosen_file_id, context, query.message.message_thread_id)
        return
    
    # Delete cancel from selection menu
    if data.startswith("delete_select_cancel"):
        await send_bot_message(chat_id, context, "❌ Deletion cancelled.", message_thread_id=query.message.message_thread_id)
        return

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    status_lines = []
    status_lines.append("📊 **FileX Status**")
    status_lines.append("──────────────")
    status_lines.append(f"🔧 **Operation:** {state.operation.capitalize()} content")
    status_lines.append(f"📋 **Queue:** {get_queue_position_safe(state)}/{MAX_QUEUE_SIZE} files")
    
    if state.processing:
        if state.current_parts:
            remaining = len(state.current_parts) - state.current_index
            status_lines.append(f"📤 **Processing (Remaining Parts):** Yes ({remaining})")
        else:
            status_lines.append("📤 **Processing (Remaining Parts):** Yes")
        
        if state.paused:
            status_lines.append("⏸️ **Paused:** Yes")
        
        if state.queue:
            current_file = state.queue[0].get('name', 'Unknown')
            status_lines.append(f"📄 **Current file:** `{current_file}`")
    else:
        status_lines.append("📤 **Processing (Remaining Parts):** No")
    
    if state.cancel_requested:
        status_lines.append("🚫 **Cancel requested:** Yes")
    
    if state.last_send:
        if isinstance(state.last_send, datetime):
            last_send_str = state.last_send.strftime('%H:%M:%S')
            status_lines.append(f"⏱ **Last send time:** {last_send_str}")
    
    status_lines.append(f"⏰ **Queue interval:** {QUEUE_INTERVAL // 60} minutes between files")
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message="\n".join(status_lines),
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    
    twelve_hours_ago = datetime.now(UTC_PLUS_1) - timedelta(hours=12)
    
    recent_files = [
        entry for entry in file_history.get(chat_id, [])
        if entry['timestamp'] >= twelve_hours_ago
    ]
    
    if not recent_files:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="📊 **Last 12 Hours Stats**\n\n"
                 "No files processed in the last 12 hours.",
            message_thread_id=message_thread_id
        )
        return
    
    recent_files.sort(key=lambda x: x['timestamp'], reverse=True)
    
    stats_text = "📊 **Last 12 Hours Stats**\n"
    stats_text += "────────────────────\n\n"
    
    for entry in recent_files:
        time_str = entry['timestamp'].strftime('%H:%M:%S')
        status_emoji = {
            'completed': '✅',
            'skipped': '⏭️',
            'deleted': '🗑️',
            'cancelled': '🚫',
            'running': '📤',
            'paused': '⏸️',
            'timeout_cancelled': '⏱️'
        }.get(entry['status'], '📝')
        
        count_info = ""
        if entry['status'] == 'completed':
            if entry.get('parts_count', 0) > 0:
                count_info = f" ({entry['parts_count']} part{'s' if entry['parts_count'] > 1 else ''}"
                if entry.get('messages_count', 0) > 0:
                    count_info += f", {entry['messages_count']} messages"
                count_info += ")"
            elif entry.get('messages_count', 0) > 0:
                count_info = f" ({entry['messages_count']} message{'s' if entry['messages_count'] > 1 else ''})"
        
        stats_text += f"{status_emoji} `{entry['filename']}`\n"
        stats_text += f"   Status: {entry['status'].capitalize()}{count_info}\n"
        stats_text += f"   Time: {time_str}\n\n"
    
    stats_text += f"Total files: {len(recent_files)}"
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message=stats_text,
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.queue:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="📭 **Queue is empty**",
            message_thread_id=message_thread_id
        )
        return
    
    queue_text = "📋 **File Queue:**\n"
    queue_text += "──────────────\n"
    
    for i, file_info in enumerate(state.queue, 1):
        parts_info = ""
        if 'chunks' in file_info:
            parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
        elif 'parts' in file_info:
            parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
        
        if state.processing and i == 1:
            queue_text += f"▶️ **ACTIVE** - `{file_info.get('name', 'Unknown')}`\n"
        else:
            actual_position = i - 1 if state.processing else i
            queue_text += f"{actual_position}. `{file_info.get('name', 'Unknown')}`\n"
        
        queue_text += f"   📏 {file_info.get('size', 0):,} chars{parts_info}\n"
        queue_text += f"   🔧 {file_info.get('operation', state.operation).capitalize()} content\n"
    
    if len(state.queue) >= MAX_QUEUE_SIZE:
        queue_text += f"\n⚠️ **Queue is full** ({MAX_QUEUE_SIZE}/{MAX_QUEUE_SIZE} files)"
    
    queue_text += f"\n⏰ **Queue interval:** {QUEUE_INTERVAL // 60} minutes between files"
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message=queue_text,
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.processing:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **No active task to pause**\n"
                 "There's no task currently running.",
            message_thread_id=message_thread_id
        )
        return
    
    if state.paused:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **Task already paused**\n"
                 "Use /resume to continue.",
            message_thread_id=message_thread_id
        )
        return
    
    state.paused_progress = {
        'current_index': state.current_index,
        'current_parts': state.current_parts.copy() if state.current_parts else []
    }
    
    state.pause()
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    current_file_id = state.queue[0].get('file_id') if state.queue else None
    
    if state.queue:
        file_info = state.queue[0]
        if file_info.get('requires_intervals', False):
            update_file_history(chat_id, current_file, 'paused', parts_count=len(file_info['parts']), file_id=current_file_id, size=file_info.get('size'))
        else:
            update_file_history(chat_id, current_file, 'paused', messages_count=len(file_info['chunks']), file_id=current_file_id, size=file_info.get('size'))
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message="⏸️ **Task Paused**\n\n"
             f"Task `{current_file}` has been paused.\n"
             f"Progress saved at part {state.current_index} of {len(state.current_parts) if state.current_parts else 0}.\n\n"
             "Use /resume to continue where you left off.",
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.paused:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **No paused task to resume**\n"
                 "There's no task currently paused.",
            message_thread_id=message_thread_id
        )
        return
    
    if not state.processing:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="⚠️ **Cannot resume**\n"
                 "The paused task is no longer active.",
            message_thread_id=message_thread_id
        )
        state.paused = False
        return
    
    if state.paused_progress:
        state.current_index = state.paused_progress['current_index']
        state.current_parts = state.paused_progress['current_parts']
    
    state.resume()
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    current_file_id = state.queue[0].get('file_id') if state.queue else None
    
    if state.queue:
        file_info = state.queue[0]
        if file_info.get('requires_intervals', False):
            update_file_history(chat_id, current_file, 'running', parts_count=len(file_info['parts']), file_id=current_file_id, size=file_info.get('size'))
        else:
            update_file_history(chat_id, current_file, 'running', messages_count=len(file_info['chunks']), file_id=current_file_id, size=file_info.get('size'))
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message="▶️ **Task Resumed**\n\n"
             f"Resuming `{current_file}` from part {state.current_index} of {len(state.current_parts) if state.current_parts else 0}.\n"
             "Task will continue automatically.",
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.processing and not state.queue:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **No task to skip**\n"
                 "There's no task currently running or in queue.",
            message_thread_id=message_thread_id
        )
        return
    
    if not state.queue:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **Queue is empty**\n"
                 "No tasks to skip.",
            message_thread_id=message_thread_id
        )
        return
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    current_file_id = state.queue[0].get('file_id') if state.queue else None
    
    update_file_history(chat_id, current_file, 'skipped', file_id=current_file_id)
    
    state.skip()
    
    if state.queue:
        next_file = state.queue[0].get('name', 'Unknown')
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"⏭️ **Task Skipped**\n\n"
                 f"Skipped: `{current_file}`\n"
                 f"Starting next task: `{next_file}`",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )
        
        if not state.processing:
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    else:
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"⏭️ **Task Skipped**\n\n"
                 f"Skipped: `{current_file}`\n"
                 "Queue is now empty.",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.has_active_tasks():
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message="ℹ️ **No active tasks to cancel**\n"
            "No processing is currently running and the queue is empty.",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )
        return
    
    if state.queue:
        for file_info in state.queue:
            update_file_history(chat_id, file_info['name'], 'cancelled', parts_count=len(file_info.get('parts', [])) if file_info.get('requires_intervals') else 0, messages_count=len(file_info.get('chunks', [])) if not file_info.get('requires_intervals') else 0, file_id=file_info.get('file_id'), size=file_info.get('size'))
    
    state.cancel_current_task()
    cleared_count = state.clear_queue()
    
    response_lines = []
    response_lines.append("🚫 **Processing Cancelled**")
    response_lines.append("──────────────")
    response_lines.append(f"✅ Current task stopped")
    response_lines.append(f"✅ {cleared_count} file(s) removed from queue")
    
    if state.processing:
        response_lines.append(f"✅ Processing interrupted")
    
    response_lines.append("")
    response_lines.append("📤 Ready for new files")
    
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message="\n".join(response_lines),
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other'
    )

# -------------------- Deletion flow (improved UI & reliability) --------------------
async def delfilecontent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if state.waiting_for_filename:
        # We're already waiting — request again
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="⚠️ Already waiting for filename. Click Cancel to cancel the operation.",
            message_thread_id=message_thread_id
        )
        return
    
    state.waiting_for_filename = True
    state.last_deleted_file = None
    
    keyboard = [[InlineKeyboardButton("Cancel", callback_data="delete_input_cancel")]]
    await send_telegram_message_safe(
        chat_id=chat_id,
        context=context,
        message="🗑️ **Delete File Content**\n\n"
             "Please send me the filename you want to delete content from.\n"
             "Example: `Sudan WhatsApp.txt`\n\n"
             "Click Cancel to cancel this operation.",
        message_thread_id=message_thread_id,
        file_id=None,
        notification_type='other',
        parse_mode='Markdown'
    )
    # Send inline button separately (centralized sender doesn't yet support reply_markup param in all uses)
    await context.bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=" ", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_deletion_by_file_id(chat_id: int, file_id: str, context: ContextTypes.DEFAULT_TYPE, message_thread_id: Optional[int] = None):
    """
    Perform deletion of messages associated with a specific file_id.
    Will respect 72-hour rule: messages younger than 72 hours will not be deleted.
    """
    # Find history entry for this file_id
    entries = [e for e in file_history.get(chat_id, []) if e.get('file_id') == file_id]
    if not entries:
        await send_bot_message(chat_id, context, "❌ File entry not found.", message_thread_id=message_thread_id)
        return
    entry = entries[0]
    filename = entry.get('filename', 'Unknown')
    
    deleted_messages = []
    skipped_messages = []
    
    # Delete content messages
    if file_id in file_message_mapping:
        for msg_id in list(file_message_mapping[file_id]):
            ts = get_message_timestamp(chat_id, msg_id)
            if ts and (datetime.now(UTC_PLUS_1) - ts).total_seconds() >= 72 * 3600:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    deleted_messages.append(msg_id)
                    logger.info(f"Deleted content message {msg_id} for file {filename} (file_id={file_id})")
                except Exception as e:
                    logger.error(f"Failed to delete content message {msg_id}: {e}")
            else:
                skipped_messages.append(msg_id)
        # After attempting deletion, remove old ids from mapping
        remaining = [m for m in file_message_mapping.get(file_id, []) if m not in deleted_messages]
        if remaining:
            file_message_mapping[file_id] = remaining
        else:
            file_message_mapping.pop(file_id, None)
    
    # Delete other notifications
    if file_id in file_other_notifications:
        for msg_id in list(file_other_notifications[file_id]):
            ts = get_message_timestamp(chat_id, msg_id)
            if ts and (datetime.now(UTC_PLUS_1) - ts).total_seconds() >= 72 * 3600:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                    deleted_messages.append(msg_id)
                    logger.info(f"Deleted other notification message {msg_id} for file {filename} (file_id={file_id})")
                except Exception as e:
                    logger.error(f"Failed to delete other notification message {msg_id}: {e}")
            else:
                skipped_messages.append(msg_id)
        remaining = [m for m in file_other_notifications.get(file_id, []) if m not in deleted_messages]
        if remaining:
            file_other_notifications[file_id] = remaining
        else:
            file_other_notifications.pop(file_id, None)
    
    notification_edited = False
    if file_id in file_notification_mapping:
        notification_msg_id = file_notification_mapping[file_id]
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=notification_msg_id,
                text=f"🗑️ **File Content Deleted**\n\n"
                     f"File: `{filename}`\n"
                     f"Messages deleted: {len(deleted_messages)}\n"
                     f"Messages skipped (younger than 72h): {len(skipped_messages)}\n"
                     f"All deletable content from this file has been removed.",
                parse_mode='Markdown'
            )
            notification_edited = True
            logger.info(f"Edited acceptance/queued notification for {filename} (file_id={file_id})")
        except Exception as e:
            logger.error(f"Failed to edit notification message: {e}")
    
    # Update history entry status for this file_id
    update_file_history(chat_id, filename, 'deleted', file_id=file_id, size=entry.get('size', 0))
    
    # Remove mappings (notification remains updated but file_message_mapping/file_other_notifications cleaned above)
    file_notification_mapping.pop(file_id, None)
    file_other_notifications.pop(file_id, None)
    file_message_mapping.pop(file_id, None)
    
    await send_bot_message(
        chat_id=chat_id,
        context=context,
        message=f"🗑️ `{filename}` content deletion completed.\nMessages removed: {len(deleted_messages)}\nMessages skipped (younger than 72h or unknown): {len(skipped_messages)}",
        message_thread_id=message_thread_id
    )

async def process_file_deletion(chat_id: int, message_thread_id: Optional[int], filename: str, 
                                context: ContextTypes.DEFAULT_TYPE, state: UserState):
    # Cancel command support was replaced with inline button; treat 'cancel' just in case
    if not filename:
        await send_bot_message(chat_id, context, "❌ Invalid filename.", message_thread_id=message_thread_id)
        state.waiting_for_filename = False
        return
    if filename.lower() == 'cancel':
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="❌ **Operation cancelled**",
            message_thread_id=message_thread_id
        )
        return
    
    # Find all history entries for this chat with this filename
    matching_entries = [entry for entry in file_history.get(chat_id, []) if entry.get('filename') == filename]
    
    if not matching_entries:
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message=f"❌ **File not found**\n\n"
                 f"No record found for `{filename}` in your history.\n\n"
                 f"Use /delfilecontent to try again with a different filename.",
            message_thread_id=message_thread_id
        )
        return
    
    # If multiple entries exist, present options to choose which to delete
    if len(matching_entries) > 1:
        keyboard = []
        for i, entry in enumerate(matching_entries, 1):
            ts_str = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            btn_text = f"{i}. {ts_str}"
            keyboard.append([InlineKeyboardButton(btn_text, callback_data=f"delete_choice:{entry['file_id']}")])
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="delete_select_cancel")])
        
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"🗂 Multiple entries found for `{filename}`\n\nSelect which upload to delete:",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )
        # Send inline options
        await context.bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=" ", reply_markup=InlineKeyboardMarkup(keyboard))
        
        state.waiting_for_filename = False
        return
    
    # Only one matching entry: delete that instance
    entry = matching_entries[0]
    file_id = entry.get('file_id')
    state.waiting_for_filename = False
    state.last_deleted_file = filename
    
    # If this entry is currently processing in queue, cancel it
    is_currently_processing = False
    if state.queue and state.queue[0].get('file_id') == file_id and state.processing:
        is_currently_processing = True
        state.cancel_current_task()
        if state.queue:
            state.queue.popleft()
    
    # Perform deletion
    await perform_deletion_by_file_id(chat_id, file_id, context, message_thread_id)
    
    # After deletion: if the queue still has items and we canceled, resume processing
    if is_currently_processing and state.queue and not state.processing:
        next_file = state.queue[0].get('name', 'Unknown')
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message=f"🔄 **Moving to next task**\n\n"
                 f"Starting next task: `{next_file}`",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )
        state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    elif is_currently_processing and not state.queue:
        await send_telegram_message_safe(
            chat_id=chat_id,
            context=context,
            message="🏁 **Processing stopped**\n\n"
                 "No more tasks in queue.",
            message_thread_id=message_thread_id,
            file_id=None,
            notification_type='other'
        )

# -------------------- Message and file handlers --------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    state = await get_user_state_safe(chat_id)
    
    # Check if waiting for filename (deletion flow) - MUST CHECK THIS FIRST
    if state.waiting_for_filename:
        message_thread_id = update.effective_message.message_thread_id
        await process_file_deletion(chat_id, message_thread_id, update.message.text.strip(), context, state)
        return
    
    # Check if admin is in preview mode (AFTER checking deletion flow)
    user_id = update.effective_user.id
    if user_id in admin_preview_mode:
        await handle_admin_preview_message(update, context)
        return

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    state.waiting_for_filename = False
    state.last_deleted_file = None
    
    state.cancel_requested = False
    
    if not update.message.document:
        return
    
    doc = update.message.document
    file_name = doc.file_name
    
    if not file_name:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message="❌ **Invalid file**\nPlease upload a valid TXT or CSV file.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    if not is_supported_file(file_name):
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message=f"❌ **Unsupported file type**\n"
            f"Please upload only TXT or CSV files.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    # Check queue size thread-safely
    queue_position = get_queue_position_safe(state)
    if queue_position >= MAX_QUEUE_SIZE:
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message=f"❌ **Queue is full!**\n"
            f"Maximum {MAX_QUEUE_SIZE} files allowed.\n"
            "Please wait for current files to be processed.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
        return
    
    try:
        # Download file with timeout
        file = await context.bot.get_file(doc.file_id)
        try:
            file_bytes = await asyncio.wait_for(file.download_as_bytearray(), timeout=30.0)
        except asyncio.TimeoutError:
            await send_bot_message(
                chat_id=chat_id,
                context=context,
                message=f"❌ **Download timeout**\n"
                     f"File `{file_name}` is too large or download failed.\n"
                     f"Please try again with a smaller file.",
                message_thread_id=message_thread_id,
                parse_mode='Markdown'
            )
            return
        except Exception as e:
            logger.error(f"File download error: {e}")
            await send_bot_message(
                chat_id=chat_id,
                context=context,
                message=f"❌ **Download failed**\n"
                     f"Could not download file `{file_name}`.\n"
                     f"Please try again.",
                message_thread_id=message_thread_id,
                parse_mode='Markdown'
            )
            return
        
        ext = Path(file_name).suffix.lower()
        
        if ext == '.csv':
            content = process_csv_file(file_bytes, state.operation)
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
            content = process_content(text, state.operation)
        
        content_size = len(content)
        
        # Create unique file_id for this upload
        file_id = f"{uuid.uuid4().hex}"
        
        if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
            chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
            file_info = {
                'file_id': file_id,
                'name': file_name,
                'content': content,
                'chunks': chunks,
                'size': content_size,
                'operation': state.operation,
                'requires_intervals': False,
                'message_thread_id': message_thread_id
            }
        else:
            parts = split_large_content(content, CONTENT_LIMIT_FOR_INTERVALS)
            file_info = {
                'file_id': file_id,
                'name': file_name,
                'content': content,
                'parts': parts,
                'size': content_size,
                'operation': state.operation,
                'requires_intervals': True,
                'message_thread_id': message_thread_id
            }
        
        # Duplicate detection: same filename & size within 72 hours
        now = datetime.now(UTC_PLUS_1)
        duplicate_found = False
        for entry in file_history.get(chat_id, []):
            if entry.get('filename') == file_name and entry.get('size') == content_size:
                if (now - entry.get('timestamp')) <= timedelta(hours=72):
                    duplicate_found = True
                    break
        
        if duplicate_found:
            # Ask for confirmation
            conf_id = uuid.uuid4().hex
            pending_confirmations[conf_id] = {'file_info': file_info}
            keyboard = [
                [InlineKeyboardButton("Yes, Post", callback_data=f"dup_confirm:{conf_id}"),
                 InlineKeyboardButton("No, Don't", callback_data=f"dup_cancel:{conf_id}")]
            ]
            await send_telegram_message_safe(
                chat_id=chat_id,
                context=context,
                message=f"⚠️ **Duplicate detected**\nA file named `{file_name}` with the same size was posted within the last 72 hours.\nDo you want to post it again?",
                message_thread_id=message_thread_id,
                file_id=None,
                notification_type='other',
                parse_mode='Markdown'
            )
            await context.bot.send_message(chat_id=chat_id, message_thread_id=message_thread_id, text=" ", reply_markup=InlineKeyboardMarkup(keyboard))
            return
        
        # Thread-safe queue addition
        queue_size = await add_to_queue_safe(state, file_info)
        queue_position = get_queue_position_safe(state)
        
        # Add to history as accepted/queued
        update_file_history(chat_id, file_name, 'queued', file_id=file_id, size=content_size)
        
        # Check if this should start processing
        should_start_processing = not state.processing and queue_size == 1
        
        if should_start_processing:
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
                notification = (
                    f"✅ File accepted: `{file_name}`\n"
                    f"Size: {content_size:,} characters{parts_info}\n"
                    f"Operation: {state.operation.capitalize()}\n\n"
                    f"🟢 Starting Your Task"
                )
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
                notification = (
                    f"✅ File accepted: `{file_name}`\n"
                    f"Size: {content_size:,} characters{parts_info}\n"
                    f"Operation: {state.operation.capitalize()}\n\n"
                    f"🟢 Starting Your Task"
                )
            
            sent_msg = await send_telegram_message_safe(
                chat_id=chat_id,
                context=context,
                message=notification,
                message_thread_id=message_thread_id,
                file_id=file_id,
                notification_type='notification'
            )
            
            if not state.processing:
                state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                
        else:
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
            
            notification = (
                f"✅ File queued: `{file_name}`\n"
                f"Size: {content_size:,} characters{parts_info}\n"
                f"Operation: {state.operation.capitalize()}\n"
                f"Position in queue: {queue_position}\n"
                f"Queue interval: {QUEUE_INTERVAL // 60} minutes between files\n\n"
            )
            
            sent_msg = await send_telegram_message_safe(
                chat_id=chat_id,
                context=context,
                message=notification,
                message_thread_id=message_thread_id,
                file_id=file_id,
                notification_type='notification'
            )
            
    except asyncio.TimeoutError:
        logger.error(f"File processing timeout for {file_name}")
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message=f"❌ **Processing timeout**\n"
                 f"File `{file_name}` is too large to process.\n"
                 f"Please try with a smaller file.",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"File processing error: {e}")
        await send_bot_message(
            chat_id=chat_id,
            context=context,
            message=f"❌ Error processing file\n{str(e)[:200]}",
            message_thread_id=message_thread_id,
            parse_mode='Markdown'
        )

# ==================== HEALTH & STARTUP ====================
async def health_handler(request):
    return web.Response(
        text=json.dumps({
            "status": "running",
            "service": "filex-bot",
            "active_users": len(user_states),
            "allowed_ids_count": len(ALLOWED_IDS),
            "admin_ids_count": len(ADMIN_IDS),
            "tracked_messages": len(message_tracking),
            "admin_preview_sessions": len(admin_preview_mode)
        }),
        content_type='application/json'
    )

async def start_web_server():
    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_get('/', health_handler)
    app.router.add_post('/webhook', lambda r: web.Response(text="OK"))
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"Health check server running on port {PORT}")
    return runner

async def main():
    if not TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    authorized_users = [uid for uid in ALLOWED_IDS if uid > 0]
    authorized_groups = [cid for cid in ALLOWED_IDS if cid < 0]
    admin_users = [uid for uid in ADMIN_IDS if uid > 0]
    
    logger.info(f"Authorized users: {authorized_users}")
    logger.info(f"Authorized groups: {authorized_groups}")
    logger.info(f"Admin users: {admin_users}")
    logger.info(f"Total authorized IDs: {len(ALLOWED_IDS)}")
    logger.info(f"Total admin IDs: {len(ADMIN_IDS)}")
    
    web_runner = await start_web_server()
    
    application = Application.builder().token(TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("operation", operation_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("queue", queue_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("pause", pause_command))
    application.add_handler(CommandHandler("resume", resume_command))
    application.add_handler(CommandHandler("skip", skip_command))
    application.add_handler(CommandHandler("delfilecontent", delfilecontent_command))
    
    # Add admin command handlers
    application.add_handler(CommandHandler("adminpreview", adminpreview_command))
    application.add_handler(CommandHandler("cancelpreview", cancelpreview_command))
    application.add_handler(CommandHandler("adminstats", adminstats_command))
    
    application.add_handler(CallbackQueryHandler(button_handler))
    
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    
    application.add_handler(
        MessageHandler(filters.Document.ALL & ~filters.COMMAND, handle_file)
    )
    
    await application.initialize()
    await application.start()
    
    # Start periodic cleanup task
    asyncio.create_task(periodic_cleanup_task())
    
    logger.info("🤖 FileX Bot started")
    
    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )
    
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Shutting down...")
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await web_runner.cleanup()

def run():
    asyncio.run(main())

if __name__ == '__main__':
    run()
