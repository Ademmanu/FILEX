#!/usr/bin/env python3
"""
FileX Bot - Clean message headers, only part headers at beginning of each part
"""

import os
import json
import asyncio
import logging
import re
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
MESSAGE_DELAY = 0.5
QUEUE_INTERVAL = 2 * 60
MAX_PROCESSING_TIME = 30 * 60

ALLOWED_USERS_STR = os.getenv('ALLOWED_USERS', '6389552329')
ALLOWED_IDS = set()
for id_str in ALLOWED_USERS_STR.split(','):
    id_str = id_str.strip()
    if id_str.lstrip('-').isdigit():
        ALLOWED_IDS.add(int(id_str))

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

# ==================== FILE HISTORY & TRACKING ====================
class FileHistoryEntry:
    """Stores complete file history with unique tracking"""
    def __init__(self, chat_id: int, filename: str, size: int, timestamp: datetime, 
                 status: str, parts_count: int = 0, messages_count: int = 0,
                 duplicate_counter: int = 0):
        self.chat_id = chat_id
        self.original_filename = filename
        self.filename = filename
        self.size = size
        self.timestamp = timestamp
        self.status = status
        self.parts_count = parts_count
        self.messages_count = messages_count
        self.duplicate_counter = duplicate_counter
        
        # Add suffix for duplicates
        if duplicate_counter > 0:
            self.filename = f"{filename} ({duplicate_counter})"
        
        # Unique tracking ID
        self.tracking_id = f"{chat_id}_{filename}_{size}_{timestamp.timestamp()}_{duplicate_counter}"
    
    def to_dict(self) -> Dict:
        return {
            'chat_id': self.chat_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'size': self.size,
            'timestamp': self.timestamp,
            'status': self.status,
            'parts_count': self.parts_count,
            'messages_count': self.messages_count,
            'duplicate_counter': self.duplicate_counter,
            'tracking_id': self.tracking_id
        }
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)

# Global storage
file_history_db: Dict[str, FileHistoryEntry] = {}  # tracking_id -> FileHistoryEntry
file_message_mapping: Dict[str, List[int]] = {}  # tracking_id -> message_ids
file_notification_mapping: Dict[str, int] = {}  # tracking_id -> notification_msg_id
file_other_notifications: Dict[str, List[int]] = {}  # tracking_id -> other_msg_ids

# ==================== MESSAGE TRACKING ====================
class MessageEntry:
    """Stores first two words of sent messages"""
    def __init__(self, chat_id: int, message_id: int, message_text: str, 
                 first_two_words: str, timestamp: datetime):
        self.chat_id = chat_id
        self.message_id = message_id
        self.message_text = message_text
        self.first_two_words = first_two_words
        self.timestamp = timestamp
        # Store all words for partial matching
        self.all_words = self.extract_all_words(message_text)
    
    def extract_all_words(self, text: str) -> List[str]:
        """Extract all words from text"""
        clean_text = text
        clean_text = re.sub(r'https?://\S+', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text.split()
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)

message_tracking: List[MessageEntry] = []
admin_preview_mode: Set[int] = set()

# ==================== UTILITY FUNCTIONS ====================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def check_admin_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    
    if is_admin(user_id):
        return True
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="⛔ **Admin Access Required**\n\nThis command is for administrators only."
    )
    return False

def is_authorized(user_id: int, chat_id: int) -> bool:
    return user_id in ALLOWED_IDS or chat_id in ALLOWED_IDS

async def check_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if is_authorized(user_id, chat_id):
        return True
    
    message_thread_id = update.effective_message.message_thread_id
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="⛔ **Access Denied**\n\nYou are not authorized to use this bot."
    )
    
    try:
        is_group = chat_id < 0
        entity_type = "Group" if is_group else "User"
        
        await context.bot.send_message(
            chat_id=ADMIN_USER_ID,
            text=f"⚠️ **Unauthorized Access Attempt**\n\n"
                 f"**Entity Type:** {entity_type}\n"
                 f"**Chat ID:** `{chat_id}`\n"
                 f"**User ID:** `{user_id}`\n"
                 f"**Username:** @{update.effective_user.username if update.effective_user.username else 'N/A'}\n"
                 f"**Full Name:** {update.effective_user.full_name}\n"
                 f"**Time:** {datetime.now(UTC_PLUS_1).strftime('%Y-%m-%d %H:%M:%S')}",
            parse_mode='Markdown'
        )
        logger.warning(f"Unauthorized access attempt - Chat: {chat_id}, User: {user_id}")
    except Exception as e:
        logger.error(f"Failed to notify admin about unauthorized access: {e}")
    
    return False

# ==================== MESSAGE TRACKING FUNCTIONS ====================
def extract_first_two_words(text: str) -> str:
    """Extract first two words from text"""
    clean_text = text
    clean_text = re.sub(r'https?://\S+', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    words = clean_text.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    elif len(words) == 1:
        return words[0]
    else:
        return ""

def track_message(chat_id: int, message_id: int, message_text: str):
    """Track message for preview system"""
    try:
        if not message_text or len(message_text.strip()) < 2:
            return
        
        first_two_words = extract_first_two_words(message_text)
        if not first_two_words:
            return
        
        # Remove expired entries first
        cleanup_old_messages()
        
        # Add new entry
        entry = MessageEntry(
            chat_id=chat_id,
            message_id=message_id,
            message_text=message_text,
            first_two_words=first_two_words,
            timestamp=datetime.now(UTC_PLUS_1)
        )
        
        message_tracking.append(entry)
        logger.debug(f"Tracked message: {first_two_words}")
        
    except Exception as e:
        logger.error(f"Error tracking message: {e}")

def cleanup_old_messages():
    """Remove messages older than 72 hours"""
    global message_tracking
    
    if not message_tracking:
        return
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_count = len(message_tracking)
    
    message_tracking = [entry for entry in message_tracking if not entry.is_expired()]
    
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

# ==================== PREVIEW PROCESSING FUNCTIONS ====================
def extract_preview_sections(text: str) -> List[str]:
    """Extract all 'Preview:' sections from report text"""
    preview_sections = []
    
    pattern = r'📝\s*[Pp]review:\s*(.+?)(?=\n|\r|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        preview_text = match.strip()
        if preview_text:
            preview_sections.append(preview_text)
    
    if not preview_sections:
        pattern2 = r'[Pp]review:\s*(.+?)(?=\n|\r|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for match in matches2:
            preview_text = match.strip()
            if preview_text:
                preview_sections.append(preview_text)
    
    return preview_sections

def check_preview_against_database(preview_texts: List[str]) -> Tuple[List[Tuple[str, str, List[int]]], 
                                                                     List[Tuple[str, Dict[str, List[int]]]], 
                                                                     List[str]]:
    """
    Check preview texts against tracked messages
    Returns: (full_matches, partial_matches, non_matches)
    full_matches: [(preview_text, matched_words, [])]
    partial_matches: [(preview_text, {matched_word: [positions]})]
    non_matches: [preview_text]
    """
    if not message_tracking:
        return ([], [], preview_texts)
    
    full_matches = []
    partial_matches = []
    non_matches = []
    
    for preview in preview_texts:
        # Extract words from preview
        preview_words = [w for w in re.sub(r'\s+', ' ', preview).strip().split() if w]
        
        if not preview_words:
            non_matches.append(preview)
            continue
        
        # Check for full match (first two words)
        first_two_preview = ' '.join(preview_words[:2]) if len(preview_words) >= 2 else preview_words[0]
        is_full_match = False
        
        for entry in message_tracking:
            if entry.first_two_words == first_two_preview:
                full_matches.append((preview, first_two_preview, []))
                is_full_match = True
                break
        
        if is_full_match:
            continue
        
        # Check for partial matches
        partial_matches_dict = {}
        for preview_word in preview_words:
            word_positions = []
            
            for entry in message_tracking:
                for idx, word in enumerate(entry.all_words, 1):
                    if word == preview_word:
                        word_positions.append(idx)
            
            if word_positions:
                partial_matches_dict[preview_word] = word_positions
        
        if partial_matches_dict:
            partial_matches.append((preview, partial_matches_dict))
        else:
            non_matches.append(preview)
    
    return (full_matches, partial_matches, non_matches)

def format_preview_report(full_matches: List[Tuple[str, str, List[int]]], 
                         partial_matches: List[Tuple[str, Dict[str, List[int]]]], 
                         non_matches: List[str], total_previews: int) -> str:
    """Format the preview report"""
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Preview Analysis Report**")
    report_lines.append("─" * 35)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    full_percent = (len(full_matches) / total_previews * 100) if total_previews > 0 else 0
    partial_percent = (len(partial_matches) / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (len(non_matches) / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Full matches found: {len(full_matches)} ({full_percent:.1f}%)")
    report_lines.append(f"• Partial matches: {len(partial_matches)} ({partial_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {len(non_matches)} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if full_matches:
        report_lines.append("✅ **Full matches found in database:**")
        for i, (preview, matched_words, _) in enumerate(full_matches, 1):
            report_lines.append(f"{i}. {preview}")
    
    if partial_matches:
        if full_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Partial matches found:**")
        for i, (preview, matches_dict) in enumerate(partial_matches, 1):
            report_lines.append(f"{i}. Preview: {preview}")
            for word, positions in matches_dict.items():
                pos_str = ', '.join(map(str, positions[:5]))  # Show first 5 positions
                if len(positions) > 5:
                    pos_str += f"... ({len(positions)} total)"
                report_lines.append(f"   Matched words: {word}")
                report_lines.append(f"   Word position: {pos_str}")
            if i < len(partial_matches):
                report_lines.append("")
    
    if non_matches:
        if full_matches or partial_matches:
            report_lines.append("")
        report_lines.append("❌ **Not found in database:**")
        for i, non_match in enumerate(non_matches, 1):
            report_lines.append(f"{i}. {non_match}")

    report_lines.append("")
    report_lines.append("Use /cancelpreview to cancel.")
                                   
    return "\n".join(report_lines)

# ==================== FILE HISTORY MANAGEMENT ====================
def get_duplicate_counter(chat_id: int, filename: str, size: int) -> int:
    """Get the next duplicate counter for a file"""
    counters = []
    for entry in file_history_db.values():
        if entry.chat_id == chat_id and entry.original_filename == filename and entry.size == size:
            counters.append(entry.duplicate_counter)
    
    if not counters:
        return 0
    return max(counters) + 1

def update_file_history(chat_id: int, filename: str, size: int, status: str, 
                       parts_count: int = 0, messages_count: int = 0) -> str:
    """Update file history with unique tracking"""
    # Check if this is a duplicate (same name & size within 72 hours)
    duplicate_counter = 0
    cutoff_time = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    
    for entry in file_history_db.values():
        if (entry.chat_id == chat_id and 
            entry.original_filename == filename and 
            entry.size == size and
            entry.timestamp >= cutoff_time):
            duplicate_counter = get_duplicate_counter(chat_id, filename, size)
            break
    
    # Create unique entry
    entry = FileHistoryEntry(
        chat_id=chat_id,
        filename=filename,
        size=size,
        timestamp=datetime.now(UTC_PLUS_1),
        status=status,
        parts_count=parts_count,
        messages_count=messages_count,
        duplicate_counter=duplicate_counter
    )
    
    file_history_db[entry.tracking_id] = entry
    
    # Clean up old entries for this chat
    cleanup_old_file_history(chat_id)
    
    return entry.tracking_id

def cleanup_old_file_history(chat_id: int = None):
    """Remove file history entries older than 72 hours"""
    global file_history_db
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    to_remove = []
    
    for tracking_id, entry in file_history_db.items():
        if entry.is_expired() and (chat_id is None or entry.chat_id == chat_id):
            to_remove.append(tracking_id)
    
    for tracking_id in to_remove:
        # Also clean up message mappings
        if tracking_id in file_message_mapping:
            del file_message_mapping[tracking_id]
        if tracking_id in file_notification_mapping:
            del file_notification_mapping[tracking_id]
        if tracking_id in file_other_notifications:
            del file_other_notifications[tracking_id]
        del file_history_db[tracking_id]
    
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} expired file history entries")

def get_file_history(chat_id: int, hours: int = 12) -> List[FileHistoryEntry]:
    """Get recent file history for a chat"""
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=hours)
    
    entries = []
    for entry in file_history_db.values():
        if entry.chat_id == chat_id and entry.timestamp >= cutoff:
            entries.append(entry)
    
    entries.sort(key=lambda x: x.timestamp, reverse=True)
    return entries

def find_files_by_name(chat_id: int, filename: str) -> List[FileHistoryEntry]:
    """Find all files with given name (any size)"""
    files = []
    for entry in file_history_db.values():
        if entry.chat_id == chat_id and entry.original_filename == filename:
            files.append(entry)
    
    files.sort(key=lambda x: x.timestamp, reverse=True)
    return files

def find_duplicate_file(chat_id: int, filename: str, size: int) -> Optional[FileHistoryEntry]:
    """Check if same file (name & size) was processed within 72 hours"""
    cutoff_time = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    
    for entry in file_history_db.values():
        if (entry.chat_id == chat_id and 
            entry.original_filename == filename and 
            entry.size == size and
            entry.timestamp >= cutoff_time):
            return entry
    
    return None

# ==================== ADMIN COMMANDS ====================
async def adminpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to start preview mode"""
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    admin_preview_mode.add(user_id)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="🔍 **Admin Preview Mode Activated**\n\n"
             "Please send the report data with 'Preview:' sections.\n"
             "Example format:\n"
             "```\n"
             "Preview: 97699115546 97699115547\n"
             "Preview: 237620819778 237620819780\n"
             "```\n\n"
             "Use /cancelpreview to cancel.",
        parse_mode='Markdown'
    )

async def cancelpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to cancel preview mode"""
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    
    if user_id not in admin_preview_mode:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.effective_message.message_thread_id,
            text="ℹ️ **Not in preview mode**\n"
                 "Use /adminpreview to start preview mode.",
            parse_mode='Markdown'
        )
        return
    
    admin_preview_mode.remove(user_id)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="🚫 **Preview Mode Cancelled**\n\n"
             "Preview mode has been deactivated.\n\n"
                 "Use /adminpreview to start preview mode again.",
        parse_mode='Markdown'
    )

async def adminstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to show tracking statistics"""
    if not await check_admin_authorization(update, context):
        return
    
    stats = get_tracking_stats()
    
    if stats['total'] == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.effective_message.message_thread_id,
            text="📊 **Message Tracking Statistics**\n\n"
                 "No messages tracked yet.\n"
                 "Database is empty.",
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
    message += f"**Active preview sessions:** {len(admin_preview_mode)}\n"
    message += f"**File history entries:** {len(file_history_db)}\n\n"
    message += f"Messages auto-delete after 72 hours."
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text=message,
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
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="⚠️ **Empty message**\n"
                 "Please send a message with preview content to check.\n\n"
                 "Use /cancelpreview to cancel.",
            parse_mode='Markdown'
        )
        return
    
    previews = extract_preview_sections(message_text)
    
    if not previews:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="❌ **No preview content found**\n\n"
                 "I couldn't find any 'Preview:' sections in your message.\n"
                 "Please use the exact format:\n"
                 "```\n"
                 "📝 Preview: [content]\n"
                 "```\n\n"
             "Use /cancelpreview to cancel.",
            parse_mode='Markdown'
        )
        return
    
    # Check against database
    full_matches, partial_matches, non_matches = check_preview_against_database(previews)
    total_previews = len(previews)
    
    report_text = format_preview_report(full_matches, partial_matches, non_matches, total_previews)
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=report_text,
        parse_mode='Markdown'
    )

# ==================== USER STATE MANAGEMENT ====================
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
        self.duplicate_confirmation_pending = False
        self.duplicate_file_info = None
        
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
    
    def remove_task_by_tracking_id(self, tracking_id: str):
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
            if task.get('tracking_id') != tracking_id:
                new_queue.append(task)
        self.queue = new_queue

user_states: Dict[int, UserState] = {}
user_state_locks: Dict[int, asyncio.Lock] = {}
queue_locks: Dict[int, asyncio.Lock] = {}

def get_user_state(chat_id: int) -> UserState:
    if chat_id not in user_states:
        user_states[chat_id] = UserState(chat_id)
    return user_states[chat_id]

async def get_user_state_safe(chat_id: int) -> UserState:
    async with user_state_locks.setdefault(chat_id, asyncio.Lock()):
        return get_user_state(chat_id)

async def add_to_queue_safe(state: UserState, file_info: dict) -> int:
    async with queue_locks.setdefault(state.chat_id, asyncio.Lock()):
        state.queue.append(file_info)
        return len(state.queue)

def get_queue_position_safe(state: UserState) -> int:
    if not state.queue:
        return 0
    
    position = len(state.queue)
    if state.processing:
        position -= 1
    return max(0, position)

# ==================== FILE PROCESSING FUNCTIONS ====================
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

async def send_telegram_message_safe(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                    message: str, message_thread_id: Optional[int] = None, 
                                    retries: int = 5, filename: Optional[str] = None,
                                    tracking_id: Optional[str] = None,
                                    notification_type: str = 'content') -> bool:
    for attempt in range(retries):
        try:
            if len(message) > TELEGRAM_MESSAGE_LIMIT:
                message = message[:TELEGRAM_MESSAGE_LIMIT]
            
            sent_message = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=message,
                disable_notification=True,
                parse_mode='Markdown'
            )
            
            if sent_message and sent_message.text:
                track_message(chat_id, sent_message.message_id, sent_message.text)
            
            if tracking_id and sent_message:
                if notification_type == 'content':
                    if tracking_id not in file_message_mapping:
                        file_message_mapping[tracking_id] = []
                    file_message_mapping[tracking_id].append(sent_message.message_id)
                elif notification_type == 'notification':
                    file_notification_mapping[tracking_id] = sent_message.message_id
            
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for chat {chat_id}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return False

async def track_other_notification(tracking_id: str, message_id: int):
    if tracking_id not in file_other_notifications:
        file_other_notifications[tracking_id] = []
    file_other_notifications[tracking_id].append(message_id)

async def send_chunks_immediately(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                chunks: List[str], filename: str, tracking_id: str,
                                message_thread_id: Optional[int] = None) -> bool:
    try:
        total_messages_sent = 0
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, 
                                              filename=filename, tracking_id=tracking_id):
                total_messages_sent += 1
                
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for `{filename}`")
                return False
        
        state = await get_user_state_safe(chat_id)
        state.last_send = datetime.now(UTC_PLUS_1)
        
        if total_messages_sent > 0:
            completion_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"✅ Completed: `{filename}`\n📊 Sent {total_messages_sent} message{'s' if total_messages_sent > 1 else ''}",
                disable_notification=True,
                parse_mode='Markdown'
            )
            await track_other_notification(tracking_id, completion_msg.message_id)
            
            update_file_history(chat_id, filename, total_messages_sent, 'completed', messages_count=total_messages_sent)
            return True
        else:
            logger.error(f"No chunks sent for `{filename}`")
            return False
        
    except Exception as e:
        logger.error(f"Error in send_chunks_immediately: {e}")
        return False

async def send_large_content_part(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                part: str, part_num: int, total_parts: int, 
                                filename: str, tracking_id: str, message_thread_id: Optional[int] = None) -> int:
    try:
        chunks = split_into_telegram_chunks_without_cutting_words(part, TELEGRAM_MESSAGE_LIMIT)
        total_messages_in_part = 0
        
        if total_parts > 1:
            part_header_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"📄 `{filename}` - Part {part_num}/{total_parts}",
                disable_notification=True,
                parse_mode='Markdown'
            )
            await track_other_notification(tracking_id, part_header_msg.message_id)
            total_messages_in_part += 1
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, 
                                              filename=filename, tracking_id=tracking_id):
                total_messages_in_part += 1
                
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for part {part_num} of `{filename}`")
                return 0
        
        return total_messages_in_part
        
    except Exception as e:
        logger.error(f"Error sending large content part: {e}")
        return 0

async def send_with_intervals(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                            parts: List[str], filename: str, tracking_id: str,
                            state: UserState, message_thread_id: Optional[int] = None) -> bool:
    try:
        total_parts = len(parts)
        total_messages_sent = 0
        
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
                chat_id, context, part, i, total_parts, filename, tracking_id, message_thread_id
            )
            
            if not messages_in_part:
                logger.error(f"Failed to send part {i} of `{filename}`")
                return False
            
            total_messages_sent += messages_in_part
            
            state.last_send = datetime.now(UTC_PLUS_1)
            
            if i < total_parts:
                await asyncio.sleep(SEND_INTERVAL * 60)
        
        completion_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"✅ Completed: `{filename}`\n📊 Sent {total_parts} part{'s' if total_parts > 1 else ''} ({total_messages_sent} messages total)",
            disable_notification=True,
            parse_mode='Markdown'
        )
        await track_other_notification(tracking_id, completion_msg.message_id)
        
        update_file_history(chat_id, filename, len(parts[0]), 'completed', 
                          parts_count=total_parts, messages_count=total_messages_sent)
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def cleanup_completed_file(tracking_id: str, chat_id: int):
    """Clean up tracking for completed file"""
    if tracking_id in file_message_mapping:
        del file_message_mapping[tracking_id]
    if tracking_id in file_notification_mapping:
        del file_notification_mapping[tracking_id]
    if tracking_id in file_other_notifications:
        del file_other_notifications[tracking_id]
    
    logger.info(f"Cleaned up tracking for completed file: {tracking_id} in chat {chat_id}")

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
            tracking_id = file_info['tracking_id']
            file_message_thread_id = file_info.get('message_thread_id', message_thread_id)
            
            if file_info.get('requires_intervals', False):
                update_file_history(chat_id, filename, file_info['size'], 'running', parts_count=len(file_info['parts']))
            else:
                update_file_history(chat_id, filename, file_info['size'], 'running', messages_count=len(file_info['chunks']))
            
            if len(state.queue) > 0 and state.queue[0] == file_info:
                if file_info.get('requires_intervals', False):
                    sending_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"📤 Sending: `{filename}`\n"
                             f"📊 Total parts: {len(file_info['parts'])}\n"
                             f"⏰ Interval: {SEND_INTERVAL} minutes between parts",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    await track_other_notification(tracking_id, sending_msg.message_id)
                else:
                    sending_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"📤 Sending: `{filename}`\n"
                             f"📊 Total messages: {len(file_info['chunks'])}",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    await track_other_notification(tracking_id, sending_msg.message_id)
            
            success = False
            if file_info.get('requires_intervals', False):
                success = await send_with_intervals(
                    chat_id, context, 
                    file_info['parts'], 
                    filename,
                    tracking_id,
                    state,
                    file_message_thread_id
                )
            else:
                success = await send_chunks_immediately(
                    chat_id, context,
                    file_info['chunks'],
                    filename,
                    tracking_id,
                    file_message_thread_id
                )
            
            if state.queue and state.queue[0]['tracking_id'] == tracking_id:
                processed_file = state.queue.popleft()
                
                if success and not state.cancel_requested:
                    logger.info(f"Successfully processed `{filename}` for chat {chat_id}")
                    await cleanup_completed_file(tracking_id, chat_id)
                else:
                    logger.error(f"Failed to process `{filename}` for chat {chat_id}")
                    failed_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"❌ Failed to send: `{filename}`\nPlease try uploading again.",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    await track_other_notification(tracking_id, failed_msg.message_id)
            
            state.current_parts = []
            state.current_index = 0
            
            if state.queue and not state.cancel_requested:
                next_file = state.queue[0]['name']
                next_file_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=f"⏰ **Queue Interval**\n\n"
                         f"Next file `{next_file}` will start in 2 minutes...",
                    parse_mode='Markdown'
                )
                
                wait_start = datetime.now(UTC_PLUS_1)
                while (datetime.now(UTC_PLUS_1) - wait_start).seconds < QUEUE_INTERVAL:
                    if state.cancel_requested:
                        await context.bot.delete_message(chat_id=chat_id, message_id=next_file_msg.message_id)
                        break
                    await asyncio.sleep(1)
                
                if not state.cancel_requested:
                    await context.bot.delete_message(chat_id=chat_id, message_id=next_file_msg.message_id)
            
            if state.cancel_requested:
                break
                
    except asyncio.CancelledError:
        logger.info(f"Queue processing cancelled for chat {chat_id}")
    except Exception as e:
        logger.error(f"Queue processing error: {e}")
        error_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"⚠️ Processing error\n{str(e)[:200]}",
            disable_notification=True,
            parse_mode='Markdown'
        )
        if state.queue:
            current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
            tracking_id = state.queue[0].get('tracking_id', 'unknown')
            await track_other_notification(tracking_id, error_msg.message_id)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

# ==================== DUPLICATE FILE HANDLING ====================
async def handle_duplicate_file(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                               chat_id: int, message_thread_id: int, 
                               filename: str, size: int, content: str, 
                               operation: str, requires_intervals: bool,
                               chunks: List[str] = None, parts: List[str] = None,
                               duplicate_entry: FileHistoryEntry = None):
    """Handle duplicate file detection with confirmation"""
    state = await get_user_state_safe(chat_id)
    
    if duplicate_entry:
        time_diff = datetime.now(UTC_PLUS_1) - duplicate_entry.timestamp
        hours_ago = time_diff.total_seconds() / 3600
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Post", callback_data=f"duplicate_yes_{filename}_{size}"),
                InlineKeyboardButton("❌ No, Don't", callback_data=f"duplicate_no_{filename}_{size}")
            ]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"⚠️ **Duplicate File Detected**\n\n"
                 f"File: `{filename}`\n"
                 f"Size: {size:,} characters\n"
                 f"Previous status: {duplicate_entry.status.capitalize()} ({hours_ago:.1f} hours ago)\n\n"
                 f"Do you want to process it again?\n\n",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        
        state.duplicate_confirmation_pending = True
        state.duplicate_file_info = {
            'filename': filename,
            'size': size,
            'content': content,
            'operation': operation,
            'requires_intervals': requires_intervals,
            'chunks': chunks,
            'parts': parts
        }
    else:
        await process_file_upload(update, context, chat_id, message_thread_id,
                                 filename, size, content, operation, requires_intervals,
                                 chunks, parts)

async def process_duplicate_confirmation(query, context: ContextTypes.DEFAULT_TYPE, 
                                        action: str, filename: str, size: int):
    """Process duplicate file confirmation"""
    chat_id = query.message.chat_id
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.duplicate_confirmation_pending or not state.duplicate_file_info:
        await query.answer("This confirmation has expired. Please upload the file again.")
        return
    
    if action == 'yes':
        await query.answer("Processing duplicate file...")
        
        file_info = state.duplicate_file_info
        
        await process_file_upload(None, context, chat_id, message_thread_id,
                                 filename, size, file_info['content'], 
                                 file_info['operation'], file_info['requires_intervals'],
                                 file_info.get('chunks'), file_info.get('parts'),
                                 is_duplicate=True)
    else:
        await query.answer("File upload cancelled.")
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **File upload cancelled**\n\n"
                 f"`{filename}` will not be processed.\n"
                 f"You may upload a different file.",
            parse_mode='Markdown'
        )
    
    state.duplicate_confirmation_pending = False
    state.duplicate_file_info = None

# ==================== FILE DELETION FUNCTIONS ====================
async def process_file_deletion_selection(query, context: ContextTypes.DEFAULT_TYPE, 
                                         file_index: int, files: List[FileHistoryEntry]):
    """Process file deletion selection from multiple files"""
    chat_id = query.message.chat_id
    message_thread_id = query.message.message_thread_id
    
    if file_index < 0 or file_index >= len(files):
        await query.answer("Invalid selection.")
        return
    
    selected_file = files[file_index]
    await query.answer(f"Deleting {selected_file.filename}...")
    
    await delete_file_content(chat_id, message_thread_id, selected_file, context)

async def delete_file_content(chat_id: int, message_thread_id: Optional[int], 
                             file_entry: FileHistoryEntry, context: ContextTypes.DEFAULT_TYPE):
    """Actually delete file content from chat"""
    tracking_id = file_entry.tracking_id
    filename = file_entry.filename
    original_filename = file_entry.original_filename
    
    messages_to_delete = 0
    deleted_messages = 0
    already_deleted = 0
    
    # Delete content messages
    if tracking_id in file_message_mapping:
        messages_to_delete += len(file_message_mapping[tracking_id])
        for msg_id in file_message_mapping[tracking_id]:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                deleted_messages += 1
                logger.info(f"Deleted content message {msg_id} for file {filename}")
            except Exception as e:
                logger.error(f"Failed to delete content message {msg_id}: {e}")
                already_deleted += 1
    
    # Delete other notifications
    if tracking_id in file_other_notifications:
        for msg_id in file_other_notifications[tracking_id]:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                deleted_messages += 1
                logger.info(f"Deleted other notification message {msg_id} for file {filename}")
            except Exception as e:
                logger.error(f"Failed to delete other notification message {msg_id}: {e}")
                already_deleted += 1
    
    # Edit acceptance/queued notification
    notification_edited = False
    if tracking_id in file_notification_mapping:
        notification_msg_id = file_notification_mapping[tracking_id]
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=notification_msg_id,
                text=f"🗑️ **File Content Deleted**\n\n"
                     f"File: `{filename}`\n"
                     f"Messages deleted: {deleted_messages}\n"
                     f"All content from this file has been removed.",
                parse_mode='Markdown'
            )
            notification_edited = True
            logger.info(f"Edited notification for {filename}")
        except Exception as e:
            logger.error(f"Failed to edit notification message: {e}")
    
    # Clean up tracking
    if tracking_id in file_message_mapping:
        del file_message_mapping[tracking_id]
    if tracking_id in file_notification_mapping:
        del file_notification_mapping[tracking_id]
    if tracking_id in file_other_notifications:
        del file_other_notifications[tracking_id]
    
    # Remove from history
    if tracking_id in file_history_db:
        del file_history_db[tracking_id]
    
    # Update user state
    state = await get_user_state_safe(chat_id)
    state.remove_task_by_tracking_id(tracking_id)
    
    # Send confirmation
    result_message = f"🗑️ `{filename}` content deleted\n"
    
    if messages_to_delete > 0:
        result_message += f"Messages found: {messages_to_delete}\n"
        result_message += f"Successfully deleted: {deleted_messages}"
        if already_deleted > 0:
            result_message += f"\nAlready deleted: {already_deleted}"
    else:
        result_message += "No tracked messages found (may have been already deleted)"
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=result_message,
        parse_mode='Markdown'
    )
    
    # Check if we need to start next task
    if state.processing and not state.queue:
        state.cancel_current_task()
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="🏁 **Processing stopped**\n\n"
                 "No more tasks in queue.",
            parse_mode='Markdown'
        )
    elif not state.processing and state.queue:
        next_file = state.queue[0].get('name', 'Unknown')
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🔄 **Moving to next task**\n\n"
                 f"Starting next task: `{next_file}`",
            parse_mode='Markdown'
        )
        state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))

# ==================== COMMAND HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="🤖 **FileX Bot**\n\n"
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
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=f"{current_op_text}\n"
        "🔧 **Select new operation:**\n\n"
        "This will be remembered for all future file uploads.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    if not is_authorized(user_id, chat_id):
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=query.message.message_thread_id,
            text="⛔ **Access Denied**\n\nYou are not authorized to use this bot."
        )
        return
    
    data = query.data
    
    # Handle duplicate confirmation
    if data.startswith('duplicate_'):
        parts = data.split('_')
        if len(parts) >= 4:
            action = parts[1]
            filename = parts[2]
            try:
                size = int(parts[3])
                await process_duplicate_confirmation(query, context, action, filename, size)
            except ValueError:
                await query.answer("Invalid file size.")
        return
    
    # Handle file deletion selection
    if data.startswith('delete_file_'):
        parts = data.split('_')
        if len(parts) >= 3:
            try:
                file_index = int(parts[2])
                # Get files from callback context
                files = context.user_data.get('delete_files', [])
                await process_file_deletion_selection(query, context, file_index, files)
                return
            except ValueError:
                await query.answer("Invalid file selection.")
        return
    
    # Handle deletion cancel
    if data == 'delete_cancel':
        await query.answer("Deletion cancelled.")
        state = await get_user_state_safe(chat_id)
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=query.message.message_thread_id,
            text="❌ **Deletion operation cancelled**",
            parse_mode='Markdown'
        )
        return
    
    # Handle operation change
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    operation = data
    
    operation_names = {
        'all': '✅ All content',
        'number': '🔢 Number only',
        'alphabet': '🔤 Alphabet only'
    }
    
    if operation in operation_names:
        state.operation = operation
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"✅ **Operation updated:** {operation_names[operation]}\n\n"
            "All future files will be processed with this operation.\n\n"
            "📤 Now upload a TXT or CSV file!",
            parse_mode='Markdown'
        )

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
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="\n".join(status_lines),
        parse_mode='Markdown'
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    
    recent_files = get_file_history(chat_id, hours=12)
    
    if not recent_files:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="📊 **Last 12 Hours Stats**\n\n"
                 "No files processed in the last 12 hours."
        )
        return
    
    stats_text = "📊 **Last 12 Hours Stats**\n"
    stats_text += "────────────────────\n\n"
    
    for entry in recent_files:
        time_str = entry.timestamp.strftime('%H:%M:%S')
        status_emoji = {
            'completed': '✅',
            'skipped': '⏭️',
            'deleted': '🗑️',
            'cancelled': '🚫',
            'running': '📤',
            'paused': '⏸️',
            'timeout_cancelled': '⏱️'
        }.get(entry.status, '📝')
        
        count_info = ""
        if entry.status == 'completed':
            if entry.parts_count > 0:
                count_info = f" ({entry.parts_count} part{'s' if entry.parts_count > 1 else ''}"
                if entry.messages_count > 0:
                    count_info += f", {entry.messages_count} messages"
                count_info += ")"
            elif entry.messages_count > 0:
                count_info = f" ({entry.messages_count} message{'s' if entry.messages_count > 1 else ''})"
        
        stats_text += f"{status_emoji} `{entry.filename}`\n"
        stats_text += f"   Status: {entry.status.capitalize()}{count_info}\n"
        stats_text += f"   Size: {entry.size:,} chars\n"
        stats_text += f"   Time: {time_str}\n\n"
    
    stats_text += f"Total files: {len(recent_files)}"
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=stats_text,
        parse_mode='Markdown'
    )

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.queue:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="📭 **Queue is empty**"
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
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=queue_text,
        parse_mode='Markdown'
    )

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.processing:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **No active task to pause**\n"
                 "There's no task currently running."
        )
        return
    
    if state.paused:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **Task already paused**\n"
                 "Use /resume to continue."
        )
        return
    
    state.paused_progress = {
        'current_index': state.current_index,
        'current_parts': state.current_parts.copy() if state.current_parts else []
    }
    
    state.pause()
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    
    if state.queue:
        file_info = state.queue[0]
        if file_info.get('requires_intervals', False):
            update_file_history(chat_id, current_file, file_info.get('size', 0), 'paused', parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, file_info.get('size', 0), 'paused', messages_count=len(file_info['chunks']))
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="⏸️ **Task Paused**\n\n"
             f"Task `{current_file}` has been paused.\n"
             f"Progress saved at part {state.current_index} of {len(state.current_parts) if state.current_parts else 0}.\n\n"
             "Use /resume to continue where you left off.",
        parse_mode='Markdown'
    )

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.paused:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **No paused task to resume**\n"
                 "There's no task currently paused."
        )
        return
    
    if not state.processing:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="⚠️ **Cannot resume**\n"
                 "The paused task is no longer active."
        )
        state.paused = False
        return
    
    if state.paused_progress:
        state.current_index = state.paused_progress['current_index']
        state.current_parts = state.paused_progress['current_parts']
    
    state.resume()
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    
    if state.queue:
        file_info = state.queue[0]
        if file_info.get('requires_intervals', False):
            update_file_history(chat_id, current_file, file_info.get('size', 0), 'running', parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, file_info.get('size', 0), 'running', messages_count=len(file_info['chunks']))
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="▶️ **Task Resumed**\n\n"
             f"Resuming `{current_file}` from part {state.current_index} of {len(state.current_parts) if state.current_parts else 0}.\n"
             "Task will continue automatically.",
        parse_mode='Markdown'
    )

async def skip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.processing and not state.queue:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **No task to skip**\n"
                 "There's no task currently running or in queue."
        )
        return
    
    if not state.queue:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **Queue is empty**\n"
                 "No tasks to skip."
        )
        return
    
    current_file = state.queue[0].get('name', 'Unknown') if state.queue else 'Unknown'
    
    update_file_history(chat_id, current_file, state.queue[0].get('size', 0), 'skipped')
    
    state.skip()
    
    if state.queue:
        next_file = state.queue[0].get('name', 'Unknown')
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"⏭️ **Task Skipped**\n\n"
                 f"Skipped: `{current_file}`\n"
                 f"Starting next task: `{next_file}`",
            parse_mode='Markdown'
        )
        
        if not state.processing:
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"⏭️ **Task Skipped**\n\n"
                 f"Skipped: `{current_file}`\n"
                 "Queue is now empty.",
            parse_mode='Markdown'
        )

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.has_active_tasks():
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="ℹ️ **No active tasks to cancel**\n"
            "No processing is currently running and the queue is empty.",
            parse_mode='Markdown'
        )
        return
    
    if state.queue:
        for file_info in state.queue:
            if file_info.get('requires_intervals', False):
                update_file_history(chat_id, file_info['name'], file_info.get('size', 0), 
                                  'cancelled', parts_count=len(file_info.get('parts', [])))
            else:
                update_file_history(chat_id, file_info['name'], file_info.get('size', 0), 
                                  'cancelled', messages_count=len(file_info.get('chunks', [])))
    
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
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="\n".join(response_lines),
        parse_mode='Markdown'
    )

async def delfilecontent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if state.waiting_for_filename:
        filename = update.message.text.strip()
        
        if filename == state.last_deleted_file:
            state.waiting_for_filename = False
            state.last_deleted_file = None
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"⚠️ **File already deleted**\n\n"
                     f"`{filename}` was already deleted in the previous operation.\n"
                     f"Use /delfilecontent to delete a different file.",
                parse_mode='Markdown'
            )
            return
        
        state.waiting_for_filename = False
        state.last_deleted_file = filename
        
        # Find files with this name
        files = find_files_by_name(chat_id, filename)
        
        if not files:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **File not found**\n\n"
                     f"No record found for `{filename}` in your history.\n\n"
                     f"Use /delfilecontent to try again with a different filename.",
                parse_mode='Markdown'
            )
            return
        
        # If only one file found, delete it directly
        if len(files) == 1:
            await delete_file_content(chat_id, message_thread_id, files[0], context)
            return
        
        # Multiple files found - show selection
        keyboard = []
        for i, file_entry in enumerate(files):
            time_diff = datetime.now(UTC_PLUS_1) - file_entry.timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            button_text = f"File {i+1} ({file_entry.size:,} chars) - {hours_ago:.1f}h ago"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"delete_file_{i}")])
        
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="delete_cancel")])
        
        # Store files in context for callback
        context.user_data['delete_files'] = files
        
        file_list_text = "⚠️ **Multiple Files Found**\n\n"
        file_list_text += f"Found {len(files)} files named `{filename}`:\n\n"
        
        for i, file_entry in enumerate(files, 1):
            time_diff = datetime.now(UTC_PLUS_1) - file_entry.timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            file_list_text += f"{i}. `{file_entry.filename}`\n"
            file_list_text += f"   Size: {file_entry.size:,} chars\n"
            file_list_text += f"   Status: {file_entry.status.capitalize()}\n"
            file_list_text += f"   Age: {hours_ago:.1f} hours ago\n\n"
        
        file_list_text += "Select which file to delete:"
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=file_list_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return
    
    state.waiting_for_filename = True
    state.last_deleted_file = None
    
    keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="delete_cancel")]]
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="🗑️ **Delete File Content**\n\n"
             "Please send me the filename you want to delete content from.\n"
             "Example: `Sudan WhatsApp.txt`\n\n"
             "Click Cancel to cancel this operation.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def process_file_upload(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE,
                             chat_id: int, message_thread_id: int,
                             filename: str, size: int, content: str, operation: str,
                             requires_intervals: bool, chunks: List[str] = None,
                             parts: List[str] = None, is_duplicate: bool = False):
    """Process file upload and add to queue"""
    state = await get_user_state_safe(chat_id)
    
    # Generate tracking ID
    tracking_id = update_file_history(chat_id, filename, size, 'queued')
    
    if requires_intervals:
        file_info = {
            'name': filename,
            'tracking_id': tracking_id,
            'content': content,
            'parts': parts,
            'size': size,
            'operation': operation,
            'requires_intervals': True,
            'message_thread_id': message_thread_id
        }
    else:
        file_info = {
            'name': filename,
            'tracking_id': tracking_id,
            'content': content,
            'chunks': chunks,
            'size': size,
            'operation': operation,
            'requires_intervals': False,
            'message_thread_id': message_thread_id
        }
    
    # Thread-safe queue addition
    queue_size = await add_to_queue_safe(state, file_info)
    queue_position = get_queue_position_safe(state)
    
    # Check if this should start processing
    should_start_processing = not state.processing and queue_size == 1
    
    if should_start_processing:
        if 'chunks' in file_info:
            parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
            notification = (
                f"✅ File accepted: `{filename}`\n"
                f"Size: {size:,} characters{parts_info}\n"
                f"Operation: {operation.capitalize()}\n\n"
                f"🟢 Starting Your Task"
            )
        else:
            parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
            notification = (
                f"✅ File accepted: `{filename}`\n"
                f"Size: {size:,} characters{parts_info}\n"
                f"Operation: {operation.capitalize()}\n\n"
                f"🟢 Starting Your Task"
            )
        
        if is_duplicate:
            notification += "\n(Note: Duplicate - reprocessing confirmed)"
        
        sent_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=notification,
            disable_notification=True,
            parse_mode='Markdown'
        )
        file_notification_mapping[tracking_id] = sent_msg.message_id
        
        if not state.processing:
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    else:
        if 'chunks' in file_info:
            parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
        else:
            parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
        
        notification = (
            f"✅ File queued: `{filename}`\n"
            f"Size: {size:,} characters{parts_info}\n"
            f"Operation: {operation.capitalize()}\n"
            f"Position in queue: {queue_position}\n"
            f"Queue interval: {QUEUE_INTERVAL // 60} minutes between files\n\n"
        )
        
        if is_duplicate:
            notification += "(Note: Duplicate - reprocessing confirmed)"
        
        sent_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=notification,
            disable_notification=True,
            parse_mode='Markdown'
        )
        file_notification_mapping[tracking_id] = sent_msg.message_id

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    state = await get_user_state_safe(chat_id)
    
    # Check if waiting for filename (deletion flow)
    if state.waiting_for_filename:
        message_thread_id = update.effective_message.message_thread_id
        filename = update.message.text.strip()
        
        if filename.lower() == 'cancel':
            state.waiting_for_filename = False
            state.last_deleted_file = None
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ **Operation cancelled**",
                parse_mode='Markdown'
            )
            return
        
        state.waiting_for_filename = False
        state.last_deleted_file = filename
        
        # Find files with this name
        files = find_files_by_name(chat_id, filename)
        
        if not files:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **File not found**\n\n"
                     f"No record found for `{filename}` in your history.\n\n"
                     f"Use /delfilecontent to try again with a different filename.",
                parse_mode='Markdown'
            )
            return
        
        # If only one file found, delete it directly
        if len(files) == 1:
            await delete_file_content(chat_id, message_thread_id, files[0], context)
            return
        
        # Multiple files found - show selection
        keyboard = []
        for i, file_entry in enumerate(files):
            time_diff = datetime.now(UTC_PLUS_1) - file_entry.timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            button_text = f"File {i+1} ({file_entry.size:,} chars) - {hours_ago:.1f}h ago"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"delete_file_{i}")])
        
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="delete_cancel")])
        
        # Store files in context for callback
        context.user_data['delete_files'] = files
        
        file_list_text = "⚠️ **Multiple Files Found**\n\n"
        file_list_text += f"Found {len(files)} files named `{filename}`:\n\n"
        
        for i, file_entry in enumerate(files, 1):
            time_diff = datetime.now(UTC_PLUS_1) - file_entry.timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            file_list_text += f"{i}. `{file_entry.filename}`\n"
            file_list_text += f"   Size: {file_entry.size:,} chars\n"
            file_list_text += f"   Status: {file_entry.status.capitalize()}\n"
            file_list_text += f"   Age: {hours_ago:.1f} hours ago\n\n"
        
        file_list_text += "Select which file to delete:"
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=file_list_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return
    
    # Check if admin is in preview mode
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
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="❌ **Invalid file**\nPlease upload a valid TXT or CSV file.",
            parse_mode='Markdown'
        )
        return
    
    if not is_supported_file(file_name):
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Unsupported file type**\n"
            f"Please upload only TXT or CSV files.",
            parse_mode='Markdown'
        )
        return
    
    # Check queue size thread-safely
    queue_position = get_queue_position_safe(state)
    if queue_position >= MAX_QUEUE_SIZE:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Queue is full!**\n"
            f"Maximum {MAX_QUEUE_SIZE} files allowed.\n"
            "Please wait for current files to be processed.",
            parse_mode='Markdown'
        )
        return
    
    try:
        file = await context.bot.get_file(doc.file_id)
        
        try:
            file_bytes = await asyncio.wait_for(
                file.download_as_bytearray(),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **Download timeout**\n"
                     f"File `{file_name}` is too large or download failed.\n"
                     f"Please try again with a smaller file.",
                parse_mode='Markdown'
            )
            return
        except Exception as e:
            logger.error(f"File download error: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **Download failed**\n"
                     f"Could not download file `{file_name}`.\n"
                     f"Please try again.",
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
        
        # Check for duplicate file (same name AND size within 72 hours)
        duplicate_entry = find_duplicate_file(chat_id, file_name, content_size)
        
        if duplicate_entry and not duplicate_entry.is_expired():
            # Duplicate detected - ask for confirmation
            if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
                chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
                await handle_duplicate_file(update, context, chat_id, message_thread_id,
                                          file_name, content_size, content, state.operation,
                                          False, chunks=chunks, duplicate_entry=duplicate_entry)
            else:
                parts = split_large_content(content, CONTENT_LIMIT_FOR_INTERVALS)
                await handle_duplicate_file(update, context, chat_id, message_thread_id,
                                          file_name, content_size, content, state.operation,
                                          True, parts=parts, duplicate_entry=duplicate_entry)
            return
        
        # No duplicate or duplicate expired - process normally
        if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
            chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
            await process_file_upload(update, context, chat_id, message_thread_id,
                                     file_name, content_size, content, state.operation,
                                     False, chunks=chunks)
        else:
            parts = split_large_content(content, CONTENT_LIMIT_FOR_INTERVALS)
            await process_file_upload(update, context, chat_id, message_thread_id,
                                     file_name, content_size, content, state.operation,
                                     True, parts=parts)
        
    except asyncio.TimeoutError:
        logger.error(f"File processing timeout for {file_name}")
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Processing timeout**\n"
                 f"File `{file_name}` is too large to process.\n"
                 f"Please try with a smaller file.",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"File processing error: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ Error processing file\n{str(e)[:200]}",
            parse_mode='Markdown'
        )

# ==================== PERIODIC CLEANUP ====================
async def periodic_cleanup_task():
    """Periodically clean up old messages and file history"""
    while True:
        try:
            cleanup_old_messages()
            cleanup_old_file_history()
            logger.info("Periodic cleanup completed")
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

async def health_handler(request):
    return web.Response(
        text=json.dumps({
            "status": "running",
            "service": "filex-bot",
            "active_users": len(user_states),
            "allowed_ids_count": len(ALLOWED_IDS),
            "admin_ids_count": len(ADMIN_IDS),
            "tracked_messages": len(message_tracking),
            "file_history_entries": len(file_history_db),
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
