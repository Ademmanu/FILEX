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
MESSAGE_DELAY = 0.5  # Added: Delay between messages in seconds
QUEUE_INTERVAL = 2 * 60  # Added: 2-minute interval between files in seconds
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

file_history = defaultdict(list)
file_message_mapping = {}
file_notification_mapping = {}
file_other_notifications = {}

# ==================== ENHANCED MESSAGE TRACKING SYSTEM ====================
class MessageEntry:
    """Stores complete message text and extracted words"""
    def __init__(self, chat_id: int, message_id: int, full_text: str, words: List[str], timestamp: datetime):
        self.chat_id = chat_id
        self.message_id = message_id
        self.full_text = full_text
        self.words = words  # All words in the message
        self.first_two_words = ' '.join(words[:2]) if len(words) >= 2 else (' '.join(words) if words else "")
        self.timestamp = timestamp
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"MessageEntry(chat={self.chat_id}, words={self.words[:5]}, time={self.timestamp.strftime('%H:%M:%S')})"

# Global storage for message tracking - NO MAX LIMIT
message_tracking: List[MessageEntry] = []
admin_preview_mode: Set[int] = set()  # Track which admins are in preview mode

# ==================== ADMIN FUNCTIONS ====================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def check_admin_authorization(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is admin"""
    user_id = update.effective_user.id
    
    if is_admin(user_id):
        return True
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="⛔ **Admin Access Required**\n\nThis command is for administrators only."
    )
    return False

# ==================== ENHANCED MESSAGE TRACKING FUNCTIONS ====================
def extract_all_words(text: str) -> List[str]:
    """Extract all words from text, handling markdown and special chars"""
    if not text:
        return []
    
    # Remove markdown formatting but keep normal text
    clean_text = text
    # Remove URLs
    clean_text = re.sub(r'https?://\S+', ' ', clean_text)
    # Remove non-alphanumeric characters except spaces
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Split into words
    return clean_text.split()

def track_message(chat_id: int, message_id: int, message_text: str):
    """Track all words of sent message"""
    try:
        if not message_text or len(message_text.strip()) < 2:
            return
        
        words = extract_all_words(message_text)
        if not words:
            return
        
        # Remove expired entries first
        cleanup_old_messages()
        
        # Add new entry
        entry = MessageEntry(
            chat_id=chat_id,
            message_id=message_id,
            full_text=message_text,
            words=words,
            timestamp=datetime.now(UTC_PLUS_1)
        )
        
        message_tracking.append(entry)
        
        logger.debug(f"Tracked message with {len(words)} words: {words[:5]}...")
        
    except Exception as e:
        logger.error(f"Error tracking message: {e}")

def cleanup_old_messages():
    """Remove messages older than 72 hours"""
    global message_tracking
    
    if not message_tracking:
        return
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_count = len(message_tracking)
    
    # Filter out expired messages
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
            'unique_words': 0,
            'total_words': 0
        }
    
    oldest = min(entry.timestamp for entry in message_tracking)
    newest = max(entry.timestamp for entry in message_tracking)
    unique_word_pairs = len(set(entry.first_two_words for entry in message_tracking))
    total_words = sum(len(entry.words) for entry in message_tracking)
    
    return {
        'total': len(message_tracking),
        'oldest': oldest,
        'newest': newest,
        'unique_word_pairs': unique_word_pairs,
        'total_words': total_words
    }

# ==================== ENHANCED PREVIEW PROCESSING FUNCTIONS ====================
def extract_preview_sections(text: str) -> List[str]:
    """Extract all 'Preview:' sections from report text - exact matching"""
    preview_sections = []
    
    # Pattern to match "Preview: " followed by content until end of line
    pattern = r'📝\s*[Pp]review:\s*(.+?)(?=\n|\r|$)'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        # Clean up the match - remove trailing whitespace
        preview_text = match.strip()
        if preview_text:
            preview_sections.append(preview_text)
    
    # If no matches with emoji format, try without emoji
    if not preview_sections:
        pattern2 = r'[Pp]review:\s*(.+?)(?=\n|\r|$)'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for match in matches2:
            preview_text = match.strip()
            if preview_text:
                preview_sections.append(preview_text)
    
    return preview_sections

def check_preview_against_database(preview_texts: List[str]) -> Tuple[List[Tuple[str, MessageEntry]], List[Tuple[str, Dict]], List[str]]:
    """
    Check preview texts against tracked messages
    Returns: (full_matches, partial_matches, non_matches)
    
    full_matches: (preview_text, message_entry) - exact first two words match
    partial_matches: (preview_text, match_info) - partial word matches
    non_matches: preview_text - no matches found
    """
    cleanup_old_messages()
    
    if not message_tracking:
        return ([], [], preview_texts)
    
    full_matches = []
    partial_matches = []
    non_matches = []
    
    for preview in preview_texts:
        # Extract words from preview
        preview_words = extract_all_words(preview)
        if not preview_words:
            non_matches.append(preview)
            continue
        
        # Check for full match (first two words exactly match)
        preview_first_two = ' '.join(preview_words[:2]) if len(preview_words) >= 2 else preview_words[0]
        full_match_found = False
        
        for entry in message_tracking:
            if entry.first_two_words == preview_first_two:
                full_matches.append((preview, entry))
                full_match_found = True
                break
        
        if full_match_found:
            continue
        
        # Check for partial matches (individual words found in any message)
        partial_match_info = []
        matched_preview_words = []
        word_positions = {}
        
        for preview_word in preview_words:
            word_found = False
            for entry in message_tracking:
                if preview_word in entry.words:
                    word_found = True
                    position = entry.words.index(preview_word) + 1  # 1-based index
                    matched_preview_words.append(preview_word)
                    word_positions[preview_word] = position
                    # Store entry for reference
                    if 'entry' not in partial_match_info:
                        partial_match_info.append(('entry', entry))
                    break
            
            if word_found:
                break  # Stop after finding first match for this preview
        
        if matched_preview_words:
            match_details = {
                'matched_words': matched_preview_words,
                'word_positions': word_positions,
                'message_entry': partial_match_info[0][1] if partial_match_info else None
            }
            partial_matches.append((preview, match_details))
        else:
            non_matches.append(preview)
    
    return (full_matches, partial_matches, non_matches)

def format_preview_report_enhanced(full_matches: List[Tuple[str, MessageEntry]], 
                                  partial_matches: List[Tuple[str, Dict]], 
                                  non_matches: List[str], 
                                  total_previews: int) -> str:
    """Format the enhanced preview report with partial matches"""
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Preview Analysis Report**")
    report_lines.append("─" * 35)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    full_match_count = len(full_matches)
    partial_match_count = len(partial_matches)
    non_match_count = len(non_matches)
    
    full_percent = (full_match_count / total_previews * 100) if total_previews > 0 else 0
    partial_percent = (partial_match_count / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (non_match_count / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Full matches: {full_match_count} ({full_percent:.1f}%)")
    report_lines.append(f"• Partial matches: {partial_match_count} ({partial_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {non_match_count} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if full_matches:
        report_lines.append("✅ **Full matches found in database:**")
        for i, (preview, entry) in enumerate(full_matches, 1):
            report_lines.append(f"{i}. {preview}")
    
    if partial_matches:
        if full_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Partial matches found:**")
        for i, (preview, match_info) in enumerate(partial_matches, 1):
            report_lines.append(f"{i}. Preview: {preview}")
            matched_words_str = ", ".join(match_info['matched_words'])
            positions = [str(match_info['word_positions'][word]) for word in match_info['matched_words']]
            positions_str = ", ".join(positions)
            report_lines.append(f"   Matched words: {matched_words_str}")
            report_lines.append(f"   Word positions: {positions_str}")
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

# ==================== DUPLICATE FILE DETECTION ====================
def check_duplicate_file(chat_id: int, filename: str, file_size: int) -> Tuple[bool, Optional[Dict]]:
    """Check if same file (name + size) was posted in last 72 hours"""
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    
    for entry in file_history.get(chat_id, []):
        if (entry['filename'] == filename and 
            entry.get('size') == file_size and
            entry['timestamp'] >= cutoff and
            entry['status'] in ['completed', 'running']):
            return True, entry
    
    return False, None

# ==================== ENHANCED ADMIN COMMANDS ====================
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
    
    # Calculate expiration time
    if stats['oldest']:
        expires_in = stats['oldest'] + timedelta(hours=72) - datetime.now(UTC_PLUS_1)
        expires_hours = max(0, expires_in.total_seconds() / 3600)
    else:
        expires_hours = 0
    
    message = f"📊 **Enhanced Message Tracking Statistics**\n\n"
    message += f"**Total tracked messages:** {stats['total']}\n"
    message += f"**Total words tracked:** {stats['total_words']}\n"
    message += f"**Unique word pairs:** {stats['unique_word_pairs']}\n"
    message += f"**Oldest entry:** {oldest_str}\n"
    message += f"**Newest entry:** {newest_str}\n"
    message += f"**Expires in:** {expires_hours:.1f}h\n"
    message += f"**Active preview sessions:** {len(admin_preview_mode)}\n\n"
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
    
    # Extract previews from the message
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
    
    # Extract previews using exact format matching
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
    
    # Check against database with enhanced matching
    full_matches, partial_matches, non_matches = check_preview_against_database(previews)
    total_previews = len(previews)
    
    # Format enhanced report
    report_text = format_preview_report_enhanced(full_matches, partial_matches, non_matches, total_previews)
    
    # Send the report
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=report_text,
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

def update_file_history(chat_id: int, filename: str, status: str, size: int = 0, parts_count: int = 0, messages_count: int = 0):
    """Update file history with size parameter"""
    file_history[chat_id] = [entry for entry in file_history.get(chat_id, []) 
                           if entry['filename'] != filename or entry.get('size') != size]
    
    entry = {
        'filename': filename,
        'size': size,
        'timestamp': datetime.now(UTC_PLUS_1),
        'status': status,
        'parts_count': parts_count,
        'messages_count': messages_count
    }
    file_history[chat_id].append(entry)
    
    if len(file_history[chat_id]) > 100:
        file_history[chat_id] = file_history[chat_id][-100:]

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
        self.awaiting_duplicate_confirmation = False
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
        self.awaiting_duplicate_confirmation = False
        self.duplicate_file_info = None
        
    def clear_queue(self):
        count = len(self.queue)
        self.queue.clear()
        self.cancel_requested = False
        self.processing_start_time = None
        self.awaiting_duplicate_confirmation = False
        self.duplicate_file_info = None
        return count
        
    def has_active_tasks(self) -> bool:
        return self.processing or len(self.queue) > 0
    
    def remove_task_by_name(self, filename: str, file_size: Optional[int] = None):
        """Remove task by filename and optionally by size"""
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
            if file_size is not None:
                if not (task.get('name') == filename and task.get('size') == file_size):
                    new_queue.append(task)
            else:
                if task.get('name') != filename:
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

async def send_telegram_message_safe(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                    message: str, message_thread_id: Optional[int] = None, 
                                    retries: int = 5, filename: Optional[str] = None,
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
            
            # Enhanced message tracking
            if sent_message and sent_message.text:
                track_message(chat_id, sent_message.message_id, sent_message.text)
            
            if filename and sent_message:
                if notification_type == 'content':
                    if filename not in file_message_mapping:
                        file_message_mapping[filename] = []
                    file_message_mapping[filename].append(sent_message.message_id)
                elif notification_type == 'notification':
                    file_notification_mapping[filename] = sent_message.message_id
            
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for chat {chat_id}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return False

async def track_other_notification(chat_id: int, filename: str, message_id: int):
    if filename not in file_other_notifications:
        file_other_notifications[filename] = []
    file_other_notifications[filename].append(message_id)

async def send_chunks_immediately(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                chunks: List[str], filename: str, message_thread_id: Optional[int] = None) -> bool:
    try:
        total_messages_sent = 0
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, filename=filename):
                total_messages_sent += 1
                
                # Added: Delay between messages to prevent rate limiting
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for `{filename}`")
                return False  # Stop on failure
        
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
            await track_other_notification(chat_id, filename, completion_msg.message_id)
            
            update_file_history(chat_id, filename, 'completed', messages_count=total_messages_sent)
            return True
        else:
            logger.error(f"No chunks sent for `{filename}`")
            return False
        
    except Exception as e:
        logger.error(f"Error in send_chunks_immediately: {e}")
        return False

async def send_large_content_part(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                part: str, part_num: int, total_parts: int, 
                                filename: str, message_thread_id: Optional[int] = None) -> int:
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
            await track_other_notification(chat_id, filename, part_header_msg.message_id)
            total_messages_in_part += 1
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, filename=filename):
                total_messages_in_part += 1
                
                # Added: Delay between messages to prevent rate limiting
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for part {part_num} of `{filename}`")
                return 0  # Return 0 on failure
        
        return total_messages_in_part  # Return actual count
        
    except Exception as e:
        logger.error(f"Error sending large content part: {e}")
        return 0  # Return 0 on error

async def send_with_intervals(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                            parts: List[str], filename: str, state: UserState, 
                            message_thread_id: Optional[int] = None) -> bool:
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
                chat_id, context, part, i, total_parts, filename, message_thread_id
            )
            
            if not messages_in_part:  # Check for 0 (failure) instead of <= 0
                logger.error(f"Failed to send part {i} of `{filename}`")
                return False
            
            total_messages_sent += messages_in_part  # Accumulate actual messages
            
            state.last_send = datetime.now(UTC_PLUS_1)
            
            if i < total_parts:
                await asyncio.sleep(SEND_INTERVAL * 60)
        
        # Use actual messages count, not parts count
        completion_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"✅ Completed: `{filename}`\n📊 Sent {total_parts} part{'s' if total_parts > 1 else ''} ({total_messages_sent} messages total)",
            disable_notification=True,
            parse_mode='Markdown'
        )
        await track_other_notification(chat_id, filename, completion_msg.message_id)
        
        update_file_history(chat_id, filename, 'completed', parts_count=total_parts, messages_count=total_messages_sent)
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def cleanup_completed_file(filename: str, chat_id: int):
    if filename in file_message_mapping:
        del file_message_mapping[filename]
    if filename in file_notification_mapping:
        del file_notification_mapping[filename]
    if filename in file_other_notifications:
        del file_other_notifications[filename]
    
    logger.info(f"Cleaned up tracking for completed file: {filename} in chat {chat_id}")

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
            file_message_thread_id = file_info.get('message_thread_id', message_thread_id)
            
            if file_info.get('requires_intervals', False):
                update_file_history(chat_id, filename, 'running', size=file_info.get('size', 0), parts_count=len(file_info['parts']))
            else:
                update_file_history(chat_id, filename, 'running', size=file_info.get('size', 0), messages_count=len(file_info['chunks']))
            
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
                    await track_other_notification(chat_id, filename, sending_msg.message_id)
                else:
                    sending_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"📤 Sending: `{filename}`\n"
                             f"📊 Total messages: {len(file_info['chunks'])}",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    await track_other_notification(chat_id, filename, sending_msg.message_id)
            
            success = False
            if file_info.get('requires_intervals', False):
                success = await send_with_intervals(
                    chat_id, context, 
                    file_info['parts'], 
                    filename,
                    state,
                    file_message_thread_id
                )
            else:
                success = await send_chunks_immediately(
                    chat_id, context,
                    file_info['chunks'],
                    filename,
                    file_message_thread_id
                )
            
            # Always remove the file from queue after processing (success or failure)
            if state.queue and state.queue[0]['name'] == filename:
                processed_file = state.queue.popleft()
                processed_filename = processed_file['name']
                
                if success and not state.cancel_requested:
                    logger.info(f"Successfully processed `{processed_filename}` for chat {chat_id}")
                    await cleanup_completed_file(processed_filename, chat_id)
                else:
                    logger.error(f"Failed to process `{processed_filename}` for chat {chat_id}")
                    failed_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"❌ Failed to send: `{processed_filename}`\nPlease try uploading again.",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    await track_other_notification(chat_id, processed_filename, failed_msg.message_id)
            
            state.current_parts = []
            state.current_index = 0
            
            # Wait 2 minutes before processing next file (if any remain)
            if state.queue and not state.cancel_requested:
                next_file = state.queue[0]['name']
                next_file_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=f"⏰ **Queue Interval**\n\n"
                         f"Next file `{next_file}` will start in 2 minutes...",
                    parse_mode='Markdown'
                )
                
                # Wait for QUEUE_INTERVAL seconds
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
            await track_other_notification(chat_id, current_file, error_msg.message_id)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

# ==================== ENHANCED FILE HANDLING WITH DUPLICATE DETECTION ====================
async def handle_duplicate_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle duplicate file confirmation buttons"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    if not is_authorized(user_id, chat_id):
        return
    
    state = await get_user_state_safe(chat_id)
    
    if not state.awaiting_duplicate_confirmation:
        return
    
    action = query.data
    
    if action == 'duplicate_yes':
        # Process the duplicate file
        file_info = state.duplicate_file_info
        if file_info:
            # Remove the confirmation message
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
            except:
                pass
            
            # Add to queue and start processing
            queue_size = await add_to_queue_safe(state, file_info)
            queue_position = get_queue_position_safe(state)
            
            filename = file_info['name']
            content_size = file_info['size']
            
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
            
            notification = (
                f"✅ File accepted: `{filename}`\n"
                f"Size: {content_size:,} characters{parts_info}\n"
                f"Operation: {state.operation.capitalize()}\n\n"
                f"🟢 Starting Your Task"
            )
            
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=query.message.message_thread_id,
                text=notification,
                disable_notification=True,
                parse_mode='Markdown'
            )
            file_notification_mapping[filename] = sent_msg.message_id
            
            if not state.processing:
                state.processing_task = asyncio.create_task(process_queue(chat_id, context, query.message.message_thread_id))
    
    elif action == 'duplicate_no':
        # Cancel the duplicate file
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
        except:
            pass
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=query.message.message_thread_id,
            text=f"❌ File `{state.duplicate_file_info['name']}` cancelled.\n\n"
                 f"Upload a different file to continue.",
            parse_mode='Markdown'
        )
    
    # Reset duplicate confirmation state
    state.awaiting_duplicate_confirmation = False
    state.duplicate_file_info = None

# ==================== ENHANCED DELETION FUNCTIONS ====================
async def handle_deletion_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle deletion confirmation buttons"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    if not is_authorized(user_id, chat_id):
        return
    
    state = await get_user_state_safe(chat_id)
    message_thread_id = query.message.message_thread_id
    
    if query.data == 'delete_cancel':
        # Cancel deletion
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
        except:
            pass
        
        state.waiting_for_filename = False
        state.last_deleted_file = None
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="❌ **Deletion cancelled**",
            parse_mode='Markdown'
        )
        return
    
    elif query.data.startswith('delete_file_'):
        # Parse which file to delete
        parts = query.data.split('_')
        if len(parts) >= 3:
            try:
                index = int(parts[2]) - 1  # Convert to 0-based index
                
                # Get the file entries for this chat
                file_entries = [entry for entry in file_history.get(chat_id, []) 
                              if entry['filename'] == state.last_deleted_file]
                
                if 0 <= index < len(file_entries):
                    file_to_delete = file_entries[index]
                    
                    # Delete the selected file
                    await delete_specific_file(chat_id, message_thread_id, file_to_delete, context, state, query.message.message_id)
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text="❌ **Invalid selection**",
                        parse_mode='Markdown'
                    )
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing deletion index: {e}")
                await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text="❌ **Error processing selection**",
                    parse_mode='Markdown'
                )

async def delete_specific_file(chat_id: int, message_thread_id: Optional[int], file_entry: Dict,
                               context: ContextTypes.DEFAULT_TYPE, state: UserState, query_message_id: int):
    """Delete a specific file entry"""
    try:
        filename = file_entry['filename']
        file_size = file_entry.get('size', 0)
        
        # Try to delete the confirmation message
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=query_message_id)
        except:
            pass
        
        # Check if currently processing
        is_currently_processing = False
        if state.queue and state.queue[0].get('name') == filename and state.queue[0].get('size') == file_size and state.processing:
            is_currently_processing = True
            state.cancel_current_task()
            if state.queue:
                state.queue.popleft()
        
        messages_to_delete = 0
        deleted_messages = []
        
        # Delete content messages
        if filename in file_message_mapping:
            for msg_id in file_message_mapping[filename]:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=msg_id
                    )
                    messages_to_delete += 1
                    deleted_messages.append(msg_id)
                    logger.info(f"Deleted content message {msg_id} for file {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete content message {msg_id}: {e}")
        
        # Delete other notifications
        if filename in file_other_notifications:
            for msg_id in file_other_notifications[filename]:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=msg_id
                    )
                    messages_to_delete += 1
                    deleted_messages.append(msg_id)
                    logger.info(f"Deleted other notification message {msg_id} for file {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete other notification message {msg_id}: {e}")
        
        # Edit notification message if it exists
        if filename in file_notification_mapping:
            notification_msg_id = file_notification_mapping[filename]
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=notification_msg_id,
                    text=f"🗑️ **File Content Deleted**\n\n"
                         f"File: `{filename}`\n"
                         f"Size: {file_size:,} chars\n"
                         f"Messages deleted: {messages_to_delete}\n"
                         f"All content from this file has been removed.",
                    parse_mode='Markdown'
                )
                logger.info(f"Edited notification for {filename}")
            except Exception as e:
                logger.error(f"Failed to edit notification message: {e}")
        
        # Update history and remove from queue
        update_file_history(chat_id, filename, 'deleted', size=file_size)
        state.remove_task_by_name(filename, file_size)
        
        # Clean up tracking dictionaries
        await cleanup_file_tracking(filename)
        
        # Send confirmation
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🗑️ `{filename}` content deleted\n"
                 f"Size: {file_size:,} chars\n"
                 f"Messages removed: {len(deleted_messages)}",
            parse_mode='Markdown'
        )
        
        state.waiting_for_filename = False
        state.last_deleted_file = None
        
        # Handle queue continuation
        if is_currently_processing and state.queue and not state.processing:
            next_file = state.queue[0].get('name', 'Unknown')
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"🔄 **Moving to next task**\n\n"
                     f"Starting next task: `{next_file}`",
                parse_mode='Markdown'
            )
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
        elif is_currently_processing and not state.queue:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="🏁 **Processing stopped**\n\n"
                     "No more tasks in queue.",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"Error in delete_specific_file: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Error deleting file**\n\n{str(e)[:200]}",
            parse_mode='Markdown'
        )

async def cleanup_file_tracking(filename: str):
    """Clean up tracking for a file (partial cleanup)"""
    # Note: We don't fully delete from dictionaries as other instances might exist
    # The dictionaries will be cleaned up when all instances are deleted
    pass

async def process_file_deletion(chat_id: int, message_thread_id: Optional[int], filename: str, 
                                context: ContextTypes.DEFAULT_TYPE, state: UserState):
    """Enhanced file deletion with duplicate filename handling"""
    if filename.lower() == 'cancel':
        # Handle text cancellation (for backward compatibility)
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="❌ **Operation cancelled**",
            parse_mode='Markdown'
        )
        return
    
    # Store filename for reference
    state.last_deleted_file = filename
    
    # Find all files with this name in history
    file_entries = [entry for entry in file_history.get(chat_id, []) 
                   if entry['filename'] == filename]
    
    if not file_entries:
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **File not found**\n\n"
                 f"No record found for `{filename}` in your history.\n\n"
                 f"Use /delfilecontent to try again with a different filename.",
            parse_mode='Markdown'
        )
        return
    
    if len(file_entries) == 1:
        # Only one file found - delete it directly
        await delete_specific_file(chat_id, message_thread_id, file_entries[0], context, state, None)
    else:
        # Multiple files found - ask which one to delete
        keyboard = []
        
        for i, entry in enumerate(file_entries, 1):
            time_str = entry['timestamp'].strftime('%H:%M:%S')
            size_str = f"{entry.get('size', 0):,} chars"
            status = entry['status'].capitalize()
            
            button_text = f"{i}. {size_str} | {time_str} | {status}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"delete_file_{i}")])
        
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="delete_cancel")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🗑️ **Multiple files found**\n\n"
                 f"Found {len(file_entries)} files named `{filename}`:\n\n"
                 f"Select which one to delete:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

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
    
    # Check for duplicate confirmation
    if query.data in ['duplicate_yes', 'duplicate_no']:
        await handle_duplicate_confirmation(update, context)
        return
    
    # Check for deletion confirmation
    if query.data.startswith('delete_'):
        await handle_deletion_confirmation(update, context)
        return
    
    # Handle operation selection
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    operation = query.data
    
    operation_names = {
        'all': '✅ All content',
        'number': '🔢 Number only',
        'alphabet': '🔤 Alphabet only'
    }
    
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
    
    twelve_hours_ago = datetime.now(UTC_PLUS_1) - timedelta(hours=12)
    
    recent_files = [
        entry for entry in file_history.get(chat_id, [])
        if entry['timestamp'] >= twelve_hours_ago
    ]
    
    if not recent_files:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="📊 **Last 12 Hours Stats**\n\n"
                 "No files processed in the last 12 hours."
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
        stats_text += f"   Size: {entry.get('size', 0):,} chars\n"
        stats_text += f"   Status: {entry['status'].capitalize()}{count_info}\n"
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
            update_file_history(chat_id, current_file, 'paused', size=file_info.get('size', 0), parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, 'paused', size=file_info.get('size', 0), messages_count=len(file_info['chunks']))
    
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
            update_file_history(chat_id, current_file, 'running', size=file_info.get('size', 0), parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, 'running', size=file_info.get('size', 0), messages_count=len(file_info['chunks']))
    
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
    current_size = state.queue[0].get('size', 0) if state.queue else 0
    
    update_file_history(chat_id, current_file, 'skipped', size=current_size)
    
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
                update_file_history(chat_id, file_info['name'], 'cancelled', 
                                  size=file_info.get('size', 0), 
                                  parts_count=len(file_info.get('parts', [])))
            else:
                update_file_history(chat_id, file_info['name'], 'cancelled', 
                                  size=file_info.get('size', 0), 
                                  messages_count=len(file_info.get('chunks', [])))
    
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
    """Enhanced deletion command with inline buttons"""
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if state.waiting_for_filename:
        filename = update.message.text.strip()
        await process_file_deletion(chat_id, message_thread_id, filename, context, state)
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
    """Enhanced file handler with duplicate detection"""
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    state.waiting_for_filename = False
    state.last_deleted_file = None
    
    state.cancel_requested = False
    state.awaiting_duplicate_confirmation = False
    
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
        # Added timeout for file download to prevent "Timed out" error
        file = await context.bot.get_file(doc.file_id)
        
        # Download with timeout
        try:
            file_bytes = await asyncio.wait_for(
                file.download_as_bytearray(),
                timeout=30.0  # 30 second timeout for file download
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
        
        # Check for duplicate file
        is_duplicate, duplicate_entry = check_duplicate_file(chat_id, file_name, content_size)
        
        if is_duplicate and duplicate_entry:
            # Ask for confirmation
            state.awaiting_duplicate_confirmation = True
            state.duplicate_file_info = {
                'name': file_name,
                'content': content,
                'size': content_size,
                'operation': state.operation,
                'message_thread_id': message_thread_id
            }
            
            if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
                chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
                state.duplicate_file_info['chunks'] = chunks
                state.duplicate_file_info['requires_intervals'] = False
            else:
                parts = split_large_content(content, CONTENT_LIMIT_FOR_INTERVALS)
                state.duplicate_file_info['parts'] = parts
                state.duplicate_file_info['requires_intervals'] = True
            
            time_diff = datetime.now(UTC_PLUS_1) - duplicate_entry['timestamp']
            hours_ago = int(time_diff.total_seconds() // 3600)
            minutes_ago = int((time_diff.total_seconds() % 3600) // 60)
            
            time_ago_str = f"{hours_ago}h {minutes_ago}m" if hours_ago > 0 else f"{minutes_ago}m"
            
            keyboard = [
                [InlineKeyboardButton("✅ Yes, Post", callback_data="duplicate_yes")],
                [InlineKeyboardButton("❌ No, Don't", callback_data="duplicate_no")]
            ]
            
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"⚠️ **Duplicate File Detected**\n\n"
                     f"File `{file_name}` ({content_size:,} chars) was posted {time_ago_str} ago.\n\n"
                     f"Do you want to post it again?",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            return
        
        # Continue with normal processing
        if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
            chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
            file_info = {
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
                'name': file_name,
                'content': content,
                'parts': parts,
                'size': content_size,
                'operation': state.operation,
                'requires_intervals': True,
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
            
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=notification,
                disable_notification=True,
                parse_mode='Markdown'
            )
            file_notification_mapping[file_name] = sent_msg.message_id
            
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
            
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=notification,
                disable_notification=True,
                parse_mode='Markdown'
            )
            file_notification_mapping[file_name] = sent_msg.message_id
            
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
    
    # Add callback query handlers for all button types
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(all|number|alphabet)$'))
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(duplicate_yes|duplicate_no)$'))
    application.add_handler(CallbackQueryHandler(button_handler, pattern='^(delete_cancel|delete_file_\d+)$'))
    
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
