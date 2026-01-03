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
class EnhancedMessageEntry:
    """Stores full message content and metadata for tracking"""
    def __init__(self, chat_id: int, message_id: int, full_message: str, 
                 first_two_words: str, timestamp: datetime, filename: Optional[str] = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.full_message = full_message
        self.first_two_words = first_two_words
        self.timestamp = timestamp
        self.filename = filename
        self.words = self.extract_words(full_message)
    
    def extract_words(self, text: str) -> List[str]:
        """Extract all words from message text"""
        # Remove URLs and special formatting
        clean_text = re.sub(r'https?://\S+', ' ', text)
        # Replace multiple spaces with single space
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Split into words
        return clean_text.split()
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"EnhancedMessageEntry(chat={self.chat_id}, words={len(self.words)}, time={self.timestamp.strftime('%H:%M:%S')})"

# Global storage for enhanced message tracking
enhanced_message_tracking: List[EnhancedMessageEntry] = []
admin_preview_mode: Set[int] = set()

# ==================== DUPLICATE FILE TRACKING ====================
class FileHistoryEntry:
    """Track file upload history for duplicate detection"""
    def __init__(self, filename: str, size: int, chat_id: int, timestamp: datetime, status: str = "uploaded", 
                 instance_id: Optional[int] = None):
        self.filename = filename
        self.size = size
        self.chat_id = chat_id
        self.timestamp = timestamp
        self.status = status  # completed, cancelled, deleted, etc.
        self.instance_id = instance_id  # Unique ID for same filename+size
    
    def is_recent(self) -> bool:
        """Check if file was uploaded within last 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp <= timedelta(hours=72)
    
    def __repr__(self):
        return f"FileHistoryEntry({self.filename}, {self.size} bytes, {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, status={self.status}, instance={self.instance_id})"

file_upload_history: List[FileHistoryEntry] = []

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
def extract_first_two_words(text: str) -> str:
    """Extract first two words from text"""
    if not text or len(text.strip()) < 2:
        return ""
    
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

def track_enhanced_message(chat_id: int, message_id: int, message_text: str, filename: Optional[str] = None):
    """Track full message content for enhanced preview matching"""
    try:
        if not message_text:
            return
        
        first_two_words = extract_first_two_words(message_text)
        
        entry = EnhancedMessageEntry(
            chat_id=chat_id,
            message_id=message_id,
            full_message=message_text,
            first_two_words=first_two_words,
            timestamp=datetime.now(UTC_PLUS_1),
            filename=filename
        )
        
        enhanced_message_tracking.append(entry)
        
        logger.debug(f"Tracked enhanced message: {len(entry.words)} words, first_two='{first_two_words}'")
        
    except Exception as e:
        logger.error(f"Error tracking enhanced message: {e}")

def cleanup_old_messages_and_files():
    """Remove messages and file history older than 72 hours"""
    global enhanced_message_tracking, file_upload_history
    
    # Clean up old messages
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_msg_count = len(enhanced_message_tracking)
    initial_file_count = len(file_upload_history)
    
    # Filter out expired messages
    enhanced_message_tracking = [entry for entry in enhanced_message_tracking if not entry.is_expired()]
    
    # Filter out expired file history
    file_upload_history = [entry for entry in file_upload_history if entry.is_recent()]
    
    removed_msgs = initial_msg_count - len(enhanced_message_tracking)
    removed_files = initial_file_count - len(file_upload_history)
    
    if removed_msgs > 0 or removed_files > 0:
        logger.info(f"Cleaned up {removed_msgs} expired messages and {removed_files} expired file entries")
        logger.info(f"Current: {len(enhanced_message_tracking)} messages, {len(file_upload_history)} file entries")

def check_duplicate_file(filename: str, size: int, chat_id: int) -> Optional[FileHistoryEntry]:
    """Check if same file (name + size) was uploaded within last 72 hours"""
    cleanup_old_messages_and_files()
    
    for entry in file_upload_history:
        if (entry.filename == filename and 
            entry.size == size and 
            entry.chat_id == chat_id and 
            entry.is_recent() and
            entry.status != 'deleted'):
            return entry
    
    return None

def add_file_to_history(filename: str, size: int, chat_id: int, status: str = "uploaded") -> FileHistoryEntry:
    """Add file to upload history for duplicate detection with instance tracking"""
    # Count existing instances of same filename+size
    existing_instances = [
        entry for entry in file_upload_history 
        if entry.filename == filename and entry.size == size and entry.chat_id == chat_id
    ]
    
    # Create new instance ID
    instance_id = len(existing_instances) + 1
    
    entry = FileHistoryEntry(
        filename=filename,
        size=size,
        chat_id=chat_id,
        timestamp=datetime.now(UTC_PLUS_1),
        status=status,
        instance_id=instance_id
    )
    file_upload_history.append(entry)
    
    logger.info(f"Added file to history: {filename} ({size} bytes) - instance {instance_id}")
    return entry

def get_file_status_from_history(chat_id: int, filename: str, size: int) -> Optional[str]:
    """Get the status of a file from history"""
    cleanup_old_messages_and_files()
    
    for entry in file_upload_history:
        if (entry.filename == filename and 
            entry.size == size and 
            entry.chat_id == chat_id):
            return entry.status
    
    return None

# ==================== ENHANCED PREVIEW MATCHING FUNCTIONS ====================
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

def check_preview_against_enhanced_database(preview_texts: List[str]) -> Tuple[
    List[Dict[str, Any]],  # full_matches
    List[Dict[str, Any]],  # partial_matches
    List[str]              # non_matches
]:
    """
    Check preview texts against tracked messages with enhanced matching
    Returns: (full_matches, partial_matches, non_matches)
    """
    if not enhanced_message_tracking:
        return ([], [], preview_texts)
    
    full_matches = []
    partial_matches = []
    non_matches = []
    
    for preview in preview_texts:
        # Extract words from preview
        preview_words = [w.strip() for w in preview.split() if w.strip()]
        
        if not preview_words:
            non_matches.append(preview)
            continue
        
        # Check for full match (first two words exactly match a message's first two words)
        first_two_preview = " ".join(preview_words[:2]) if len(preview_words) >= 2 else preview_words[0]
        full_match_found = False
        
        for entry in enhanced_message_tracking:
            if entry.first_two_words == first_two_preview:
                full_matches.append({
                    'preview': preview,
                    'matched_entry': entry
                })
                full_match_found = True
                break
        
        if full_match_found:
            continue
        
        # Check for partial matches (any word matches any word in any message)
        partial_match_info = {
            'preview': preview,
            'matched_words': [],
            'word_positions': [],
            'source_messages': []
        }
        
        for entry in enhanced_message_tracking:
            for i, preview_word in enumerate(preview_words):
                if preview_word in entry.words:
                    if preview_word not in partial_match_info['matched_words']:
                        partial_match_info['matched_words'].append(preview_word)
                        # Find position in source message (1-indexed)
                        position = entry.words.index(preview_word) + 1 if preview_word in entry.words else 0
                        partial_match_info['word_positions'].append(position)
                        partial_match_info['source_messages'].append({
                            'message_id': entry.message_id,
                            'filename': entry.filename,
                            'position': position
                        })
        
        if partial_match_info['matched_words']:
            partial_matches.append(partial_match_info)
        else:
            non_matches.append(preview)
    
    return (full_matches, partial_matches, non_matches)

def format_enhanced_preview_report(full_matches: List[Dict[str, Any]], 
                                  partial_matches: List[Dict[str, Any]], 
                                  non_matches: List[str],
                                  total_previews: int) -> str:
    """Format the enhanced preview report"""
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Preview Analysis Report**")
    report_lines.append("─" * 35)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    full_match_percent = (len(full_matches) / total_previews * 100) if total_previews > 0 else 0
    partial_match_percent = (len(partial_matches) / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (len(non_matches) / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Full matches found: {len(full_matches)} ({full_match_percent:.1f}%)")
    report_lines.append(f"• Partial matches: {len(partial_matches)} ({partial_match_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {len(non_matches)} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if full_matches:
        report_lines.append("✅ **Full matches found in database:**")
        for i, match in enumerate(full_matches, 1):
            report_lines.append(f"{i}. {match['preview']}")
    
    if partial_matches:
        if full_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Partial matches found:**")
        for i, match_info in enumerate(partial_matches, 1):
            report_lines.append(f"{i}. Preview: {match_info['preview']}")
            if match_info['matched_words']:
                report_lines.append(f"   Matched words: {', '.join(match_info['matched_words'])}")
                report_lines.append(f"   Word positions: {', '.join(map(str, match_info['word_positions']))}")
    
    if non_matches:
        if full_matches or partial_matches:
            report_lines.append("")
        report_lines.append("❌ **Not found in database:**")
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
    
    cleanup_old_messages_and_files()
    
    if not enhanced_message_tracking:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.effective_message.message_thread_id,
            text="📊 **Enhanced Message Tracking Statistics**\n\n"
                 "No messages tracked yet.\n"
                 "Database is empty.",
            parse_mode='Markdown'
        )
        return
    
    oldest = min(entry.timestamp for entry in enhanced_message_tracking)
    newest = max(entry.timestamp for entry in enhanced_message_tracking)
    
    total_words = sum(len(entry.words) for entry in enhanced_message_tracking)
    unique_first_two = len(set(entry.first_two_words for entry in enhanced_message_tracking))
    
    oldest_str = oldest.strftime('%Y-%m-%d %H:%M:%S')
    newest_str = newest.strftime('%Y-%m-%d %H:%M:%S')
    
    if oldest:
        expires_in = oldest + timedelta(hours=72) - datetime.now(UTC_PLUS_1)
        expires_hours = max(0, expires_in.total_seconds() / 3600)
    else:
        expires_hours = 0
    
    message = f"📊 **Enhanced Message Tracking Statistics**\n\n"
    message += f"**Total tracked messages:** {len(enhanced_message_tracking)}\n"
    message += f"**Total words tracked:** {total_words}\n"
    message += f"**Unique first-two-word pairs:** {unique_first_two}\n"
    message += f"**Oldest entry:** {oldest_str}\n"
    message += f"**Newest entry:** {newest_str}\n"
    message += f"**Expires in:** {expires_hours:.1f}h\n"
    message += f"**File history entries:** {len(file_upload_history)}\n"
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
    
    # Check against enhanced database
    full_matches, partial_matches, non_matches = check_preview_against_enhanced_database(previews)
    total_previews = len(previews)
    
    # Format enhanced report
    report_text = format_enhanced_preview_report(full_matches, partial_matches, non_matches, total_previews)
    
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
            cleanup_old_messages_and_files()
            cleanup_stuck_tasks()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

def update_file_history(chat_id: int, filename: str, size: int, status: str, parts_count: int = 0, messages_count: int = 0):
    """Update file history - track by filename + size"""
    # Update local file_history (track by filename+size)
    unique_id = f"{filename}_{size}"
    
    # Remove old entry with same filename+size
    file_history[chat_id] = [entry for entry in file_history.get(chat_id, []) 
                           if entry.get('unique_id') != unique_id]
    
    entry = {
        'filename': filename,
        'size': size,
        'unique_id': unique_id,
        'timestamp': datetime.now(UTC_PLUS_1),
        'status': status,
        'parts_count': parts_count,
        'messages_count': messages_count
    }
    file_history[chat_id].append(entry)
    
    if len(file_history[chat_id]) > 100:
        file_history[chat_id] = file_history[chat_id][-100:]
    
    # Also update file_upload_history with status if we can find the file
    for upload_entry in file_upload_history:
        if (upload_entry.filename == filename and 
            upload_entry.size == size and
            upload_entry.chat_id == chat_id and 
            upload_entry.is_recent()):
            upload_entry.status = status
            break

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
        
        self.paused = False
        self.paused_at = None
        self.paused_progress = None
        
        # Duplicate file confirmation
        self.waiting_duplicate_confirmation = False
        self.pending_duplicate_file = None
        
        # Multi-file deletion state
        self.waiting_file_selection = False
        self.pending_delete_files = []
        self.pending_delete_filename = None
        
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
    
    def remove_task_by_name(self, filename: str):
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
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
                    update_file_history(chat_id, "Unknown", 0, "timeout_cancelled")
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
            
            # Track message for enhanced preview system
            if sent_message and sent_message.text:
                track_enhanced_message(chat_id, sent_message.message_id, sent_message.text, filename)
            
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
            await track_other_notification(chat_id, filename, completion_msg.message_id)
            
            # Get size from context or use default
            file_size = len(''.join(chunks)) if chunks else 0
            update_file_history(chat_id, filename, file_size, 'completed', messages_count=total_messages_sent)
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
                            parts: List[str], filename: str, state: UserState, 
                            message_thread_id: Optional[int] = None) -> bool:
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
                chat_id, context, part, i, total_parts, filename, message_thread_id
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
        await track_other_notification(chat_id, filename, completion_msg.message_id)
        
        # Get size from parts
        file_size = len(''.join(parts)) if parts else 0
        update_file_history(chat_id, filename, file_size, 'completed', parts_count=total_parts, messages_count=total_messages_sent)
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def cleanup_completed_file(filename: str, chat_id: int):
    """Clean up all tracking data for a completed file"""
    try:
        # Clean up file message mapping
        if filename in file_message_mapping:
            del file_message_mapping[filename]
        
        # Clean up notification mapping
        if filename in file_notification_mapping:
            del file_notification_mapping[filename]
        
        # Clean up other notifications
        if filename in file_other_notifications:
            del file_other_notifications[filename]
        
        logger.info(f"Cleaned up tracking for completed file: {filename} in chat {chat_id}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filename}: {e}")

async def delete_file_messages_completely(chat_id: int, filename: str, context: ContextTypes.DEFAULT_TYPE, 
                                         file_size: Optional[int] = None) -> int:
    """Completely delete all messages related to a file, return number deleted"""
    deleted_count = 0
    
    try:
        # Delete content messages
        if filename in file_message_mapping:
            for msg_id in file_message_mapping[filename]:
                try:
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=msg_id
                    )
                    deleted_count += 1
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
                    deleted_count += 1
                    logger.info(f"Deleted notification message {msg_id} for file {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete notification message {msg_id}: {e}")
        
        # Edit notification message if it exists
        if filename in file_notification_mapping:
            notification_msg_id = file_notification_mapping[filename]
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=notification_msg_id,
                    text=f"🗑️ **File Content Deleted**\n\n"
                         f"File: `{filename}`\n"
                         f"Messages deleted: {deleted_count}\n"
                         f"All content from this file has been removed.",
                    parse_mode='Markdown'
                )
                logger.info(f"Edited notification message for {filename}")
            except Exception as e:
                logger.error(f"Failed to edit notification message: {e}")
        
        # Clean up tracking data
        await cleanup_completed_file(filename, chat_id)
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting messages for file {filename}: {e}")
        return deleted_count

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
            
            # Always remove the file from queue after processing
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
            
            # Wait QUEUE_INTERVAL before processing next file
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
            await track_other_notification(chat_id, current_file, error_msg.message_id)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

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
    
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    # Handle duplicate file confirmation
    if state.waiting_duplicate_confirmation:
        if query.data == 'dup_yes':
            state.waiting_duplicate_confirmation = False
            if state.pending_duplicate_file:
                file_info = state.pending_duplicate_file
                state.pending_duplicate_file = None
                
                # Add to file history with "uploaded" status
                add_file_to_history(file_info['name'], file_info['size'], chat_id, "uploaded")
                
                # Add to queue
                queue_size = await add_to_queue_safe(state, file_info)
                queue_position = get_queue_position_safe(state)
                
                should_start_processing = not state.processing and queue_size == 1
                
                if should_start_processing:
                    notification = f"✅ **Duplicate file accepted**\n\nFile: `{file_info['name']}`\nSize: {file_info['size']:,} characters\n\n🟢 Starting Your Task"
                    
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=notification,
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    file_notification_mapping[file_info['name']] = sent_msg.message_id
                    
                    if not state.processing:
                        state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                else:
                    notification = f"✅ **Duplicate file queued**\n\nFile: `{file_info['name']}`\nSize: {file_info['size']:,} characters\nPosition in queue: {queue_position}"
                    
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=notification,
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    file_notification_mapping[file_info['name']] = sent_msg.message_id
                
                await query.edit_message_text(
                    text="✅ **Upload Confirmed**\n\nFile added to processing queue.",
                    parse_mode='Markdown'
                )
                return
                
        elif query.data == 'dup_no':
            state.waiting_duplicate_confirmation = False
            state.pending_duplicate_file = None
            
            await query.edit_message_text(
                text="❌ **Upload Cancelled**\n\nFile was not added to queue.\nYou can upload a different file.",
                parse_mode='Markdown'
            )
            return
    
    # Handle file selection for deletion (multiple files with same name)
    if state.waiting_file_selection:
        if query.data == 'del_cancel':
            state.waiting_file_selection = False
            state.pending_delete_files = []
            state.pending_delete_filename = None
            
            await query.edit_message_text(
                text="❌ **Operation cancelled**\n\nNo files were deleted.",
                parse_mode='Markdown'
            )
            return
        elif query.data.startswith('del_file_'):
            file_index = int(query.data.split('_')[2])
            if 0 <= file_index < len(state.pending_delete_files):
                selected_file = state.pending_delete_files[file_index]
                state.waiting_file_selection = False
                
                # Process deletion for the selected file
                deleted_count = await delete_file_messages_completely(
                    chat_id, 
                    selected_file['filename'], 
                    context,
                    selected_file.get('size')
                )
                
                # Update file history with "deleted" status
                update_file_history(chat_id, selected_file['filename'], selected_file.get('size', 0), 'deleted')
                
                # Update file_upload_history status
                for upload_entry in file_upload_history:
                    if (upload_entry.filename == selected_file['filename'] and 
                        upload_entry.chat_id == chat_id and 
                        abs((upload_entry.timestamp - selected_file['timestamp']).total_seconds()) < 60):
                        upload_entry.status = 'deleted'
                        break
                
                # Remove from queue if present
                state.remove_task_by_name(selected_file['filename'])
                
                await query.edit_message_text(
                    text=f"🗑️ `{selected_file['filename']}` content deleted\nMessages removed: {deleted_count}",
                    parse_mode='Markdown'
                )
                
                # Check if we need to start processing next task
                if state.processing and state.queue and not state.processing:
                    next_file = state.queue[0].get('name', 'Unknown')
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=f"🔄 **Moving to next task**\n\nStarting next task: `{next_file}`",
                        parse_mode='Markdown'
                    )
                    state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                elif state.processing and not state.queue:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text="🏁 **Processing stopped**\n\nNo more tasks in queue.",
                        parse_mode='Markdown'
                    )
            return
    
    # Handle operation selection
    operation = query.data
    
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
        
        filename_display = entry['filename']
        if 'size' in entry and entry['size'] > 0:
            filename_display = f"{entry['filename']} ({entry['size']:,} bytes)"
        
        stats_text += f"{status_emoji} `{filename_display}`\n"
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
    current_size = state.queue[0].get('size', 0) if state.queue else 0
    
    update_file_history(chat_id, current_file, current_size, 'skipped')
    
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
                update_file_history(chat_id, file_info['name'], file_info.get('size', 0), 'cancelled', parts_count=len(file_info.get('parts', [])))
            else:
                update_file_history(chat_id, file_info['name'], file_info.get('size', 0), 'cancelled', messages_count=len(file_info.get('chunks', [])))
    
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
    
    # Check if already waiting for filename or file selection
    if state.waiting_for_filename or state.waiting_file_selection:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="⚠️ **Already in deletion process**\n\n"
                 "Please complete the current deletion operation first.\n"
                 "Or send 'cancel' to cancel the current operation.",
            parse_mode='Markdown'
        )
        return
    
    # Send inline button prompt
    keyboard = [[InlineKeyboardButton("🚫 Cancel", callback_data="del_cancel")]]
    
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
    
    state.waiting_for_filename = True
    state.last_deleted_file = None

async def handle_deletion_filename(chat_id: int, message_thread_id: Optional[int], filename: str, 
                                  context: ContextTypes.DEFAULT_TYPE, state: UserState):
    """Handle filename input for deletion - EXCLUDE already deleted files"""
    
    # Find all files with this name in history EXCEPT deleted ones
    matching_files = []
    for entry in file_history.get(chat_id, []):
        if entry['filename'] == filename and entry['status'] != 'deleted':
            matching_files.append({
                'filename': entry['filename'],
                'size': entry.get('size', 0),
                'timestamp': entry['timestamp'],
                'status': entry['status'],
                'parts_count': entry.get('parts_count', 0),
                'messages_count': entry.get('messages_count', 0),
                'unique_id': entry.get('unique_id', f"{filename}_{entry.get('size', 0)}")
            })
    
    if not matching_files:
        # Also check if file exists but is marked as deleted
        deleted_exists = any(
            entry['filename'] == filename and entry['status'] == 'deleted'
            for entry in file_history.get(chat_id, [])
        )
        
        if deleted_exists:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"🗑️ **File Already Deleted**\n\n"
                     f"File `{filename}` was already deleted previously.\n"
                     f"Use /delfilecontent to delete a different file.",
                parse_mode='Markdown'
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **File not found**\n\n"
                     f"No record found for `{filename}` in your history.\n\n"
                     f"Use /delfilecontent to try again with a different filename.",
                parse_mode='Markdown'
            )
        
        state.waiting_for_filename = False
        return
    
    # If only one file found, delete it directly
    if len(matching_files) == 1:
        file_info = matching_files[0]
        deleted_count = await delete_file_messages_completely(chat_id, file_info['filename'], context, file_info.get('size'))
        
        update_file_history(chat_id, file_info['filename'], file_info.get('size', 0), 'deleted')
        
        # Update file_upload_history status
        for upload_entry in file_upload_history:
            if (upload_entry.filename == file_info['filename'] and 
                upload_entry.size == file_info.get('size', 0) and
                upload_entry.chat_id == chat_id and 
                abs((upload_entry.timestamp - file_info['timestamp']).total_seconds()) < 60):
                upload_entry.status = 'deleted'
                break
        
        state.remove_task_by_name(file_info['filename'])
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🗑️ `{file_info['filename']}` content deleted\nMessages removed: {deleted_count}",
            parse_mode='Markdown'
        )
        
        # Check if we need to start processing next task
        if state.processing and state.queue and not state.processing:
            next_file = state.queue[0].get('name', 'Unknown')
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"🔄 **Moving to next task**\n\nStarting next task: `{next_file}`",
                parse_mode='Markdown'
            )
            state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
        elif state.processing and not state.queue:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="🏁 **Processing stopped**\n\nNo more tasks in queue.",
                parse_mode='Markdown'
            )
        
        state.waiting_for_filename = False
        return
    
    # Multiple files found - show inline buttons for selection
    state.waiting_file_selection = True
    state.waiting_for_filename = False
    state.pending_delete_files = matching_files
    state.pending_delete_filename = filename
    
    # Create message with inline buttons - SIMPLE (File 1), (File 2), etc.
    keyboard = []
    for i, file_info in enumerate(matching_files):
        # Simple button label: (File 1), (File 2), etc.
        button_text = f"(File {i+1})"
        keyboard.append([
            InlineKeyboardButton(button_text, callback_data=f"del_file_{i}")
        ])
    
    keyboard.append([InlineKeyboardButton("🚫 Cancel", callback_data="del_cancel")])
    
    # Format message
    message = f"📋 **Multiple Files Found**\n\n"
    message += f"Found {len(matching_files)} active files named `{filename}`:\n\n"
    
    for i, file_info in enumerate(matching_files):
        time_str = file_info['timestamp'].strftime('%H:%M:%S')
        status_info = file_info['status'].capitalize()
        count_info = ""
        
        if file_info['status'] == 'completed':
            if file_info.get('parts_count', 0) > 0:
                count_info = f" ({file_info['parts_count']} part{'s' if file_info['parts_count'] > 1 else ''}"
                if file_info.get('messages_count', 0) > 0:
                    count_info += f", {file_info['messages_count']} messages"
                count_info += ")"
            elif file_info.get('messages_count', 0) > 0:
                count_info = f" ({file_info['messages_count']} message{'s' if file_info['messages_count'] > 1 else ''})"
        
        size_info = f" ({file_info.get('size', 0):,} bytes)" if file_info.get('size', 0) > 0 else ""
        message += f"{i+1}. {file_info['filename']}{size_info} - {time_str} ({status_info}{count_info})\n"
    
    message += "\nWhich file do you want to delete?"
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=message,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    state = await get_user_state_safe(chat_id)
    
    # Check if admin is in preview mode
    user_id = update.effective_user.id
    if user_id in admin_preview_mode:
        await handle_admin_preview_message(update, context)
        return
    
    # Check if waiting for filename (deletion flow)
    if state.waiting_for_filename:
        message_thread_id = update.effective_message.message_thread_id
        filename = update.message.text.strip()
        
        # Handle cancel text (backward compatibility)
        if filename.lower() == 'cancel':
            state.waiting_for_filename = False
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ **Operation cancelled**",
                parse_mode='Markdown'
            )
            return
        
        await handle_deletion_filename(chat_id, message_thread_id, filename, context, state)
        return

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    # Reset deletion states
    state.waiting_for_filename = False
    state.waiting_file_selection = False
    state.pending_delete_files = []
    state.pending_delete_filename = None
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
    
    # Check queue size
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
        # Download file
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
        
        # Check for duplicate file
        duplicate_entry = check_duplicate_file(file_name, content_size, chat_id)
        
        if duplicate_entry and duplicate_entry.is_recent():
            # Get status of the old file
            old_status = duplicate_entry.status
            status_display = old_status.capitalize()
            
            # Get status emoji
            status_emoji = {
                'completed': '✅',
                'uploaded': '📤',
                'cancelled': '🚫',
                'deleted': '🗑️',
                'skipped': '⏭️',
                'running': '▶️',
                'paused': '⏸️',
                'timeout_cancelled': '⏱️'
            }.get(old_status, '📝')
            
            time_diff = datetime.now(UTC_PLUS_1) - duplicate_entry.timestamp
            hours_ago = time_diff.total_seconds() / 3600
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Yes, Post", callback_data="dup_yes"),
                    InlineKeyboardButton("❌ No, Don't", callback_data="dup_no")
                ]
            ]
            
            state.waiting_duplicate_confirmation = True
            state.pending_duplicate_file = {
                'name': file_name,
                'content': content,
                'size': content_size,
                'operation': state.operation,
                'requires_intervals': content_size > CONTENT_LIMIT_FOR_INTERVALS,
                'message_thread_id': message_thread_id
            }
            
            if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
                chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
                state.pending_duplicate_file['chunks'] = chunks
            else:
                parts = split_large_content(content, CONTENT_LIMIT_FOR_INTERVALS)
                state.pending_duplicate_file['parts'] = parts
            
            # Create status info message
            status_info = f"Status: {status_emoji} {status_display}"
            if old_status == 'completed':
                # Try to get message/parts count from history
                for entry in file_history.get(chat_id, []):
                    if (entry['filename'] == file_name and 
                        entry.get('size', 0) == content_size and
                        abs((entry['timestamp'] - duplicate_entry.timestamp).total_seconds()) < 60):
                        if entry.get('parts_count', 0) > 0:
                            status_info += f" ({entry['parts_count']} part{'s' if entry['parts_count'] > 1 else ''}"
                            if entry.get('messages_count', 0) > 0:
                                status_info += f", {entry['messages_count']} messages"
                            status_info += ")"
                        elif entry.get('messages_count', 0) > 0:
                            status_info += f" ({entry['messages_count']} message{'s' if entry['messages_count'] > 1 else ''})"
                        break
            
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"⚠️ **Duplicate File Detected**\n\n"
                     f"File: `{file_name}`\n"
                     f"Size: {content_size:,} bytes\n\n"
                     f"This exact file was already processed {hours_ago:.1f} hours ago.\n"
                     f"{status_info}\n\n"
                     f"Do you want to post it again?\n\n"
                     f"✅ Yes, Post  ❌ No, Don't",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            return
        
        # No duplicate or user will confirm later - proceed with normal processing
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
        
        # Add file to history with "uploaded" status (creates new instance)
        add_file_to_history(file_name, content_size, chat_id, "uploaded")
        
        # Thread-safe queue addition
        queue_size = await add_to_queue_safe(state, file_info)
        queue_position = get_queue_position_safe(state)
        
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
            "tracked_messages": len(enhanced_message_tracking),
            "file_history_entries": len(file_upload_history),
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
