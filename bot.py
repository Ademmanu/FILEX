#!/usr/bin/env python3
"""
FileX Bot - Enhanced with message tracking, duplicate detection, and reliable deletion
"""

import os
import json
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Deque, Set, Tuple
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

# ==================== GLOBAL STORAGE ====================
file_history = defaultdict(list)
file_message_mapping = {}
file_notification_mapping = {}
file_other_notifications = {}

# ==================== ENHANCED MESSAGE TRACKING ====================
class EnhancedMessageEntry:
    """Stores full message content and first two words"""
    def __init__(self, chat_id: int, message_id: int, full_content: str, 
                 first_two_words: str, timestamp: datetime):
        self.chat_id = chat_id
        self.message_id = message_id
        self.full_content = full_content
        self.first_two_words = first_two_words
        self.timestamp = timestamp
        self.words = self._extract_words(full_content)
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract all words from message content"""
        # Remove URLs and markdown, keep only words
        clean_text = re.sub(r'https?://\S+', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text.split()
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"MessageEntry(chat={self.chat_id}, words='{self.first_two_words}', time={self.timestamp.strftime('%H:%M:%S')})"

# Enhanced tracking storage - NO LIMIT
enhanced_message_tracking: List[EnhancedMessageEntry] = []
admin_preview_mode: Set[int] = set()

# ==================== FILE UPLOAD TRACKING ====================
class FileUploadRecord:
    """Tracks file uploads for duplicate detection"""
    def __init__(self, chat_id: int, user_id: int, filename: str, 
                 size: int, timestamp: datetime, message_thread_id: Optional[int] = None):
        self.chat_id = chat_id
        self.user_id = user_id
        self.filename = filename
        self.size = size
        self.timestamp = timestamp
        self.message_thread_id = message_thread_id
        self.unique_id = f"{chat_id}_{filename}_{size}_{timestamp.timestamp()}"
    
    def is_duplicate(self, other_filename: str, other_size: int, 
                     other_chat_id: int = None, other_user_id: int = None) -> bool:
        """Check if this is a duplicate within 72 hours"""
        if self.filename != other_filename or self.size != other_size:
            return False
        
        # Check time window
        time_diff = datetime.now(UTC_PLUS_1) - self.timestamp
        if time_diff > timedelta(hours=72):
            return False
        
        # Optional: check chat/user context
        if other_chat_id and self.chat_id != other_chat_id:
            return False
        if other_user_id and self.user_id != other_user_id:
            return False
            
        return True
    
    def __repr__(self):
        return f"FileUploadRecord({self.filename}, {self.size} bytes, {self.timestamp})"

file_upload_history: List[FileUploadRecord] = []

# ==================== ADMIN FUNCTIONS ====================
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

# ==================== ENHANCED MESSAGE TRACKING FUNCTIONS ====================
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

def track_enhanced_message(chat_id: int, message_id: int, message_text: str):
    """Track full message content and first two words"""
    try:
        if not message_text or len(message_text.strip()) < 2:
            return
        
        first_two_words = extract_first_two_words(message_text)
        if not first_two_words:
            return
        
        # Remove expired entries first
        cleanup_old_enhanced_messages()
        
        # Add new entry
        entry = EnhancedMessageEntry(
            chat_id=chat_id,
            message_id=message_id,
            full_content=message_text,
            first_two_words=first_two_words,
            timestamp=datetime.now(UTC_PLUS_1)
        )
        
        enhanced_message_tracking.append(entry)
        logger.debug(f"Tracked enhanced message: {first_two_words}")
        
    except Exception as e:
        logger.error(f"Error tracking enhanced message: {e}")

def cleanup_old_enhanced_messages():
    """Remove messages older than 72 hours"""
    global enhanced_message_tracking
    
    if not enhanced_message_tracking:
        return
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_count = len(enhanced_message_tracking)
    
    enhanced_message_tracking = [entry for entry in enhanced_message_tracking 
                               if not entry.is_expired()]
    
    removed = initial_count - len(enhanced_message_tracking)
    if removed > 0:
        logger.info(f"Cleaned up {removed} expired message entries (older than 72h)")
        logger.info(f"Current database: {len(enhanced_message_tracking)} active entries")

def get_enhanced_tracking_stats() -> Dict[str, any]:
    """Get statistics about tracked messages"""
    cleanup_old_enhanced_messages()
    
    if not enhanced_message_tracking:
        return {
            'total': 0,
            'oldest': None,
            'newest': None,
            'unique_first_words': 0
        }
    
    oldest = min(entry.timestamp for entry in enhanced_message_tracking)
    newest = max(entry.timestamp for entry in enhanced_message_tracking)
    unique_first_words = len(set(entry.first_two_words for entry in enhanced_message_tracking))
    
    return {
        'total': len(enhanced_message_tracking),
        'oldest': oldest,
        'newest': newest,
        'unique_first_words': unique_first_words
    }

# ==================== ENHANCED PREVIEW MATCHING ====================
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

def check_enhanced_preview_match(preview_texts: List[str]) -> Tuple[
    List[Tuple[str, str]],  # Full matches (preview, matched_words)
    List[Tuple[str, Dict[str, List[int]]]],  # Partial matches (preview, {word: positions})
    List[str]  # Non-matches
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
        preview_words = extract_first_two_words(preview)
        all_preview_words = re.findall(r'\b\d+\b|\b\w+\b', preview)
        
        # Check for full match (first two words exact match)
        full_match_found = False
        partial_match_data = {}
        
        for entry in enhanced_message_tracking:
            # Full match check
            if preview_words and preview_words == entry.first_two_words:
                full_matches.append((preview, entry.first_two_words))
                full_match_found = True
                break
            
            # Partial match check
            for preview_word in all_preview_words:
                if preview_word in entry.words:
                    if preview_word not in partial_match_data:
                        partial_match_data[preview_word] = []
                    # Find all positions of this word in the original message
                    for pos, word in enumerate(entry.words, 1):
                        if word == preview_word:
                            partial_match_data[preview_word].append(pos)
        
        if full_match_found:
            continue
        elif partial_match_data:
            partial_matches.append((preview, partial_match_data))
        else:
            non_matches.append(preview)
    
    return (full_matches, partial_matches, non_matches)

def format_enhanced_preview_report(full_matches: List[Tuple[str, str]],
                                  partial_matches: List[Tuple[str, Dict[str, List[int]]]],
                                  non_matches: List[str]) -> str:
    """Format the enhanced preview report with partial match details"""
    
    total_previews = len(full_matches) + len(partial_matches) + len(non_matches)
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Enhanced Preview Analysis Report**")
    report_lines.append("─" * 45)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    full_match_percent = (len(full_matches) / total_previews * 100) if total_previews > 0 else 0
    partial_match_percent = (len(partial_matches) / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (len(non_matches) / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Full matches: {len(full_matches)} ({full_match_percent:.1f}%)")
    report_lines.append(f"• Partial matches: {len(partial_matches)} ({partial_match_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {len(non_matches)} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if full_matches:
        report_lines.append("✅ **Full matches found in database:**")
        for i, (preview, matched_words) in enumerate(full_matches, 1):
            report_lines.append(f"{i}. {preview}")
            report_lines.append(f"   Matched: {matched_words}")
    
    if partial_matches:
        if full_matches:
            report_lines.append("")
        report_lines.append("⚠️ **Partial matches found:**")
        for i, (preview, word_positions) in enumerate(partial_matches, 1):
            report_lines.append(f"{i}. Preview: {preview}")
            for word, positions in word_positions.items():
                positions_str = ', '.join(map(str, positions))
                report_lines.append(f"   Matched word: {word}")
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
def cleanup_old_file_uploads():
    """Remove file upload records older than 72 hours"""
    global file_upload_history
    
    if not file_upload_history:
        return
    
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    initial_count = len(file_upload_history)
    
    file_upload_history = [record for record in file_upload_history 
                          if record.timestamp > cutoff]
    
    removed = initial_count - len(file_upload_history)
    if removed > 0:
        logger.info(f"Cleaned up {removed} expired file upload records")

def check_duplicate_file(chat_id: int, user_id: int, filename: str, 
                        size: int, message_thread_id: Optional[int] = None) -> Optional[FileUploadRecord]:
    """Check if file is a duplicate within 72 hours"""
    cleanup_old_file_uploads()
    
    for record in file_upload_history:
        if record.is_duplicate(filename, size, chat_id, user_id):
            return record
    
    return None

def record_file_upload(chat_id: int, user_id: int, filename: str, 
                      size: int, message_thread_id: Optional[int] = None):
    """Record a file upload"""
    cleanup_old_file_uploads()
    
    record = FileUploadRecord(
        chat_id=chat_id,
        user_id=user_id,
        filename=filename,
        size=size,
        timestamp=datetime.now(UTC_PLUS_1),
        message_thread_id=message_thread_id
    )
    
    file_upload_history.append(record)
    logger.info(f"Recorded file upload: {filename} ({size} bytes)")

# ==================== ENHANCED ADMIN COMMANDS ====================
async def adminpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    admin_preview_mode.add(user_id)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="🔍 **Enhanced Admin Preview Mode Activated**\n\n"
             "Please send the report data with 'Preview:' sections.\n"
             "Example format:\n"
             "```\n"
             "Preview: 97699115546 97699115547\n"
             "Preview: 237620819778 237620819780\n"
             "```\n\n"
             "Now includes:\n"
             "• Full matches (first two words exact match)\n"
             "• Partial matches (words found in any position)\n"
             "• Word position tracking\n\n"
             "Use /cancelpreview to cancel.",
        parse_mode='Markdown'
    )

async def cancelpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
             "Preview mode has been deactivated.\n"
             "Use /adminpreview to start preview mode again.",
        parse_mode='Markdown'
    )

async def adminstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin_authorization(update, context):
        return
    
    stats = get_enhanced_tracking_stats()
    file_stats = len(file_upload_history)
    
    if stats['total'] == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            message_thread_id=update.effective_message.message_thread_id,
            text="📊 **Enhanced Message Tracking Statistics**\n\n"
                 "No messages tracked yet.\n"
                 f"File upload records: {file_stats}\n"
                 "Database is empty.",
            parse_mode='Markdown'
        )
        return
    
    oldest_str = stats['oldest'].strftime('%Y-%m-%d %H:%M:%S')
    newest_str = stats['newest'].strftime('%Y-%m-%d %H:%M:%S')
    
    message = f"📊 **Enhanced Message Tracking Statistics**\n\n"
    message += f"**Total tracked messages:** {stats['total']} (unlimited)\n"
    message += f"**Unique first word pairs:** {stats['unique_first_words']}\n"
    message += f"**File upload records:** {file_stats}\n"
    message += f"**Oldest entry:** {oldest_str}\n"
    message += f"**Newest entry:** {newest_str}\n"
    message += f"**Active preview sessions:** {len(admin_preview_mode)}\n\n"
    message += f"Messages auto-delete after 72 hours."
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text=message,
        parse_mode='Markdown'
    )

async def handle_admin_preview_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    
    full_matches, partial_matches, non_matches = check_enhanced_preview_match(previews)
    report_text = format_enhanced_preview_report(full_matches, partial_matches, non_matches)
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=report_text,
        parse_mode='Markdown'
    )

# ==================== PERIODIC CLEANUP ====================
async def periodic_cleanup_task():
    while True:
        try:
            cleanup_old_enhanced_messages()
            cleanup_old_file_uploads()
            cleanup_stuck_tasks()
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

# ==================== EXISTING BOT FUNCTIONS (UPDATED) ====================
def update_file_history(chat_id: int, filename: str, status: str, parts_count: int = 0, messages_count: int = 0):
    file_history[chat_id] = [entry for entry in file_history.get(chat_id, []) 
                           if entry['filename'] != filename]
    
    entry = {
        'filename': filename,
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

def cleanup_stuck_tasks():
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

# ==================== ENHANCED FILE PROCESSING ====================
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
            
            # UPDATED: Use enhanced message tracking
            if sent_message and sent_message.text:
                track_enhanced_message(chat_id, sent_message.message_id, sent_message.text)
            
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
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return False

# ==================== ENHANCED DELETION FUNCTIONS ====================
async def delete_file_content_with_confirmation(chat_id: int, message_thread_id: Optional[int], 
                                               filename: str, context: ContextTypes.DEFAULT_TYPE, 
                                               state: UserState, specific_index: int = None):
    """Enhanced deletion with multiple file handling"""
    
    # Find all occurrences of the filename
    file_occurrences = []
    for i, entry in enumerate(file_history.get(chat_id, [])):
        if entry['filename'] == filename:
            file_occurrences.append({
                'index': i,
                'entry': entry,
                'timestamp': entry['timestamp'],
                'size': entry.get('size', 0)
            })
    
    # If multiple occurrences and no specific index selected, ask for clarification
    if len(file_occurrences) > 1 and specific_index is None:
        keyboard = []
        for i, occ in enumerate(file_occurrences, 1):
            time_str = occ['timestamp'].strftime('%H:%M:%S')
            size_str = f"{occ['size']:,} chars" if occ['size'] > 0 else "size unknown"
            button_text = f"{i}. {filename} ({time_str}, {size_str})"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"delfile_{occ['index']}")])
        
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="delfile_cancel")])
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"📋 **Multiple Files Found!**\n\n"
                 f"Which `{filename}` do you want to delete?\n\n"
                 f"Found {len(file_occurrences)} occurrences:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return False  # Not deleted yet, waiting for user selection
    
    # Single file or specific index selected
    target_index = specific_index if specific_index is not None else (
        file_occurrences[0]['index'] if file_occurrences else None
    )
    
    if target_index is None:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **File not found**\n\n"
                 f"No record found for `{filename}` in your history.",
            parse_mode='Markdown'
        )
        state.waiting_for_filename = False
        state.last_deleted_file = None
        return False
    
    # Perform deletion
    return await perform_file_deletion(chat_id, message_thread_id, filename, 
                                       context, state, target_index)

async def perform_file_deletion(chat_id: int, message_thread_id: Optional[int], 
                               filename: str, context: ContextTypes.DEFAULT_TYPE, 
                               state: UserState, target_index: int) -> bool:
    """Actually delete file content (enhanced reliability)"""
    
    # Check if currently processing
    is_currently_processing = False
    if state.queue and state.queue[0].get('name') == filename and state.processing:
        is_currently_processing = True
        state.cancel_current_task()
        if state.queue:
            state.queue.popleft()
    
    messages_to_delete = 0
    deleted_messages = []
    failed_deletions = []
    
    # Delete content messages (handle already deleted ones gracefully)
    if filename in file_message_mapping:
        for msg_id in file_message_mapping[filename]:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                deleted_messages.append(msg_id)
                messages_to_delete += 1
                logger.info(f"Deleted content message {msg_id} for file {filename}")
            except Exception as e:
                if "message to delete not found" not in str(e).lower():
                    logger.error(f"Failed to delete content message {msg_id}: {e}")
                    failed_deletions.append(msg_id)
    
    # Delete other notifications
    if filename in file_other_notifications:
        for msg_id in file_other_notifications[filename]:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                deleted_messages.append(msg_id)
                messages_to_delete += 1
                logger.info(f"Deleted other notification {msg_id} for file {filename}")
            except Exception as e:
                if "message to delete not found" not in str(e).lower():
                    logger.error(f"Failed to delete notification {msg_id}: {e}")
                    failed_deletions.append(msg_id)
    
    # Edit acceptance/queued notification if exists
    notification_edited = False
    if filename in file_notification_mapping:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=file_notification_mapping[filename],
                text=f"🗑️ **File Content Deleted**\n\n"
                     f"File: `{filename}`\n"
                     f"Messages deleted: {messages_to_delete}\n"
                     f"Status: {'Partially deleted' if failed_deletions else 'Fully deleted'}",
                parse_mode='Markdown'
            )
            notification_edited = True
        except Exception as e:
            logger.error(f"Failed to edit notification: {e}")
    
    # Update history
    update_file_history(chat_id, filename, 'deleted')
    
    # Remove from queue
    state.remove_task_by_name(filename)
    
    # Send confirmation
    result_text = f"🗑️ `{filename}` content deletion completed\n"
    result_text += f"✅ Messages removed: {messages_to_delete}\n"
    if failed_deletions:
        result_text += f"⚠️ Failed to delete: {len(failed_deletions)} messages\n"
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=result_text,
        parse_mode='Markdown'
    )
    
    state.waiting_for_filename = False
    
    # Resume processing if needed
    if is_currently_processing and state.queue and not state.processing:
        next_file = state.queue[0].get('name', 'Unknown')
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🔄 **Moving to next task**\n\nStarting next task: `{next_file}`",
            parse_mode='Markdown'
        )
        state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    
    return True

# ==================== UPDATED COMMAND HANDLERS ====================
async def delfilecontent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if state.waiting_for_filename:
        # User already in deletion flow - send inline buttons instead of text
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="delfile_cancel")]
        ]
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="🗑️ **Delete File Content**\n\n"
                 "Please send me the filename you want to delete content from.\n"
                 "Example: `Sudan WhatsApp.txt`\n\n"
                 "✅ Click Cancel to cancel this operation.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        return
    
    state.waiting_for_filename = True
    state.last_deleted_file = None
    
    keyboard = [
        [InlineKeyboardButton("❌ Cancel", callback_data="delfile_cancel")]
    ]
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="🗑️ **Delete File Content**\n\n"
             "Please send me the filename you want to delete content from.\n"
             "Example: `Sudan WhatsApp.txt`\n\n"
             "✅ Click Cancel to cancel this operation.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def handle_deletion_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button selections for file deletion"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    if not is_authorized(user_id, chat_id):
        return
    
    data = query.data
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if data == "delfile_cancel":
        state.waiting_for_filename = False
        state.last_deleted_file = None
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=query.message.message_id,
            text="❌ **Deletion cancelled**\n\nOperation cancelled by user.",
            parse_mode='Markdown'
        )
        return
    
    if data.startswith("delfile_"):
        # Extract index from callback data
        try:
            index_str = data.split("_")[1]
            target_index = int(index_str)
            
            # Get filename from message text
            message_text = query.message.text
            filename_match = re.search(r'`(.+?)`', message_text)
            if filename_match:
                filename = filename_match.group(1)
                await perform_file_deletion(chat_id, message_thread_id, filename, 
                                          context, state, target_index)
                
                # Delete the selection message
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=query.message.message_id
                )
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing deletion callback: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="❌ **Error**\n\nInvalid selection. Please try again.",
                parse_mode='Markdown'
            )

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
        
        # Use enhanced deletion with multiple file handling
        await delete_file_content_with_confirmation(
            chat_id, message_thread_id, filename, context, state
        )
        return
    
    # Check if admin is in preview mode
    user_id = update.effective_user.id
    if user_id in admin_preview_mode:
        await handle_admin_preview_message(update, context)
        return

# ==================== DUPLICATE FILE HANDLING ====================
async def ask_duplicate_confirmation(chat_id: int, message_thread_id: Optional[int],
                                    filename: str, size: int, previous_upload: FileUploadRecord,
                                    context: ContextTypes.DEFAULT_TYPE, state: UserState):
    """Ask user if they want to post duplicate file"""
    
    time_diff = datetime.now(UTC_PLUS_1) - previous_upload.timestamp
    hours_ago = time_diff.total_seconds() / 3600
    
    keyboard = [
        [
            InlineKeyboardButton("✅ Yes, Post", callback_data=f"dup_yes_{filename}_{size}"),
            InlineKeyboardButton("❌ No, Don't", callback_data=f"dup_no_{filename}_{size}")
        ]
    ]
    
    message = await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=f"⚠️ **Duplicate File Detected!**\n\n"
             f"File: `{filename}`\n"
             f"Size: {size:,} characters\n"
             f"Last uploaded: {hours_ago:.1f} hours ago\n\n"
             f"This file was already posted within the last 72 hours.\n"
             f"Do you want to repost it?\n\n"
             f"_Note: This check applies to files with same name and size._",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    
    state.awaiting_duplicate_confirmation = True
    state.duplicate_file_info = {
        'message_id': message.message_id,
        'filename': filename,
        'size': size,
        'previous_upload': previous_upload
    }

async def handle_duplicate_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle duplicate file confirmation response"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    
    if not is_authorized(user_id, chat_id):
        return
    
    data = query.data
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    if not state.awaiting_duplicate_confirmation:
        return
    
    if data.startswith("dup_yes_"):
        # User wants to post duplicate
        parts = data.split("_")
        if len(parts) >= 4:
            filename = parts[2]
            try:
                size = int(parts[3])
                
                # Record the upload
                record_file_upload(chat_id, user_id, filename, size, message_thread_id)
                
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=query.message.message_id,
                    text=f"✅ **Duplicate file accepted**\n\n"
                         f"File `{filename}` will be processed as requested.",
                    parse_mode='Markdown'
                )
                
                # Continue with normal file processing
                state.awaiting_duplicate_confirmation = False
                state.duplicate_file_info = None
                
                # Note: The actual file processing needs to be triggered from handle_file
                # This requires passing the file info back to handle_file
                return True
                
            except (ValueError, IndexError):
                logger.error(f"Error parsing duplicate confirmation: {data}")
    
    elif data.startswith("dup_no_"):
        # User doesn't want to post duplicate
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=query.message.message_id,
            text="❌ **Duplicate file rejected**\n\n"
                 "File upload cancelled by user.",
            parse_mode='Markdown'
        )
    
    state.awaiting_duplicate_confirmation = False
    state.duplicate_file_info = None
    return False

# ==================== UPDATED FILE HANDLER ====================
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    user_id = update.effective_user.id
    state = await get_user_state_safe(chat_id)
    
    # Check for duplicate confirmation response first
    if state.awaiting_duplicate_confirmation:
        # If we're awaiting confirmation, ignore new file uploads
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="⚠️ **Please respond to the duplicate file prompt first.**\n\n"
                 "You have a pending duplicate file confirmation.",
            parse_mode='Markdown'
        )
        return
    
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
    
    # Check supported file type
    ext = Path(file_name).suffix.lower()
    if ext not in ['.txt', '.csv']:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Unsupported file type**\nPlease upload only TXT or CSV files.",
            parse_mode='Markdown'
        )
        return
    
    # Check queue size
    queue_position = get_queue_position_safe(state)
    if queue_position >= MAX_QUEUE_SIZE:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"❌ **Queue is full!**\nMaximum {MAX_QUEUE_SIZE} files allowed.\nPlease wait for current files to be processed.",
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
                text=f"❌ **Download timeout**\nFile `{file_name}` is too large or download failed.\nPlease try again with a smaller file.",
                parse_mode='Markdown'
            )
            return
        except Exception as e:
            logger.error(f"File download error: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"❌ **Download failed**\nCould not download file `{file_name}`.\nPlease try again.",
                parse_mode='Markdown'
            )
            return
        
        content_size = len(file_bytes)
        
        # Check for duplicate file within 72 hours
        duplicate_record = check_duplicate_file(chat_id, user_id, file_name, content_size, message_thread_id)
        if duplicate_record:
            await ask_duplicate_confirmation(chat_id, message_thread_id, file_name, 
                                           content_size, duplicate_record, context, state)
            return
        
        # Process file content
        if ext == '.csv':
            text = file_bytes.decode('utf-8', errors='ignore')
            csv_file = io.StringIO(text)
            reader = csv.reader(csv_file)
            lines = [' '.join(str(cell) for cell in row) for row in reader]
            content = '\n'.join(lines)
        else:
            content = file_bytes.decode('utf-8', errors='ignore')
        
        # Apply operation
        if state.operation == 'number':
            spaced = re.sub(r'[^0-9]', ' ', content)
            content = re.sub(r'\s+', ' ', spaced).strip()
        elif state.operation == 'alphabet':
            spaced = re.sub(r'[^a-zA-Z ]', ' ', content)
            content = re.sub(r'\s+', ' ', spaced).strip()
        
        content_size = len(content)
        
        # Record file upload
        record_file_upload(chat_id, user_id, file_name, content_size, message_thread_id)
        
        # Split content
        if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
            # Split into chunks for immediate sending
            chunks = []
            if content_size <= TELEGRAM_MESSAGE_LIMIT:
                chunks = [content]
            else:
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
                
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) <= TELEGRAM_MESSAGE_LIMIT:
                        current_chunk += word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        if len(word) > TELEGRAM_MESSAGE_LIMIT:
                            for i in range(0, len(word), TELEGRAM_MESSAGE_LIMIT):
                                chunks.append(word[i:i + TELEGRAM_MESSAGE_LIMIT])
                            current_chunk = ""
                        else:
                            current_chunk = word
                if current_chunk:
                    chunks.append(current_chunk)
            
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
            # Split into parts for interval sending
            parts = []
            if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
                parts = [content]
            else:
                current_part = ""
                current_size = 0
                lines = content.split('\n')
                
                for line in lines:
                    if current_size + len(line) + 1 <= CONTENT_LIMIT_FOR_INTERVALS:
                        if current_part:
                            current_part += '\n' + line
                            current_size += len(line) + 1
                        else:
                            current_part = line
                            current_size = len(line)
                    else:
                        if current_part:
                            parts.append(current_part)
                        
                        if len(line) > CONTENT_LIMIT_FOR_INTERVALS:
                            words = line.split(' ')
                            current_part = ""
                            current_size = 0
                            
                            for word in words:
                                if current_size + len(word) + 1 <= CONTENT_LIMIT_FOR_INTERVALS:
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
            
            file_info = {
                'name': file_name,
                'content': content,
                'parts': parts,
                'size': content_size,
                'operation': state.operation,
                'requires_intervals': True,
                'message_thread_id': message_thread_id
            }
        
        # Add to queue
        queue_size = await add_to_queue_safe(state, file_info)
        queue_position = get_queue_position_safe(state)
        
        # Check if should start processing
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
            text=f"❌ **Processing timeout**\nFile `{file_name}` is too large to process.\nPlease try with a smaller file.",
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

# ==================== REST OF THE FUNCTIONS (unchanged except for UPDATED calls) ====================
# Note: All existing functions like process_queue, send_chunks_immediately, etc. 
# remain the same but now use the UPDATED send_telegram_message_safe which calls
# track_enhanced_message instead of track_message

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
                update_file_history(chat_id, filename, 'running', parts_count=len(file_info['parts']))
            else:
                update_file_history(chat_id, filename, 'running', messages_count=len(file_info['chunks']))
            
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
                    if filename not in file_other_notifications:
                        file_other_notifications[filename] = []
                    file_other_notifications[filename].append(sending_msg.message_id)
                else:
                    sending_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"📤 Sending: `{filename}`\n"
                             f"📊 Total messages: {len(file_info['chunks'])}",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    if filename not in file_other_notifications:
                        file_other_notifications[filename] = []
                    file_other_notifications[filename].append(sending_msg.message_id)
            
            success = False
            if file_info.get('requires_intervals', False):
                # Note: send_with_intervals internally calls send_telegram_message_safe
                # which now uses track_enhanced_message
                success = await send_with_intervals(
                    chat_id, context, 
                    file_info['parts'], 
                    filename,
                    state,
                    file_message_thread_id
                )
            else:
                # Note: send_chunks_immediately internally calls send_telegram_message_safe
                # which now uses track_enhanced_message
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
                    if processed_filename not in file_other_notifications:
                        file_other_notifications[processed_filename] = []
                    file_other_notifications[processed_filename].append(failed_msg.message_id)
            
            state.current_parts = []
            state.current_index = 0
            
            # Wait 2 minutes before processing next file
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
            if current_file not in file_other_notifications:
                file_other_notifications[current_file] = []
            file_other_notifications[current_file].append(error_msg.message_id)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

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
            if filename not in file_other_notifications:
                file_other_notifications[filename] = []
            file_other_notifications[filename].append(completion_msg.message_id)
            
            update_file_history(chat_id, filename, 'completed', messages_count=total_messages_sent)
            return True
        else:
            logger.error(f"No chunks sent for `{filename}`")
            return False
        
    except Exception as e:
        logger.error(f"Error in send_chunks_immediately: {e}")
        return False

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
            
            # Send large content part
            try:
                chunks = []
                if len(part) <= TELEGRAM_MESSAGE_LIMIT:
                    chunks = [part]
                else:
                    words = []
                    current_word = ""
                    for char in part:
                        if char.isspace():
                            if current_word:
                                words.append(current_word)
                                current_word = ""
                            words.append(char)
                        else:
                            current_word += char
                    if current_word:
                        words.append(current_word)
                    
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) <= TELEGRAM_MESSAGE_LIMIT:
                            current_chunk += word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            if len(word) > TELEGRAM_MESSAGE_LIMIT:
                                for i in range(0, len(word), TELEGRAM_MESSAGE_LIMIT):
                                    chunks.append(word[i:i + TELEGRAM_MESSAGE_LIMIT])
                                current_chunk = ""
                            else:
                                current_chunk = word
                    if current_chunk:
                        chunks.append(current_chunk)
                
                total_messages_in_part = 0
                
                if total_parts > 1:
                    part_header_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=f"📄 `{filename}` - Part {i}/{total_parts}",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    if filename not in file_other_notifications:
                        file_other_notifications[filename] = []
                    file_other_notifications[filename].append(part_header_msg.message_id)
                    total_messages_in_part += 1
                
                for j, chunk in enumerate(chunks, 1):
                    if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, filename=filename):
                        total_messages_in_part += 1
                        
                        if j < len(chunks):
                            await asyncio.sleep(MESSAGE_DELAY)
                    else:
                        logger.error(f"Failed to send chunk {j} for part {i} of `{filename}`")
                        return False
                
                total_messages_sent += total_messages_in_part
                
            except Exception as e:
                logger.error(f"Error sending large content part: {e}")
                return False
            
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
        if filename not in file_other_notifications:
            file_other_notifications[filename] = []
        file_other_notifications[filename].append(completion_msg.message_id)
        
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

# ==================== OTHER COMMAND HANDLERS (unchanged) ====================
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
    
    # Handle duplicate confirmation first
    if query.data.startswith("dup_"):
        await handle_duplicate_confirmation(update, context)
        return
    
    # Handle deletion selection
    if query.data.startswith("delfile_"):
        await handle_deletion_selection(update, context)
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
            update_file_history(chat_id, current_file, 'paused', parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, 'paused', messages_count=len(file_info['chunks']))
    
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
            update_file_history(chat_id, current_file, 'running', parts_count=len(file_info['parts']))
        else:
            update_file_history(chat_id, current_file, 'running', messages_count=len(file_info['chunks']))
    
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
    
    update_file_history(chat_id, current_file, 'skipped')
    
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
                update_file_history(chat_id, file_info['name'], 'cancelled', parts_count=len(file_info.get('parts', [])))
            else:
                update_file_history(chat_id, file_info['name'], 'cancelled', messages_count=len(file_info.get('chunks', [])))
    
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

# ==================== HEALTH HANDLER ====================
async def health_handler(request):
    return web.Response(
        text=json.dumps({
            "status": "running",
            "service": "filex-bot",
            "active_users": len(user_states),
            "allowed_ids_count": len(ALLOWED_IDS),
            "admin_ids_count": len(ADMIN_IDS),
            "tracked_messages": len(enhanced_message_tracking),
            "file_upload_records": len(file_upload_history),
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
    
    # Add callback handlers
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Add message handlers
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
    
    logger.info("🤖 Enhanced FileX Bot started")
    
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
