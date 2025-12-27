#!/usr/bin/env python3
"""
FileX Bot - Clean message headers, only part headers at beginning of each part
Admin Preview System: Track first two words of all messages
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

UTC_PLUS_1 = timezone(timedelta(hours=1))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

file_history = defaultdict(list)
file_message_mapping = defaultdict(lambda: defaultdict(list))  # chat_id -> filename -> [message_ids]
file_notification_mapping = defaultdict(dict)  # chat_id -> filename -> message_id
file_other_notifications = defaultdict(lambda: defaultdict(list))  # chat_id -> filename -> [message_ids]

# ==================== ADMIN PREVIEW SYSTEM ====================
class MessageEntry:
    """Stores first two words of sent messages"""
    def __init__(self, chat_id: int, message_id: int, first_two_words: str, timestamp: datetime):
        self.chat_id = chat_id
        self.message_id = message_id
        self.first_two_words = first_two_words
        self.timestamp = timestamp
    
    def is_expired(self) -> bool:
        """Check if entry is older than 72 hours"""
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"MessageEntry(chat={self.chat_id}, words='{self.first_two_words}', time={self.timestamp.strftime('%H:%M:%S')})"

# Global storage for message tracking
message_tracking: List[MessageEntry] = []
MAX_TRACKED_MESSAGES = 1000
admin_preview_mode: Set[int] = set()  # Just track which admins are in preview mode

# ==================== FILE HISTORY FUNCTIONS ====================
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

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

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
            chat_id=next(iter(ADMIN_IDS), 6389552329),
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

# ==================== MESSAGE TRACKING FUNCTIONS ====================
def extract_first_two_words(text: str) -> str:
    """Extract first two words from text, handling markdown and special chars"""
    # Remove markdown formatting but keep normal text
    clean_text = text
    # Remove URLs
    clean_text = re.sub(r'https?://\S+', ' ', clean_text)
    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    # Split into words and take first two
    words = clean_text.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    elif len(words) == 1:
        return words[0]
    else:
        return ""

def track_message(chat_id: int, message_id: int, message_text: str):
    """Track first two words of sent message"""
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
            first_two_words=first_two_words,
            timestamp=datetime.now(UTC_PLUS_1)
        )
        
        message_tracking.append(entry)
        
        # Keep only last MAX_TRACKED_MESSAGES
        if len(message_tracking) > MAX_TRACKED_MESSAGES:
            del message_tracking[0]
        
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
    """Extract all 'Preview:' sections from report text - exact matching"""
    preview_sections = []
    
    # Pattern to match "Preview: " followed by content until end of line
    # This handles the specific format in your example
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

def check_preview_against_database(preview_texts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check preview texts against tracked messages
    Returns: (matches, non_matches)
    """
    if not message_tracking:
        return ([], preview_texts)
    
    # Get all unique first_two_words from database
    tracked_words = {entry.first_two_words for entry in message_tracking}
    
    matches = []
    non_matches = []
    
    for preview in preview_texts:
        # Extract first two words from preview text
        preview_words = extract_first_two_words(preview)
        
        if preview_words and preview_words in tracked_words:
            matches.append(preview)
        else:
            non_matches.append(preview)
    
    return (matches, non_matches)

def format_preview_report_exact(matches: List[str], non_matches: List[str], 
                               total_previews: int) -> str:
    """Format the preview report exactly like the example provided"""
    
    if total_previews == 0:
        return "❌ **No preview content found**\n\nNo valid preview sections found in the message.\n\nUse /cancelpreview to cancel."
    
    report_lines = []
    report_lines.append("📊 **Preview Analysis Report**")
    report_lines.append("─" * 35)
    report_lines.append("")
    report_lines.append("📋 **Preview Analysis:**")
    
    match_percent = (len(matches) / total_previews * 100) if total_previews > 0 else 0
    non_match_percent = (len(non_matches) / total_previews * 100) if total_previews > 0 else 0
    
    report_lines.append(f"• Total previews checked: {total_previews}")
    report_lines.append(f"• Matches found: {len(matches)} ({match_percent:.1f}%)")
    report_lines.append(f"• Non-matches: {len(non_matches)} ({non_match_percent:.1f}%)")
    report_lines.append("")
    
    if matches:
        report_lines.append("✅ **Matches found in database:**")
        for i, match in enumerate(matches, 1):
            report_lines.append(f"{i}. {match}")
    
    if non_matches:
        if matches:
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
    
    message = f"📊 **Message Tracking Statistics**\n\n"
    message += f"**Total tracked messages:** {stats['total']}\n"
    message += f"**Unique word pairs:** {stats['unique_words']}\n"
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
    
    # Check against database
    matches, non_matches = check_preview_against_database(previews)
    total_previews = len(previews)
    
    # Format report exactly as requested
    report_text = format_preview_report_exact(matches, non_matches, total_previews)
    
    # Send the report
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=report_text,
        parse_mode='Markdown'
    )

# ==================== USER STATE CLASS ====================
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
        
    def clear_queue(self):
        count = len(self.queue)
        self.queue.clear()
        self.cancel_requested = False
        return count
        
    def has_active_tasks(self) -> bool:
        return self.processing or len(self.queue) > 0
    
    def remove_task_by_name(self, filename: str):
        """Remove task by filename from queue"""
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
            if task.get('name') != filename:
                new_queue.append(task)
        self.queue = new_queue

user_states: Dict[int, UserState] = {}

def get_user_state(chat_id: int) -> UserState:
    if chat_id not in user_states:
        user_states[chat_id] = UserState(chat_id)
    return user_states[chat_id]

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
            
            # Track message for admin preview system
            if sent_message and sent_message.text:
                track_message(chat_id, sent_message.message_id, sent_message.text)
            
            if filename and sent_message:
                if notification_type == 'content':
                    file_message_mapping[chat_id][filename].append(sent_message.message_id)
                elif notification_type == 'notification':
                    file_notification_mapping[chat_id][filename] = sent_message.message_id
            
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for chat {chat_id}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return False

async def track_other_notification(chat_id: int, filename: str, message_id: int):
    file_other_notifications[chat_id][filename].append(message_id)

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
        
        state = get_user_state(chat_id)
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
        
        update_file_history(chat_id, filename, 'completed', parts_count=total_parts, messages_count=total_messages_sent)
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def cleanup_completed_file(chat_id: int, filename: str):
    if chat_id in file_message_mapping and filename in file_message_mapping[chat_id]:
        del file_message_mapping[chat_id][filename]
    if chat_id in file_notification_mapping and filename in file_notification_mapping[chat_id]:
        del file_notification_mapping[chat_id][filename]
    if chat_id in file_other_notifications and filename in file_other_notifications[chat_id]:
        del file_other_notifications[chat_id][filename]
    
    logger.info(f"Cleaned up tracking for completed file: {filename} in chat {chat_id}")

async def process_queue(chat_id: int, context: ContextTypes.DEFAULT_TYPE, message_thread_id: Optional[int] = None):
    state = get_user_state(chat_id)
    
    if state.processing:
        return
    
    state.processing = True
    state.cancel_requested = False
    state.paused = False
    
    try:
        while state.queue and not state.cancel_requested:
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
            
            if state.queue and state.queue[0]['name'] == filename:
                processed_file = state.queue.popleft()
                processed_filename = processed_file['name']
                
                if success and not state.cancel_requested:
                    logger.info(f"Successfully processed `{processed_filename}` for chat {chat_id}")
                    await cleanup_completed_file(chat_id, processed_filename)
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

# ==================== EXISTING COMMANDS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = get_user_state(chat_id)
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="🤖 **FileX Bot**\n\n"
        f"📝 **Current operation:** {state.operation.capitalize()} content\n"
        f"📊 **Queue:** {max(0, len(state.queue) - 1) if state.processing else len(state.queue)}/{MAX_QUEUE_SIZE} files\n\n"
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
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
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
    state = get_user_state(chat_id)
    
    status_lines = []
    status_lines.append("📊 **FileX Status**")
    status_lines.append("──────────────")
    status_lines.append(f"🔧 **Operation:** {state.operation.capitalize()} content")
    status_lines.append(f"📋 **Queue:** {max(0, len(state.queue) - 1) if state.processing else len(state.queue)}/{MAX_QUEUE_SIZE} files")
    
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
            'paused': '⏸️'
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
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
    
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

async def delfilecontent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    message_thread_id = update.effective_message.message_thread_id
    state = get_user_state(chat_id)
    
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
        
        await process_file_deletion(chat_id, message_thread_id, filename, context, state)
        return
    
    state.waiting_for_filename = True
    state.last_deleted_file = None
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="🗑️ **Delete File Content**\n\n"
             "Please send me the filename you want to delete content from.\n"
             "Example: `Sudan WhatsApp.txt`\n\n"
             "Type `cancel` to cancel this operation.",
        parse_mode='Markdown'
    )

async def process_file_deletion(chat_id: int, message_thread_id: Optional[int], filename: str, 
                                context: ContextTypes.DEFAULT_TYPE, state: UserState):
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
    
    file_exists = any(
        entry['filename'] == filename 
        for entry in file_history.get(chat_id, [])
    )
    
    if not file_exists:
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
    
    is_currently_processing = False
    if state.queue and state.queue[0].get('name') == filename and state.processing:
        is_currently_processing = True
        state.cancel_current_task()
        if state.queue:
            state.queue.popleft()
    
    messages_to_delete = 0
    deleted_messages = []
    
    # Delete content messages
    if chat_id in file_message_mapping and filename in file_message_mapping[chat_id]:
        messages_to_delete += len(file_message_mapping[chat_id][filename])
        for msg_id in file_message_mapping[chat_id][filename]:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                deleted_messages.append(msg_id)
                logger.info(f"Deleted content message {msg_id} for file {filename} in chat {chat_id}")
            except Exception as e:
                logger.error(f"Failed to delete content message {msg_id}: {e}")
    
    # Delete other notification messages
    if chat_id in file_other_notifications and filename in file_other_notifications[chat_id]:
        for msg_id in file_other_notifications[chat_id][filename]:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                messages_to_delete += 1
                deleted_messages.append(msg_id)
                logger.info(f"Deleted other notification message {msg_id} for file {filename} in chat {chat_id}")
            except Exception as e:
                logger.error(f"Failed to delete other notification message {msg_id}: {e}")
    
    # Edit acceptance/queued notification
    notification_edited = False
    if chat_id in file_notification_mapping and filename in file_notification_mapping[chat_id]:
        notification_msg_id = file_notification_mapping[chat_id][filename]
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=notification_msg_id,
                text=f"🗑️ **File Content Deleted**\n\n"
                     f"File: `{filename}`\n"
                     f"Messages deleted: {messages_to_delete}\n"
                     f"All content from this file has been removed.",
                parse_mode='Markdown'
            )
            notification_edited = True
            logger.info(f"Edited acceptance/queued notification for {filename} in chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to edit notification message: {e}")
    
    update_file_history(chat_id, filename, 'deleted')
    
    # Remove task from queue
    state.remove_task_by_name(filename)
    
    # Send deletion confirmation
    deletion_msg = await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=f"🗑️ `{filename}` content deleted\n"
             f"Messages removed: {len(deleted_messages)}",
        parse_mode='Markdown'
    )
    
    state.waiting_for_filename = False
    
    # Clean up file tracking
    if chat_id in file_message_mapping and filename in file_message_mapping[chat_id]:
        del file_message_mapping[chat_id][filename]
    if chat_id in file_notification_mapping and filename in file_notification_mapping[chat_id]:
        del file_notification_mapping[chat_id][filename]
    if chat_id in file_other_notifications and filename in file_other_notifications[chat_id]:
        del file_other_notifications[chat_id][filename]
    
    # Continue processing next task if needed
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    state = get_user_state(chat_id)
    
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
    state = get_user_state(chat_id)
    
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
    
    if len(state.queue) >= MAX_QUEUE_SIZE:
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
        
        is_first_task = not state.processing and len(state.queue) == 0
        
        state.queue.append(file_info)
        
        if is_first_task:
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
            file_notification_mapping[chat_id][file_name] = sent_msg.message_id
            
            if not state.processing:
                state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                
        else:
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
            
            queue_position = max(0, len(state.queue) - 1) if state.processing else len(state.queue)
            
            notification = (
                f"✅ File queued: `{file_name}`\n"
                f"Size: {content_size:,} characters{parts_info}\n"
                f"Operation: {state.operation.capitalize()}\n"
                f"Position in queue: {queue_position}\n\n"
                f"Will begin immediately after the ongoing task is completed"
            )
            
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=notification,
                disable_notification=True,
                parse_mode='Markdown'
            )
            file_notification_mapping[chat_id][file_name] = sent_msg.message_id
            
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

# ==================== PERIODIC CLEANUP TASK ====================
async def periodic_cleanup_task():
    """Periodically clean up old messages"""
    while True:
        try:
            cleanup_old_messages()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

# ==================== WEB SERVER & MAIN ====================
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
    
    # Add callback query handler (only for operation selection)
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
    
    logger.info("🤖 FileX Bot started with Admin Preview System")
    
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
