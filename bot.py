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

# ==================== GLOBAL STORAGE ====================
# Store mapping of instance_id -> message IDs for each file
file_instance_messages: Dict[str, List[int]] = {}
file_instance_notifications: Dict[str, List[int]] = {}
file_instances: Dict[str, Dict] = {}  # instance_id -> file info

# Store file history with instance tracking
file_history_by_chat: Dict[int, List[Dict]] = defaultdict(list)

# ==================== ENHANCED MESSAGE TRACKING SYSTEM ====================
class EnhancedMessageEntry:
    def __init__(self, chat_id: int, message_id: int, full_message: str, 
                 first_two_words: str, timestamp: datetime, instance_id: Optional[str] = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.full_message = full_message
        self.first_two_words = first_two_words
        self.timestamp = timestamp
        self.instance_id = instance_id
        self.words = self.extract_words(full_message)
    
    def extract_words(self, text: str) -> List[str]:
        clean_text = re.sub(r'https?://\S+', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text.split()
    
    def is_expired(self) -> bool:
        return datetime.now(UTC_PLUS_1) - self.timestamp > timedelta(hours=72)
    
    def __repr__(self):
        return f"EnhancedMessageEntry(chat={self.chat_id}, words={len(self.words)}, time={self.timestamp.strftime('%H:%M:%S')})"

enhanced_message_tracking: List[EnhancedMessageEntry] = []
admin_preview_mode: Set[int] = set()

# ==================== FILE INSTANCE TRACKING ====================
class FileInstance:
    """Track each file upload as a unique instance"""
    def __init__(self, filename: str, size: int, chat_id: int, timestamp: datetime, 
                 status: str = "uploaded", original_filename: Optional[str] = None):
        self.filename = filename  # May include (1), (2) suffix
        self.original_filename = original_filename or filename  # Original filename without suffix
        self.size = size
        self.chat_id = chat_id
        self.timestamp = timestamp
        self.status = status
        self.instance_id = self.generate_instance_id()
        self.message_ids: List[int] = []
        self.notification_ids: List[int] = []
    
    def generate_instance_id(self) -> str:
        """Generate unique instance ID based on filename, size, chat_id, and timestamp"""
        timestamp_str = self.timestamp.strftime('%Y%m%d%H%M%S')
        return f"{self.chat_id}_{self.original_filename}_{self.size}_{timestamp_str}"
    
    def is_recent(self) -> bool:
        return datetime.now(UTC_PLUS_1) - self.timestamp <= timedelta(hours=72)
    
    def get_display_name(self) -> str:
        """Get display name with instance suffix if needed"""
        return self.filename
    
    def __repr__(self):
        return f"FileInstance({self.filename}, {self.size} bytes, {self.timestamp.strftime('%H:%M:%S')}, status={self.status}, instance={self.instance_id})"

file_instances_list: List[FileInstance] = []

# ==================== INSTANCE MANAGEMENT ====================
def create_file_instance(filename: str, size: int, chat_id: int, 
                        status: str = "uploaded") -> FileInstance:
    """Create a new file instance, adding suffix if same file exists"""
    
    # Find existing instances with same original filename and size
    existing_instances = [
        inst for inst in file_instances_list 
        if inst.original_filename == filename and inst.size == size and inst.chat_id == chat_id
    ]
    
    # Count how many instances are not deleted
    active_instances = [inst for inst in existing_instances if inst.status != 'deleted']
    
    # If there are active instances, add suffix
    if active_instances:
        instance_number = len(active_instances) + 1
        display_filename = f"{filename} ({instance_number})"
    else:
        display_filename = filename
    
    # Create new instance
    instance = FileInstance(
        filename=display_filename,
        original_filename=filename,
        size=size,
        chat_id=chat_id,
        timestamp=datetime.now(UTC_PLUS_1),
        status=status
    )
    
    file_instances_list.append(instance)
    file_instances[instance.instance_id] = {
        'instance': instance,
        'messages': [],
        'notifications': []
    }
    
    logger.info(f"Created file instance: {instance.instance_id} as '{display_filename}'")
    return instance

def find_file_instances_by_name(chat_id: int, filename: str) -> List[FileInstance]:
    """Find all instances by original filename (excluding deleted)"""
    return [
        inst for inst in file_instances_list 
        if inst.original_filename == filename and inst.chat_id == chat_id and inst.status != 'deleted'
    ]

def find_file_instance_by_id(instance_id: str) -> Optional[FileInstance]:
    """Find instance by ID"""
    for inst in file_instances_list:
        if inst.instance_id == instance_id:
            return inst
    return None

def check_duplicate_file(filename: str, size: int, chat_id: int) -> Optional[FileInstance]:
    """Check if same file (name + size) was uploaded within last 72 hours"""
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    
    for inst in file_instances_list:
        if (inst.original_filename == filename and 
            inst.size == size and 
            inst.chat_id == chat_id and 
            inst.timestamp >= cutoff and
            inst.status != 'deleted'):
            return inst
    
    return None

def add_message_to_instance(instance_id: str, message_id: int, is_notification: bool = False):
    """Add message ID to instance tracking"""
    if instance_id not in file_instances:
        return
    
    if is_notification:
        file_instances[instance_id]['notifications'].append(message_id)
    else:
        file_instances[instance_id]['messages'].append(message_id)
    
    # Also update the FileInstance object
    instance = file_instances[instance_id]['instance']
    if is_notification:
        instance.notification_ids.append(message_id)
    else:
        instance.message_ids.append(message_id)

def cleanup_old_instances_and_messages():
    """Remove old instances and messages"""
    global enhanced_message_tracking, file_instances_list
    
    # Clean up old messages
    cutoff = datetime.now(UTC_PLUS_1) - timedelta(hours=72)
    
    # Remove expired messages
    initial_msg_count = len(enhanced_message_tracking)
    enhanced_message_tracking = [entry for entry in enhanced_message_tracking 
                               if not entry.is_expired()]
    
    # Remove expired instances
    initial_inst_count = len(file_instances_list)
    file_instances_list = [inst for inst in file_instances_list if inst.is_recent()]
    
    # Clean up file_instances dict
    expired_ids = [inst_id for inst_id in file_instances 
                  if not find_file_instance_by_id(inst_id)]
    for inst_id in expired_ids:
        del file_instances[inst_id]
    
    removed_msgs = initial_msg_count - len(enhanced_message_tracking)
    removed_inst = initial_inst_count - len(file_instances_list)
    
    if removed_msgs > 0 or removed_inst > 0:
        logger.info(f"Cleaned up {removed_msgs} expired messages and {removed_inst} expired instances")

# ==================== MESSAGE TRACKING ====================
def extract_first_two_words(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return ""
    
    clean_text = re.sub(r'https?://\S+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    words = clean_text.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    elif len(words) == 1:
        return words[0]
    else:
        return ""

def track_enhanced_message(chat_id: int, message_id: int, message_text: str, 
                          instance_id: Optional[str] = None):
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
            instance_id=instance_id
        )
        
        enhanced_message_tracking.append(entry)
        
    except Exception as e:
        logger.error(f"Error tracking enhanced message: {e}")

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

# ==================== FILE HISTORY MANAGEMENT ====================
def update_file_history(chat_id: int, instance_id: str, status: str, 
                       parts_count: int = 0, messages_count: int = 0):
    """Update history for a file instance"""
    instance = find_file_instance_by_id(instance_id)
    if not instance:
        return
    
    # Update instance status
    instance.status = status
    
    # Add to chat history
    entry = {
        'instance_id': instance_id,
        'filename': instance.get_display_name(),
        'original_filename': instance.original_filename,
        'size': instance.size,
        'timestamp': datetime.now(UTC_PLUS_1),
        'status': status,
        'parts_count': parts_count,
        'messages_count': messages_count
    }
    
    file_history_by_chat[chat_id].append(entry)
    
    # Keep only last 100 entries per chat
    if len(file_history_by_chat[chat_id]) > 100:
        file_history_by_chat[chat_id] = file_history_by_chat[chat_id][-100:]

# ==================== USER STATE MANAGEMENT ====================
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
        self.processing_start_time = None
        
        self.paused = False
        self.paused_at = None
        self.paused_progress = None
        
        self.waiting_duplicate_confirmation = False
        self.pending_duplicate_file = None
        
        self.waiting_file_selection = False
        self.pending_delete_instances = []
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
    
    def remove_task_by_instance_id(self, instance_id: str):
        new_queue = deque(maxlen=MAX_QUEUE_SIZE)
        for task in self.queue:
            if task.get('instance_id') != instance_id:
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
                    logger.warning(f"Cleaning up stuck task for chat {chat_id}")
                    state.cancel_current_task()
        except Exception as e:
            logger.error(f"Error cleaning up stuck task for chat {chat_id}: {e}")

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
                                    retries: int = 5, instance_id: Optional[str] = None,
                                    is_notification: bool = False) -> bool:
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
            
            # Track message
            if sent_message and sent_message.text:
                track_enhanced_message(chat_id, sent_message.message_id, sent_message.text, instance_id)
            
            # Track in instance
            if instance_id and sent_message:
                add_message_to_instance(instance_id, sent_message.message_id, is_notification)
            
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for chat {chat_id}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send message after {retries} attempts for chat {chat_id}")
    return False

async def send_chunks_immediately(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                chunks: List[str], instance: FileInstance, 
                                message_thread_id: Optional[int] = None) -> bool:
    try:
        total_messages_sent = 0
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, 
                                               instance_id=instance.instance_id):
                total_messages_sent += 1
                
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for `{instance.get_display_name()}`")
                return False
        
        state = await get_user_state_safe(chat_id)
        state.last_send = datetime.now(UTC_PLUS_1)
        
        if total_messages_sent > 0:
            completion_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"✅ Completed: `{instance.get_display_name()}`\n📊 Sent {total_messages_sent} message{'s' if total_messages_sent > 1 else ''}",
                disable_notification=True,
                parse_mode='Markdown'
            )
            add_message_to_instance(instance.instance_id, completion_msg.message_id, True)
            
            update_file_history(chat_id, instance.instance_id, 'completed', messages_count=total_messages_sent)
            return True
        else:
            logger.error(f"No chunks sent for `{instance.get_display_name()}`")
            return False
        
    except Exception as e:
        logger.error(f"Error in send_chunks_immediately: {e}")
        return False

async def send_large_content_part(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                                part: str, part_num: int, total_parts: int, 
                                instance: FileInstance, message_thread_id: Optional[int] = None) -> int:
    try:
        chunks = split_into_telegram_chunks_without_cutting_words(part, TELEGRAM_MESSAGE_LIMIT)
        total_messages_in_part = 0
        
        if total_parts > 1:
            part_header_msg = await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"📄 `{instance.get_display_name()}` - Part {part_num}/{total_parts}",
                disable_notification=True,
                parse_mode='Markdown'
            )
            add_message_to_instance(instance.instance_id, part_header_msg.message_id, True)
            total_messages_in_part += 1
        
        for i, chunk in enumerate(chunks, 1):
            if await send_telegram_message_safe(chat_id, context, chunk, message_thread_id, 
                                               instance_id=instance.instance_id):
                total_messages_in_part += 1
                
                if i < len(chunks):
                    await asyncio.sleep(MESSAGE_DELAY)
            else:
                logger.error(f"Failed to send chunk {i} for part {part_num} of `{instance.get_display_name()}`")
                return 0
        
        return total_messages_in_part
        
    except Exception as e:
        logger.error(f"Error sending large content part: {e}")
        return 0

async def send_with_intervals(chat_id: int, context: ContextTypes.DEFAULT_TYPE, 
                            parts: List[str], instance: FileInstance, state: UserState, 
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
                chat_id, context, part, i, total_parts, instance, message_thread_id
            )
            
            if not messages_in_part:
                logger.error(f"Failed to send part {i} of `{instance.get_display_name()}`")
                return False
            
            total_messages_sent += messages_in_part
            
            state.last_send = datetime.now(UTC_PLUS_1)
            
            if i < total_parts:
                await asyncio.sleep(SEND_INTERVAL * 60)
        
        completion_msg = await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"✅ Completed: `{instance.get_display_name()}`\n📊 Sent {total_parts} part{'s' if total_parts > 1 else ''} ({total_messages_sent} messages total)",
            disable_notification=True,
            parse_mode='Markdown'
        )
        add_message_to_instance(instance.instance_id, completion_msg.message_id, True)
        
        update_file_history(chat_id, instance.instance_id, 'completed', parts_count=total_parts, messages_count=total_messages_sent)
        
        return True
        
    except asyncio.CancelledError:
        logger.info(f"Task cancelled for chat {chat_id}")
        raise
    except Exception as e:
        logger.error(f"Error in send_with_intervals: {e}")
        return False

async def delete_instance_messages(chat_id: int, instance_id: str, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Delete all messages for a file instance"""
    deleted_count = 0
    
    try:
        if instance_id not in file_instances:
            logger.error(f"Instance {instance_id} not found in tracking")
            return 0
        
        instance_data = file_instances[instance_id]
        instance = instance_data['instance']
        
        # Delete content messages
        for msg_id in instance.message_ids:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                deleted_count += 1
                logger.info(f"Deleted content message {msg_id} for instance {instance_id}")
            except Exception as e:
                logger.error(f"Failed to delete content message {msg_id}: {e}")
        
        # Delete notification messages
        for msg_id in instance.notification_ids:
            try:
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=msg_id
                )
                deleted_count += 1
                logger.info(f"Deleted notification message {msg_id} for instance {instance_id}")
            except Exception as e:
                logger.error(f"Failed to delete notification message {msg_id}: {e}")
        
        # Clean up tracking data
        if instance_id in file_instances:
            del file_instances[instance_id]
        
        # Mark instance as deleted
        instance.status = 'deleted'
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting messages for instance {instance_id}: {e}")
        return deleted_count

# ==================== QUEUE PROCESSING ====================
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
            instance_id = file_info['instance_id']
            instance = find_file_instance_by_id(instance_id)
            
            if not instance:
                logger.error(f"Instance {instance_id} not found, skipping")
                state.queue.popleft()
                continue
                
            file_message_thread_id = file_info.get('message_thread_id', message_thread_id)
            
            update_file_history(chat_id, instance_id, 'running', 
                              parts_count=len(file_info.get('parts', [])),
                              messages_count=len(file_info.get('chunks', [])))
            
            # Send starting notification
            if file_info.get('requires_intervals', False):
                sending_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=file_message_thread_id,
                    text=f"📤 Sending: `{instance.get_display_name()}`\n"
                         f"📊 Total parts: {len(file_info['parts'])}\n"
                         f"⏰ Interval: {SEND_INTERVAL} minutes between parts",
                    disable_notification=True,
                    parse_mode='Markdown'
                )
                add_message_to_instance(instance_id, sending_msg.message_id, True)
            else:
                sending_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=file_message_thread_id,
                    text=f"📤 Sending: `{instance.get_display_name()}`\n"
                         f"📊 Total messages: {len(file_info['chunks'])}",
                    disable_notification=True,
                    parse_mode='Markdown'
                )
                add_message_to_instance(instance_id, sending_msg.message_id, True)
            
            success = False
            if file_info.get('requires_intervals', False):
                success = await send_with_intervals(
                    chat_id, context, 
                    file_info['parts'], 
                    instance,
                    state,
                    file_message_thread_id
                )
            else:
                success = await send_chunks_immediately(
                    chat_id, context,
                    file_info['chunks'],
                    instance,
                    file_message_thread_id
                )
            
            # Remove from queue
            if state.queue and state.queue[0]['instance_id'] == instance_id:
                state.queue.popleft()
                
                if not success:
                    failed_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=file_message_thread_id,
                        text=f"❌ Failed to send: `{instance.get_display_name()}`\nPlease try uploading again.",
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    add_message_to_instance(instance_id, failed_msg.message_id, True)
            
            state.current_parts = []
            state.current_index = 0
            
            # Wait before next file
            if state.queue and not state.cancel_requested:
                next_instance = find_file_instance_by_id(state.queue[0]['instance_id'])
                if next_instance:
                    next_file_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=f"⏰ **Queue Interval**\n\n"
                             f"Next file `{next_instance.get_display_name()}` will start in 2 minutes...",
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
            instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
            if instance:
                add_message_to_instance(instance.instance_id, error_msg.message_id, True)
    finally:
        state.processing = False
        state.processing_task = None
        state.current_parts = []
        state.current_index = 0
        state.paused = False
        state.paused_progress = None
        state.processing_start_time = None

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
    
    message_thread_id = query.message.message_thread_id
    state = await get_user_state_safe(chat_id)
    
    # Handle duplicate file confirmation
    if state.waiting_duplicate_confirmation:
        if query.data == 'dup_yes':
            state.waiting_duplicate_confirmation = False
            if state.pending_duplicate_file:
                file_info = state.pending_duplicate_file
                state.pending_duplicate_file = None
                
                # Create new instance for duplicate
                instance = create_file_instance(file_info['name'], file_info['size'], chat_id, "uploaded")
                
                # Update file_info with instance_id
                file_info['instance_id'] = instance.instance_id
                file_info['display_name'] = instance.get_display_name()
                
                # Add to queue
                queue_size = await add_to_queue_safe(state, file_info)
                queue_position = get_queue_position_safe(state)
                
                should_start_processing = not state.processing and queue_size == 1
                
                if should_start_processing:
                    notification = f"✅ **Duplicate file accepted**\n\nFile: `{instance.get_display_name()}`\nSize: {file_info['size']:,} characters\n\n🟢 Starting Your Task"
                    
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=notification,
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    add_message_to_instance(instance.instance_id, sent_msg.message_id, True)
                    
                    if not state.processing:
                        state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                else:
                    notification = f"✅ **Duplicate file queued**\n\nFile: `{instance.get_display_name()}`\nSize: {file_info['size']:,} characters\nPosition in queue: {queue_position}"
                    
                    sent_msg = await context.bot.send_message(
                        chat_id=chat_id,
                        message_thread_id=message_thread_id,
                        text=notification,
                        disable_notification=True,
                        parse_mode='Markdown'
                    )
                    add_message_to_instance(instance.instance_id, sent_msg.message_id, True)
                
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
    
    # Handle file selection for deletion
    if state.waiting_file_selection:
        if query.data == 'del_cancel':
            state.waiting_file_selection = False
            state.pending_delete_instances = []
            state.pending_delete_filename = None
            
            await query.edit_message_text(
                text="❌ **Operation cancelled**\n\nNo files were deleted.",
                parse_mode='Markdown'
            )
            return
        elif query.data.startswith('del_file_'):
            file_index = int(query.data.split('_')[2])
            if 0 <= file_index < len(state.pending_delete_instances):
                selected_instance = state.pending_delete_instances[file_index]
                state.waiting_file_selection = False
                
                # Delete messages for this instance
                deleted_count = await delete_instance_messages(chat_id, selected_instance.instance_id, context)
                
                # Update history
                update_file_history(chat_id, selected_instance.instance_id, 'deleted')
                
                # Remove from queue if present
                state.remove_task_by_instance_id(selected_instance.instance_id)
                
                await query.edit_message_text(
                    text=f"🗑️ `{selected_instance.get_display_name()}` content deleted\nMessages removed: {deleted_count}",
                    parse_mode='Markdown'
                )
                
                # Check if we need to start processing next task
                if state.processing and state.queue and not state.processing:
                    next_instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
                    if next_instance:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            message_thread_id=message_thread_id,
                            text=f"🔄 **Moving to next task**\n\nStarting next task: `{next_instance.get_display_name()}`",
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
            instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
            if instance:
                status_lines.append(f"📄 **Current file:** `{instance.get_display_name()}`")
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
        entry for entry in file_history_by_chat.get(chat_id, [])
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
        instance = find_file_instance_by_id(file_info.get('instance_id', ''))
        if not instance:
            continue
            
        parts_info = ""
        if 'chunks' in file_info:
            parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
        elif 'parts' in file_info:
            parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
        
        if state.processing and i == 1:
            queue_text += f"▶️ **ACTIVE** - `{instance.get_display_name()}`\n"
        else:
            actual_position = i - 1 if state.processing else i
            queue_text += f"{actual_position}. `{instance.get_display_name()}`\n"
        
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
    
    if state.queue:
        instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
        if instance:
            update_file_history(chat_id, instance.instance_id, 'paused', 
                              parts_count=len(state.queue[0].get('parts', [])),
                              messages_count=len(state.queue[0].get('chunks', [])))
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="⏸️ **Task Paused**\n\n"
             "Task has been paused.\n"
             "Progress saved.\n\n"
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
    
    if state.queue:
        instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
        if instance:
            update_file_history(chat_id, instance.instance_id, 'running', 
                              parts_count=len(state.queue[0].get('parts', [])),
                              messages_count=len(state.queue[0].get('chunks', [])))
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text="▶️ **Task Resumed**\n\n"
             "Resuming task from where it was paused.\n"
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
    
    if state.queue:
        instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
        if instance:
            update_file_history(chat_id, instance.instance_id, 'skipped')
    
    state.skip()
    
    if state.queue:
        next_instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
        if next_instance:
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"⏭️ **Task Skipped**\n\n"
                     f"Starting next task: `{next_instance.get_display_name()}`",
                parse_mode='Markdown'
            )
            
            if not state.processing:
                state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="⏭️ **Task Skipped**\n\n"
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
            instance = find_file_instance_by_id(file_info.get('instance_id', ''))
            if instance:
                update_file_history(chat_id, instance.instance_id, 'cancelled')
    
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

async def handle_deletion_filename(chat_id: int, message_thread_id: Optional[int], filename: str, 
                                  context: ContextTypes.DEFAULT_TYPE, state: UserState):
    """Handle filename input for deletion"""
    
    cleanup_old_instances_and_messages()
    
    # Find all instances with this original filename (excluding deleted)
    instances = find_file_instances_by_name(chat_id, filename)
    
    if not instances:
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
    if len(instances) == 1:
        instance = instances[0]
        deleted_count = await delete_instance_messages(chat_id, instance.instance_id, context)
        
        update_file_history(chat_id, instance.instance_id, 'deleted')
        state.remove_task_by_instance_id(instance.instance_id)
        
        await context.bot.send_message(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"🗑️ `{instance.get_display_name()}` content deleted\nMessages removed: {deleted_count}",
            parse_mode='Markdown'
        )
        
        # Check if we need to start processing next task
        if state.processing and state.queue and not state.processing:
            next_instance = find_file_instance_by_id(state.queue[0].get('instance_id', ''))
            if next_instance:
                await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=f"🔄 **Moving to next task**\n\nStarting next task: `{next_instance.get_display_name()}`",
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
    state.pending_delete_instances = instances
    state.pending_delete_filename = filename
    
    # Create message with inline buttons - SIMPLE (File 1), (File 2), etc.
    keyboard = []
    for i, instance in enumerate(instances):
        # Simple button label: (File 1), (File 2), etc.
        button_text = f"(File {i+1})"
        keyboard.append([
            InlineKeyboardButton(button_text, callback_data=f"del_file_{i}")
        ])
    
    keyboard.append([InlineKeyboardButton("🚫 Cancel", callback_data="del_cancel")])
    
    # Format message
    message = f"📋 **Multiple Files Found**\n\n"
    message += f"Found {len(instances)} active files named `{filename}`:\n\n"
    
    for i, instance in enumerate(instances):
        time_str = instance.timestamp.strftime('%H:%M:%S')
        status_info = instance.status.capitalize()
        
        # Try to get message count from history
        message_count = len(instance.message_ids)
        count_info = f" ({message_count} messages)" if message_count > 0 else ""
        
        message += f"{i+1}. {instance.get_display_name()} - {time_str} ({status_info}{count_info})\n"
    
    message += "\nWhich file do you want to delete?"
    
    await context.bot.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        text=message,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# ==================== MESSAGE HANDLERS ====================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_authorization(update, context):
        return
    
    chat_id = update.effective_chat.id
    state = await get_user_state_safe(chat_id)
    
    # Check if admin is in preview mode
    user_id = update.effective_user.id
    if user_id in admin_preview_mode:
        # Handle admin preview (simplified for now)
        return
    
    # Check if waiting for filename (deletion flow)
    if state.waiting_for_filename:
        message_thread_id = update.effective_message.message_thread_id
        filename = update.message.text.strip()
        
        # Handle cancel text
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
    state.pending_delete_instances = []
    state.pending_delete_filename = None
    
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
        duplicate_instance = check_duplicate_file(file_name, content_size, chat_id)
        
        if duplicate_instance and duplicate_instance.is_recent():
            # Get status of the old file
            old_status = duplicate_instance.status
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
            
            time_diff = datetime.now(UTC_PLUS_1) - duplicate_instance.timestamp
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
            message_count = len(duplicate_instance.message_ids)
            if message_count > 0:
                status_info += f" ({message_count} messages)"
            
            await context.bot.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"⚠️ **Duplicate File Detected**\n\n"
                     f"File: `{duplicate_instance.get_display_name()}`\n"
                     f"Size: {content_size:,} bytes\n\n"
                     f"This exact file was already processed {hours_ago:.1f} hours ago.\n"
                     f"{status_info}\n\n"
                     f"Do you want to post it again?\n\n"
                     f"✅ Yes, Post  ❌ No, Don't",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )
            return
        
        # No duplicate - create new instance
        instance = create_file_instance(file_name, content_size, chat_id, "uploaded")
        
        # Prepare file info
        if content_size <= CONTENT_LIMIT_FOR_INTERVALS:
            chunks = split_into_telegram_chunks_without_cutting_words(content, TELEGRAM_MESSAGE_LIMIT)
            file_info = {
                'instance_id': instance.instance_id,
                'name': file_name,
                'display_name': instance.get_display_name(),
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
                'instance_id': instance.instance_id,
                'name': file_name,
                'display_name': instance.get_display_name(),
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
        
        should_start_processing = not state.processing and queue_size == 1
        
        if should_start_processing:
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
                notification = (
                    f"✅ File accepted: `{instance.get_display_name()}`\n"
                    f"Size: {content_size:,} characters{parts_info}\n"
                    f"Operation: {state.operation.capitalize()}\n\n"
                    f"🟢 Starting Your Task"
                )
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
                notification = (
                    f"✅ File accepted: `{instance.get_display_name()}`\n"
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
            add_message_to_instance(instance.instance_id, sent_msg.message_id, True)
            
            if not state.processing:
                state.processing_task = asyncio.create_task(process_queue(chat_id, context, message_thread_id))
                
        else:
            if 'chunks' in file_info:
                parts_info = f" ({len(file_info['chunks'])} messages)" if len(file_info['chunks']) > 1 else ""
            else:
                parts_info = f" ({len(file_info['parts'])} parts)" if len(file_info['parts']) > 1 else ""
            
            notification = (
                f"✅ File queued: `{instance.get_display_name()}`\n"
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
            add_message_to_instance(instance.instance_id, sent_msg.message_id, True)
            
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

# ==================== ADMIN COMMANDS (SIMPLIFIED) ====================
async def adminpreview_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin_authorization(update, context):
        return
    
    user_id = update.effective_user.id
    admin_preview_mode.add(user_id)
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text="🔍 **Admin Preview Mode Activated**\n\n"
             "Preview functionality will be implemented separately.",
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
             "Preview mode has been deactivated.",
        parse_mode='Markdown'
    )

async def adminstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin_authorization(update, context):
        return
    
    cleanup_old_instances_and_messages()
    
    total_instances = len([inst for inst in file_instances_list if inst.status != 'deleted'])
    total_messages = len(enhanced_message_tracking)
    
    message = f"📊 **Admin Statistics**\n\n"
    message += f"**Total file instances:** {total_instances}\n"
    message += f"**Total tracked messages:** {total_messages}\n"
    message += f"**Active preview sessions:** {len(admin_preview_mode)}\n"
    message += f"**Active user sessions:** {len(user_states)}\n\n"
    message += f"Data auto-deletes after 72 hours."
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        message_thread_id=update.effective_message.message_thread_id,
        text=message,
        parse_mode='Markdown'
    )

# ==================== PERIODIC CLEANUP ====================
async def periodic_cleanup_task():
    while True:
        try:
            cleanup_old_instances_and_messages()
            cleanup_stuck_tasks()
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            await asyncio.sleep(300)

# ==================== HEALTH ENDPOINT ====================
async def health_handler(request):
    return web.Response(
        text=json.dumps({
            "status": "running",
            "service": "filex-bot",
            "active_users": len(user_states),
            "file_instances": len([inst for inst in file_instances_list if inst.status != 'deleted']),
            "tracked_messages": len(enhanced_message_tracking)
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

# ==================== MAIN ====================
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
