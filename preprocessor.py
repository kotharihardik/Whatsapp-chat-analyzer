import re
import pandas as pd

# preprocessor.py

import re
from collections import defaultdict
from datetime import datetime

def npreprocess(chat_data):
    messages = []
    user_msg_map = defaultdict(list)
    
    for line in chat_data.split("\n"):
        match = re.match(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)", line)
        if match:
            date, time, user, msg = match.groups()
            messages.append({"date": date, "time": time, "user": user, "message": msg})
            user_msg_map[user].append(msg)
    
    return messages, user_msg_map


def preprocess(data):
    print("[INFO] Starting preprocessing...")

    # WhatsApp datetime pattern (24-hour or 12-hour with am/pm)
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?\s?-\s'
    print("[INFO] Regex pattern defined.")

    messages = re.split(pattern, data)
    # After splitting, messages[0] is likely an empty string or encryption notice
    if not messages[0].strip():
        messages = messages[1:]

    dates = []
    msg_list = []

    for i in range(0, len(messages) - 1, 4):
        date_str = f"{messages[i]} {messages[i+1]} {messages[i+2]}".strip()
        message = messages[i + 3].strip()
        msg_list.append(message)
        dates.append(date_str)

    print(f"[DEBUG] Found {len(msg_list)} messages and {len(dates)} dates.")

    df = pd.DataFrame({'user_message': msg_list, 'message_date': dates})
    print("[INFO] DataFrame created with user_message and message_date columns.")

    # Convert message_date to datetime
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y %I:%M %p', errors='coerce')
        if df['message_date'].isnull().any():
            df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y %I:%M %p', errors='coerce')
    except Exception as e:
        print(f"[ERROR] Date conversion failed: {e}")
    print("[INFO] Date conversion successful.")

    # Rename the column
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Split user and message
    print("[INFO] Splitting user and message...")
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r'^([\w\s]+?):\s', message)
        if len(entry) > 2:
            users.append(entry[1].strip())
            messages.append(entry[2].strip())
        else:
            users.append('group_notification')
            messages.append(entry[0].strip())

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    print("[INFO] User and message columns created.")

    # Extract datetime features safely
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['day_name'] = df['date'].dt.day_name()
        print("[INFO] Datetime features extracted successfully.")
    else:
        print("[ERROR] Failed extracting datetime features: Column 'date' is not datetime type.")
    
    df = df[df['user'] != 'group_notification']
    return df
