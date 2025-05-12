from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

import os
import requests
from dotenv import load_dotenv

extract = URLExtract()



load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")


API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# ✅ Function to get response from DeepSeek LLM
def get_response_from_llm(prompt, max_tokens=300):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek/deepseek-v3-0324",
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()





def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Number of messages
    num_messages = df.shape[0]

    # Number of words
    words = []
    for message in df['message']:
        if isinstance(message, str):
            words.extend(message.split())

    # Number of media messages
    num_media_messages = df[df['message'].str.lower() == '<media omitted>'].shape[0]

    # Number of links
    links = []
    for message in df['message']:
        if isinstance(message, str):
            links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    user_counts = df['user'].value_counts().head()
    user_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    user_percent.columns = ['name', 'percent']
    return user_counts, user_percent

def create_wordcloud(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r') as f:
            stop_words = f.read().splitlines()
    except FileNotFoundError:
        print("Warning: 'stop_hinglish.txt' file not found. Using default stop words.")
        stop_words = []

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'].str.lower() != '<media omitted>']

    def remove_stop_words(message):
        if isinstance(message, str):
            return " ".join([word for word in message.lower().split() if word not in stop_words])
        return ""

    temp['message'] = temp['message'].apply(remove_stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(temp['message'].str.cat(sep=" ")) if not temp.empty else None

def most_common_words(selected_user, df):
    try:
        with open('stop_hinglish.txt', 'r') as f:
            stop_words = f.read().splitlines()
    except FileNotFoundError:
        print("Warning: 'stop_hinglish.txt' file not found. Using default stop words.")
        stop_words = []

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'].str.lower() != '<media omitted>']

    words = []
    for message in temp['message']:
        if isinstance(message, str):
            words.extend([word for word in message.lower().split() if word not in stop_words])

    return pd.DataFrame(Counter(words).most_common(20))

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        if isinstance(message, str):
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'month_num' not in df.columns:
        df['month_num'] = df['date'].dt.month
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month_name()

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = [f"{row['month']}-{row['year']}" for _, row in timeline.iterrows()]
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'only_date' not in df.columns:
        df['only_date'] = df['date'].dt.date

    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'day_name' not in df.columns:
        df['day_name'] = df['date'].dt.day_name()

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month_name()

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if 'day_name' not in df.columns:
        df['day_name'] = df['date'].dt.day_name()

    if 'period' not in df.columns:
        df['period'] = df['date'].dt.hour.map(lambda x: f"{x}-{x+1}" if x != 23 else "23-00")

    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
