import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from googleapiclient.discovery import build
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import matplotlib.pyplot as plt
import requests
import time
import openai 
import os
from openai import OpenAI
from textblob import TextBlob
from nrclex import NRCLex


# Set the OpenAI API key directly from an environment variable or statically
load_dotenv()

youtube_api_key = os.getenv('YOUTUBE_API_KEY')
tt_and_ig_api_key = os.getenv('TIKTOK_AND_INSTAGRAM_API_KEY')

if 'sentiment_analysis_completed' not in st.session_state:
    st.session_state['sentiment_analysis_completed'] = False

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize session state for storing fetched comments data
if 'comments_data' not in st.session_state:
    st.session_state['comments_data'] = pd.DataFrame()

# Platform Detection and ID Extraction Functions
def detect_platform(url):
    if 'youtube.com' in url:
        return 'youtube'
    elif 'tiktok.com' in url:
        return 'tiktok'
    elif 'instagram.com' in url:
        return 'instagram'
    else:
        return None

def extract_youtube_id(url):
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def extract_tiktok_id(url):
    match = re.search(r'video/(\d+)', url)
    return match.group(1) if match else None

def extract_instagram_shortcode(url):
    match = re.search(r'(?:p|reel)/([a-zA-Z0-9_-]+)/', url)
    return match.group(1) if match else None


#plotting function
def plot_stacked_sentiment_proportions(df, selected_video_ids):
    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Filter the DataFrame for the selected video IDs
    filtered_df = df[df['Video ID'].isin(selected_video_ids)].set_index('Video ID')
    
    # Normalize the sentiments to sum up to 100%
    sentiments = ['% Positive', '% Neutral', '% Negative']
    filtered_df[sentiments] = filtered_df[sentiments].div(filtered_df[sentiments].sum(axis=1), axis=0) * 100

    # Plot the DataFrame as a stacked bar chart
    bars = filtered_df[sentiments].plot(kind='bar', stacked=True, color=['green', 'gainsboro', 'tomato'], ax=ax)

    # Adding the text labels on each bar
    for bar in bars.containers:
        ax.bar_label(bar, fmt='%.1f%%', label_type='center')
    
    # Set plot title and labels
    ax.set_title('Sentiment Proportions')
    ax.set_xlabel('Video ID')
    ax.set_ylabel('Proportions (%)')

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Adjust the layout and show the legend
    ax.legend(title="Sentiment")
    plt.tight_layout()

    # Return the matplotlib figure
    return fig


# Import Oswald font from Google Fonts and apply it to the main title
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

h1 {
    font-family: 'Oswald', sans-serif;
    color: #333; /* Change the color as needed */
    font-size: 48px; /* Adjust the size as needed */
}
</style>
""", unsafe_allow_html=True)

# Streamlit App Interface
st.title('Comment Sentiment Analysis')
st.write("Welcome! Please input a social media post URL (Available for YouTube, TikTok, or Instagram).")

# Input for the URL
input_urls = st.text_area('Enter post URLs, each on a new line')


# Function to scrape comments and return a DataFrame, now including video titles (YOUTUBE)
def scrape_comments_to_df(video_ids, youtube_api_key):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    all_comments_data = []
    
    for vid in video_ids:
        # Fetch the video title
        video_response = youtube.videos().list(part='snippet', id=vid).execute()
        video_title = video_response['items'][0]['snippet']['title'] if video_response['items'] else 'Unknown Title'

        next_page_token = None
        while True:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=vid,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                all_comments_data.append([
                    vid,
                    video_title,  # Add video title
                    comment['authorDisplayName'],
                    comment['textDisplay'],
                    comment['publishedAt'],
                    comment['likeCount'],
                    item['snippet']['totalReplyCount']
                ])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    return pd.DataFrame(all_comments_data, columns=['Video ID', 'Video Title', 'Name', 'Comment', 'Time', 'Likes', 'Reply Count'])

# Function to scrape comments and return a DataFrame (TIKTOK)
# Now includes TikTok creator's username and supports multiple video IDs
def scrape_tiktok_comments_to_df(aweme_ids, tt_and_ig_api_key):
    comments_url = "https://scraptik.p.rapidapi.com/list-comments"
    video_details_url = "https://scraptik.p.rapidapi.com/video"
    
    headers = {
        "X-RapidAPI-Key": tt_and_ig_api_key,
        "X-RapidAPI-Host": "scraptik.p.rapidapi.com"
    }

    all_comments_data = []
    creator_usernames = {}

    for aweme_id in aweme_ids:
        # Fetch the video creator's username
        video_response = requests.get(video_details_url, headers=headers, params={"aweme_id": aweme_id})
        creator_username = 'Unknown Creator'
        if video_response.status_code == 200:
            video_data = video_response.json()
            creator_username = video_data.get('author', {}).get('unique_id', 'Unknown Creator')
            creator_usernames[aweme_id] = creator_username

        cursor = "0"
        has_more = True

        while has_more:
            querystring = {"aweme_id": aweme_id, "count": "50", "cursor": cursor}
            response = requests.get(comments_url, headers=headers, params=querystring)

            if response.status_code == 200:
                comments_data = response.json()
                if 'comments' in comments_data:
                    for item in comments_data['comments']:
                        all_comments_data.append([
                            aweme_id,
                            creator_username,
                            item['user']['nickname'],
                            item['text']
                        ])
                    has_more = comments_data.get('has_more', False)
                    cursor = comments_data.get('cursor', '0')
                else:
                    break
            else:
                break

            # Add a 1-second delay before the next request
            time.sleep(1)

    comments_df = pd.DataFrame(all_comments_data, columns=['aweme_id', 'creator_username', 'username', 'comment'])
    comments_df.rename(columns={'aweme_id': 'Video ID', 'comment':'Comment'}, inplace=True)   
    
    return comments_df, creator_usernames

# Function to convert INSTAGRAM short codes to media IDs
def codes_to_media_ids(short_codes):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    code_to_media_id = {}
    for short_code in short_codes:
        media_id = 0
        for letter in short_code:
            media_id = (media_id * 64) + alphabet.index(letter)
        code_to_media_id[short_code] = media_id
    return code_to_media_id


# Function to scrape comments (INSTAGRAM)
def scrape_instagram_comments(short_codes, tt_and_ig_api_key):
    url = "https://rocketapi-for-instagram.p.rapidapi.com/instagram/media/get_comments"
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": tt_and_ig_api_key,
        "X-RapidAPI-Host": "rocketapi-for-instagram.p.rapidapi.com"
    }
    all_comments = []
    code_to_media_id = codes_to_media_ids(short_codes)
    for short_code, media_id in code_to_media_id.items():
        initial_payload = {"id": media_id, "min_id": None}
        while True:
            response = requests.post(url, json=initial_payload, headers=headers)
            data = response.json()
            body = data.get('response', {}).get('body', {})
            comments = body.get('comments', [])
            for comment in comments:
                all_comments.append([short_code, comment['user']['username'], comment['text']])
            has_more = body.get('has_more', False)
            if not has_more:
                break
            max_id = body.get('max_id')
            initial_payload['max_id'] = max_id
    return pd.DataFrame(all_comments, columns=['Video ID', 'username', 'Comment'])

# Function to perform SENTIMENT ANALYSIS (unchanged)
def sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()
    stopwords_list = stopwords.words('english')
    
    # Define a function to classify sentiment of each comment
    def classify_comment_sentiment(comment):
        # Preprocess the comment
        comment = re.sub(r'[^\w\s]', '', comment).lower()  # Remove punctuation and convert to lowercase
        tokens = word_tokenize(comment)  # Tokenize the comment
        filtered_tokens = [token for token in tokens if token not in stopwords_list]  # Remove stopwords
        # Get the sentiment score and classify
        sentiment_score = sid.polarity_scores(' '.join(filtered_tokens))
        if sentiment_score['compound'] > 0.05:
            return 'positive'
        elif sentiment_score['compound'] < -0.05:
            return 'negative'
        else:
            return 'neutral'

    # Apply the function to the 'Comment' column to create the 'Sentiment' column
    df['Sentiment'] = df['Comment'].apply(classify_comment_sentiment)
    
    
    # Aggregate sentiment counts by video ID
    video_sentiment_data = []
    for vid in df['Video ID'].unique():
        video_df = df[df['Video ID'] == vid]
        sentiment_counts = {
            'Video ID': vid,
            'Positive': len(video_df[video_df['Sentiment'] == 'positive']),
            'Neutral': len(video_df[video_df['Sentiment'] == 'neutral']),
            'Negative': len(video_df[video_df['Sentiment'] == 'negative'])
        }
        video_sentiment_data.append(sentiment_counts)
    
    sentiment_df = pd.DataFrame(video_sentiment_data)
    sentiment_df['Total Comments'] = sentiment_df['Positive'] + sentiment_df['Neutral'] + sentiment_df['Negative']
    sentiment_df['% Positive'] = (sentiment_df['Positive'] / sentiment_df['Total Comments']) * 100
    sentiment_df['% Neutral'] = (sentiment_df['Neutral'] / sentiment_df['Total Comments']) * 100
    sentiment_df['% Negative'] = (sentiment_df['Negative'] / sentiment_df['Total Comments']) * 100
    
    overall = sentiment_df[['Positive', 'Negative', 'Neutral', 'Total Comments']].sum()
    overall['Video ID'] = 'Overall'
    overall['% Positive'] = (overall['Positive'] / overall['Total Comments']) * 100
    overall['% Neutral'] = (overall['Neutral'] / overall['Total Comments']) * 100
    overall['% Negative'] = (overall['Negative'] / overall['Total Comments']) * 100
    sentiment_df = pd.concat([pd.DataFrame([overall]), sentiment_df], ignore_index=True)
    
    return df, sentiment_df


# Pull Comments Button
if st.button('Pull Comments'):
    if input_urls:
        urls = input_urls.split('\n')  # Splitting the input into a list of URLs
        all_comments_df = pd.DataFrame()  # Initialize an empty DataFrame to hold all comments
        for url in urls:
            if url.strip():  # Checking if the URL is not just whitespace
                platform = detect_platform(url)
                try:
                    if platform == 'youtube':
                        video_id = extract_youtube_id(url)
                        if video_id:
                            comments_df = scrape_comments_to_df([video_id], youtube_api_key)
                            all_comments_df = pd.concat([all_comments_df, comments_df])

                    elif platform == 'tiktok':
                        aweme_id = extract_tiktok_id(url)
                        if aweme_id:
                            tiktok_comments_df, _ = scrape_tiktok_comments_to_df([aweme_id], tt_and_ig_api_key)
                            all_comments_df = pd.concat([all_comments_df, tiktok_comments_df])

                    elif platform == 'instagram':
                        shortcode = extract_instagram_shortcode(url)
                        if shortcode:
                            instagram_comments_df = scrape_instagram_comments([shortcode], tt_and_ig_api_key)
                            all_comments_df = pd.concat([all_comments_df, instagram_comments_df])

                    else:
                        st.error(f'Invalid or unsupported URL: {url}')

                except Exception as e:
                    st.error(f'An error occurred while processing {url}: {e}')
        if not all_comments_df.empty:
            st.session_state['comments_data'] = all_comments_df  # Store the comments DataFrame in session state
        else:
            st.error('No comments were fetched. Please check the URLs and API Key.')
    else:
        st.error('Please enter at least one URL.')

# Directly display fetched comments if available
if 'comments_data' in st.session_state and not st.session_state['comments_data'].empty:
    st.write("Comments fetched successfully. You can now select videos for sentiment analysis. You can also download the dataset by hovering over the right corner of the table.")
    st.write(st.session_state['comments_data'])

# Perform Sentiment Analysis Button
if 'comments_data' in st.session_state and not st.session_state['comments_data'].empty:
    video_ids = st.session_state['comments_data']['Video ID'].unique()
    st.write('Which video IDs do you want to perform sentiment analysis on?')
    select_all = st.checkbox("Select All Videos", value=False)
    multi_select_placeholder = st.empty()
    video_ids_to_analyze = multi_select_placeholder.multiselect(
        '',  # The label is now empty since the question is asked above
        options=video_ids,
        default=video_ids if select_all else []
    )

    if st.button('Analyze Sentiments on Selected Videos'):
        if video_ids_to_analyze:
            # Filter comments based on selected video IDs
            filtered_comments_df = st.session_state['comments_data'][st.session_state['comments_data']['Video ID'].isin(video_ids_to_analyze)]
            modified_df, aggregated_df = sentiment_analysis(filtered_comments_df)
            st.session_state['comments_with_sentiment'] = modified_df
            st.session_state['aggregated_sentiment_results'] = aggregated_df
            st.session_state['sentiment_analysis_completed'] = True
            st.session_state['video_ids_to_analyze'] = video_ids_to_analyze  # Store the selected video IDs for later use

        else:
            st.error('Please select at least one video ID for sentiment analysis.')




#Directly display sentiment analysis results if analysis has been completed
if st.session_state.get('sentiment_analysis_completed', False):
    st.write("Sentiment Analysis Results:")
    st.write(st.session_state['comments_with_sentiment'])
    st.write(st.session_state['aggregated_sentiment_results'])
    #Assuming you store the figure in session_state or recreate it here
    fig = plot_stacked_sentiment_proportions(st.session_state['aggregated_sentiment_results'], video_ids_to_analyze)
    st.pyplot(fig)

def emotion_analysis(comments_df):
    # Process each comment for emotion analysis
    def analyze_emotion(comment):
        emotion_obj = NRCLex(comment)
        affect_frequencies = emotion_obj.affect_frequencies

        # Check if the comment is primarily classified as positive or negative
        if 'positive' in affect_frequencies and affect_frequencies['positive'] > 0:
            # If classified as positive, decide between love and joy based on their scores
            love_score = affect_frequencies.get('love', 0)
            joy_score = affect_frequencies.get('joy', 0)
            if love_score > joy_score:
                top_emotion = 'love'
            elif joy_score > love_score:
                top_emotion = 'joy'
            else:
                # If love and joy have the same score, choose one or use another logic
                top_emotion = 'love'  # Default to 'joy' if equal
        else:
            # For all other cases, get the most common emotion excluding 'positive' and 'negative'
            filtered_emotions = {emotion: score for emotion, score in affect_frequencies.items() if emotion not in ['positive', 'negative']}
            if filtered_emotions:
                top_emotion = max(filtered_emotions, key=filtered_emotions.get)
            else:
                top_emotion = 'None'  # Use 'None' if no emotion is detected

        return top_emotion
    
    comments_df['Emotion'] = comments_df['Comment'].apply(analyze_emotion)
    # Aggregate the results
    aggregated_emotion_results = comments_df.groupby(['Video ID', 'Emotion']).size().unstack(fill_value=0)
    return comments_df, aggregated_emotion_results

if 'comments_data' in st.session_state and not st.session_state['comments_data'].empty:
    video_ids = st.session_state['comments_data']['Video ID'].unique()
    st.write('Select video IDs for emotion analysis:')
    select_all = st.checkbox("Select All Videos", value=False, key="select_all_emotions")
    multi_select_placeholder = st.empty()
    video_ids_to_analyze = multi_select_placeholder.multiselect(
        '',  # The label is now empty since the question is asked above
        options=video_ids,
        default=video_ids if select_all else [],
        key="video_ids_emotion_analysis_multiselect"
    )

    if st.button('Analyze Emotions on Selected Videos', key="analyze_emotions_button"):
        if video_ids_to_analyze:
            filtered_comments_df = st.session_state['comments_data'][st.session_state['comments_data']['Video ID'].isin(video_ids_to_analyze)]
            modified_df, aggregated_df = emotion_analysis(filtered_comments_df)
            st.session_state['comments_with_emotion'] = modified_df
            st.session_state['aggregated_emotion_results'] = aggregated_df
            st.session_state['emotion_analysis_completed'] = True

        else:
            st.error('Please select at least one video ID for emotion analysis.')

# Function to plot aggregated emotion results
def plot_emotion_bar_chart(aggregated_df, video_ids_to_analyze):
    fig, ax = plt.subplots()
    # Flatten the DataFrame to plot individual emotions per video ID
    # First, reset index to use video IDs as a normal column
    aggregated_df = aggregated_df.reset_index()
    # Then, melt the DataFrame to have a single column for all emotions
    melted_df = aggregated_df.melt(id_vars=["Video ID"], var_name="Emotion", value_name="Number of Comments")
    # Filter based on selected video IDs if necessary
    if video_ids_to_analyze:
        melted_df = melted_df[melted_df["Video ID"].isin(video_ids_to_analyze)]

    import seaborn as sns
    sns.barplot(x="Number of Comments", y="Emotion", hue="Video ID", data=melted_df, ax=ax)
    
    plt.title('Aggregated Emotion Results')
    plt.xlabel('Count of Comments')
    plt.ylabel('Emotion')
    plt.legend(title='Video ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# Including the plotting function in the Streamlit UI
if st.session_state.get('emotion_analysis_completed', False):
    st.write("Emotion Analysis Results:")
    st.write(st.session_state['comments_with_emotion'])
    st.write(st.session_state['aggregated_emotion_results'])

    fig = plot_emotion_bar_chart(st.session_state['aggregated_emotion_results'], st.session_state.get('video_ids_to_analyze', []))
    st.pyplot(fig)

# Set the OpenAI API key directly from an environment variable or statically
load_dotenv()

client = OpenAI()
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set or invalid.")

def summarize_comments(df):
    summaries = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        sample_comments = df[df['Sentiment'] == sentiment]['Comment'].sample(n=min(len(df[df['Sentiment'] == sentiment]), 10), replace=False).tolist()
        comments_text = "\n".join(sample_comments)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following {sentiment} comments in a comprehensive 5 sentence paragraph: {comments_text}"}
        ]
        
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages = messages
            )
            summary_text = completion.choices[0].message.content.strip()
            summaries[sentiment] = summary_text
        except Exception as e:
            summaries[sentiment] = f"Failed to generate summary: {str(e)}"
    
    return summaries

# Summarization Button in Streamlit UI
if 'sentiment_analysis_completed' in st.session_state and st.session_state['sentiment_analysis_completed']:
    if st.button('Get a summary of what was being talked about!'):
        comments_with_sentiment = st.session_state.get('comments_with_sentiment', pd.DataFrame())
        if 'Sentiment' in comments_with_sentiment.columns:
            summaries = summarize_comments(comments_with_sentiment)
            for sentiment, summary in summaries.items():
                st.subheader(f"Summary of {sentiment.capitalize()} Comments:")
                st.write(summary)
                st.write("---")
        else:
            st.error("Sentiment analysis results are missing or do not contain a 'Sentiment' column.")

