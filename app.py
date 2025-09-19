import pandas as pd
import os

# The file name of your mock dataset
file_name = 'mock_comments.csv'

# Check if the file exists in the same directory as the script
if os.path.exists(file_name):
    print(f"File '{file_name}' found. Loading data...")
    # Load the CSV file into a DataFrame
    # We specify the delimiter '|~|', no header, and column names
    df = pd.read_csv(file_name, sep='\|~\|', engine='python', header=None,
                     names=['comment_id', 'stakeholder_type', 'provision_number', 'comment_text'])
    
    print("\nDataFrame loaded successfully!")
    print("\nFirst 5 rows of your data:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

else:
    print(f"Error: The file '{file_name}' was not found. Please ensure it is in the same directory as this script.")

import re
from collections import Counter
from transformers import pipeline
# If you decide to use Google's Gemini API, you'll need to install it first:
# pip install google-generativeai

# --- Step 2: AI Pipeline (Sentiment Analysis and Summarization) ---
print("\nPerforming sentiment analysis...")
try:
    # Use a pre-trained sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis")
    df['sentiment'] = df['comment_text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    print("Sentiment analysis complete!")
except Exception as e:
    print(f"Error during sentiment analysis: {e}")
    # You can add a fallback for the hackathon if the API fails
    df['sentiment'] = 'NEUTRAL'

print("\nPerforming summarization...")
def summarize_text(text):
    # This is a placeholder for the actual LLM API call.
    # For the hackathon, you would integrate a service like Gemini here.
    # For now, this will give you a mock-up to build your frontend with.
    if len(text) > 50:
        return f"Summary: {text[:50].strip()}..."
    return f"Summary: {text.strip()}."
df['ai_summary'] = df['comment_text'].apply(summarize_text)
print("Summarization complete!")

# --- Step 3: Data Aggregation for the Dashboard ---
print("\nAggregating data for dashboard...")

# Map sentiment labels to numbers for aggregation
sentiment_mapping = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

# Calculate sentiment percentages and controversy score per provision
provision_summary = df.groupby('provision_number').agg(
    comment_volume=('comment_id', 'count'),
    total_sentiment_score=('sentiment_score', 'sum'),
    negative_comments=('sentiment', lambda x: (x == 'NEGATIVE').sum()),
    positive_comments=('sentiment', lambda x: (x == 'POSITIVE').sum()),
    neutral_comments=('sentiment', lambda x: (x == 'NEUTRAL').sum())
).reset_index()

# Calculate the sentiment percentage
provision_summary['negative_percentage'] = (provision_summary['negative_comments'] / provision_summary['comment_volume']) * 100
provision_summary['positive_percentage'] = (provision_summary['positive_comments'] / provision_summary['comment_volume']) * 100

# Create the "controversy" score to rank the clauses for the Priority Worklist
# This is a key "winning" feature
provision_summary['controversy_score'] = provision_summary['comment_volume'] * (provision_summary['negative_percentage'] / 100)
provision_summary = provision_summary.sort_values(by='controversy_score', ascending=False)

# Extract keywords from negative comments for the Smart Word Cloud
keywords_dict = {}
for provision in provision_summary['provision_number'].unique():
    negative_comments = df[(df['provision_number'] == provision) & (df['sentiment'] == 'NEGATIVE')]['comment_text']
    # Define a list of keywords to search for
    keywords_list = ['surveillance', 'monopoly', 'arbitrary power', 'data sharing concerns',
                     'excessive fine', 'undue burden', 'stifles innovation', 'disproportionate',
                     'ambiguous', 'lacks clarity', 'vision is flawed', 'operational burden',
                     'hinders global business', 'restrictive', 'unnecessary friction',
                     'cumbersome process', 'lack of transparency', 'biased towards government']
    
    # Use regular expressions to find keywords in the text
    all_words = " ".join(negative_comments).lower()
    found_keywords = re.findall(r'\b(?:' + '|'.join(re.escape(k) for k in keywords_list) + r')\b', all_words)
    
    # Get top 3-5 unique keywords
    word_counts = Counter(found_keywords)
    top_keywords = [word for word, count in word_counts.most_common(5)]
    keywords_dict[provision] = top_keywords

provision_summary['key_concerns'] = provision_summary['provision_number'].map(keywords_dict)

print("\nFinal Processed Data (Ready for Frontend):")
print(provision_summary.to_string())

print("\nBackend processing complete. The data is ready for the frontend.")