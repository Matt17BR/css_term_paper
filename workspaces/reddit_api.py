# %%
import os
from dotenv import load_dotenv 
load_dotenv()

# %%
# %pip install praw
# %pip install google-genai
# %pip install google-generativeai
# %pip install ipywidgets

# %%
# import google.generativeai as genai
# import os

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# # Create the model
# generation_config = {
#   "temperature": 0, # controls randomness. 0 = most deterministic (always selects highest probability token).
#   "top_p": 0, # nucleus sampling: limits token selection to the most probable. 0 = most deterministic (used when temperature > 0).
#   "top_k": 1, # restricts to top 'k' tokens. 1 = most deterministic (used when temperature > 0).
#   "max_output_tokens": 8192,
#   "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#   model_name="gemini-exp-1206",
#   generation_config=generation_config,
# )

# %%
import base64
import os
from google import genai
from google.genai import types

def generate(prompt):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0, # controls randomness. 0 = most deterministic (always selects highest probability token)
        top_p=0, # nucleus sampling: limits token selection to the most probable. 0 = most deterministic (used when temperature > 0)
        top_k=1, # restricts to top 'k' tokens. 1 = most deterministic (used when temperature > 0)
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    complete_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        complete_response += chunk.text

    return complete_response

# %%
import praw
import pandas as pd

# Replace with your actual credentials
reddit = praw.Reddit(
    client_id=os.getenv("PRAW_CLIENT_ID"),
    client_secret=os.getenv("PRAW_CLIENT_SECRET"),
    user_agent=os.getenv("PRAW_USER_AGENT"),
    username=os.getenv("PRAW_USERNAME"),
    password=os.getenv("PRAW_PASSWORD")
)

# Fetch a large subset of popular subreddits (large limit makes this representative of the largest overall subreddits by subscribers, check: https://gummysearch.com/tools/top-subreddits/)
subreddits = list(reddit.subreddits.popular(limit=1000))

# Create a DataFrame using list comprehension for better performance
subs_df = pd.DataFrame([{
    "Name": subreddit.display_name,
    "Subscribers": subreddit.subscribers,
    "Description": subreddit.public_description,
    "Over 18": subreddit.over18,
    "Submission Type": subreddit.submission_type
} for subreddit in subreddits]).sort_values(by="Subscribers", ascending=False, ignore_index=True)

# Print the top 10
subs_df.head(10)

# %%
import ast

response = generate("What are some keywords I can use to create a list of subreddits which are likely to be influenced by bots because of their controversial nature? These are keywords that I would look for within a subreddit's name or description. For example: \"news\", \"politics\", \"discussion\", \"war\", \"vaccines\", \"controversial\", \"conflict\", etc.\n\nKeep the answer short, only including 50 keywords and saving them in a python list as follows [\"key1\",\"key2\",...]. Send the output as text not as code.")

bot_influence_keywords = ast.literal_eval(response.replace("\n", ""))

for i in range(0, len(bot_influence_keywords), 5):
    print(*bot_influence_keywords[i:i+5])

# %%
# Score subreddits based on subscribers and keywords in description
def calculate_bot_influence_score(row):
    score = 0
    
    # Large subscriber base increases potential for bot activity
    if row['Subscribers'] > 10000000:
        score += 5
    elif row['Subscribers'] > 5000000:
        score += 4
    elif row['Subscribers'] > 1000000:
        score += 3
        
    # Check for keywords in description and subreddit name
    description = row['Description'].lower()
    sub_name = row['Name'].lower()
    for keyword in bot_influence_keywords:
        if keyword in description:
            score += 1
        if keyword in sub_name:
            score += 1
            
    return score

subs_df['Bot Score'] = subs_df.apply(calculate_bot_influence_score, axis=1)

# Get top 50 most vulnerable subreddits
top_vulnerable = subs_df.nlargest(50, 'Bot Score')[['Name', 'Subscribers', 'Submission Type', 'Bot Score']].reset_index(drop=True)
top_vulnerable

# %%
# Filter the DataFrame to include only the desired subreddits
subreddits_of_interest = ['worldnews', 'news', 'politics', 'science', 'technology']
top_vulnerable_filtered = top_vulnerable[top_vulnerable['Name'].isin(subreddits_of_interest)].reset_index(drop=True)

top_vulnerable_filtered

# %%
import time

import concurrent.futures

# Function to Fetch Posts and Comments
def fetch_posts_and_comments(subreddit_name, num_posts=50, num_comments=2000):
    """
    Fetches posts and their comments from a subreddit, including comment levels and parent comment ID.

    Args:
        subreddit_name: The name of the subreddit.
        num_posts: The maximum number of posts to fetch.
        num_comments: The maximum number of comments to fetch per post (total, across all levels).

    Returns:
        A list of dictionaries, where each dictionary represents a post and its comments,
        with each comment including its level and parent comment ID.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    try:
        for post in subreddit.hot(limit=num_posts):  # You can change 'hot' to 'new', 'rising', etc.
            post_data = {
                "subreddit": subreddit_name,
                "post_id": post.id,
                "post_title": post.title,
                "post_author": str(post.author),
                "post_score": post.score,
                "post_upvote_ratio": post.upvote_ratio,
                "post_url": post.url,
                "post_selftext": post.selftext,
                "post_created_utc": post.created_utc,
                "comments": []
            }

            def fetch_comments_recursive(comments, level=1, comment_count=0, parent_id=None):
                comment_data = []
                for comment in comments:
                    if comment_count >= num_comments:
                        break  # Stop fetching comments if the limit is reached

                    comment_data.append({
                        "comment_id": comment.id,
                        "comment_author": str(comment.author),
                        "comment_body": comment.body,
                        "comment_score": comment.score,
                        "comment_created_utc": comment.created_utc,
                        "comment_level": level,  # Add the comment level
                        "parent_id": parent_id  # Add the parent comment ID
                    })
                    comment_count += 1

                    # Fetch replies recursively
                    if hasattr(comment, 'replies'):
                        comment.replies.replace_more(limit=0)  # Ensure all 'MoreComments' are resolved
                        replies_data, comment_count = fetch_comments_recursive(comment.replies, level + 1, comment_count, comment.id)
                        comment_data.extend(replies_data)

                return comment_data, comment_count

            post.comments.replace_more(limit=0)  # Ensure all top-level 'MoreComments' are resolved
            comments_data, _ = fetch_comments_recursive(post.comments)
            post_data["comments"] = comments_data

            posts_data.append(post_data)

            # Respect API rate limits
            # time.sleep(1)

    except Exception as e:
        print(f"Error fetching data from r/{subreddit_name}: {e}")

    return posts_data

# Main Data Collection Loop
all_data = []
subreddit_names = top_vulnerable_filtered['Name'].tolist()
num_cores = os.cpu_count()  # Get the number of CPU cores

with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(fetch_posts_and_comments, subreddit_name, num_posts=50, num_comments=50) for subreddit_name in subreddit_names]

    # Wait for all tasks to complete and collect results
    for future in concurrent.futures.as_completed(futures):
        try:
            subreddit_data = future.result()
            all_data.extend(subreddit_data)
        except Exception as e:
            print(f"Error fetching data: {e}")

# Convert to DataFrame
reddit_data_df = pd.DataFrame(all_data)

# Convert lists of comments to a separate DataFrame if desired
comments_data = []
for index, row in reddit_data_df.iterrows():
    for comment in row['comments']:
        comment['post_id'] = row['post_id'] # add the relationship
        comments_data.append(comment)
comments_df = pd.DataFrame(comments_data)
# Expand the comments into its own columns
# reddit_data_df = pd.concat([reddit_data_df.drop(['comments'], axis=1), pd.DataFrame(reddit_data_df['comments'].tolist()).add_prefix('comment_')], axis=1)

# Export to CSV
reddit_data_df.to_csv("reddit_posts_and_comments_03-21-1950.csv", index=False)
comments_df.to_csv("comments_03-21-1950.csv", index=False)


