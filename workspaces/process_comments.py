# %%
import pandas as pd

reddit_data_df = pd.read_csv("reddit_posts_and_comments_03-21-1950.csv")

reddit_data_df

# %%
comments_df = pd.read_csv("comments_03-21-1950.csv")
comments_df = comments_df.dropna(subset=['comment_author'])  # Drop rows with missing author names (removed posts)

comments_df

# %%
# Merge reddit_data_df with comments_df
comments_df = comments_df.merge(reddit_data_df[['post_id', 'post_author', 'subreddit', 'post_created_utc']], on='post_id', how='left')

comments_df

# %%
# Convert UTC timestamps to datetime objects and add a new column for the time taken to comment on a post
comments_df['comment_created_utc'] = pd.to_datetime(comments_df['comment_created_utc'], unit='s')
comments_df['post_created_utc'] = pd.to_datetime(comments_df['post_created_utc'], unit='s')

comments_df['post_to_comment_time'] = comments_df['comment_created_utc'] - comments_df['post_created_utc']

comments_df

# %%
import numpy as np

# Calculate comment length
comments_df['comment_length'] = comments_df['comment_body'].str.len()

# Calculate the ratio of comment length to post-to-comment time.  Convert the timedelta to seconds.
comments_df['comment_speed_from_post'] = comments_df['comment_length'] / comments_df['post_to_comment_time'].dt.total_seconds()
# Calculate the time difference between a comment and its parent comment.

# First, we need to merge comments_df with itself to get the creation time of the parent comment.
comments_df = comments_df.merge(comments_df[['comment_id', 'comment_created_utc']], left_on='parent_id', right_on='comment_id', suffixes=('', '_parent'), how='left')

# Calculate the time difference.  If parent_id is NaN, then the time difference will be NaN
comments_df['comment_to_parent_time'] = comments_df['comment_created_utc'] - comments_df['comment_created_utc_parent']

# Calculate comment speed from parent. Convert the timedelta to seconds.
comments_df['comment_speed_from_parent'] = comments_df['comment_length'] / comments_df['comment_to_parent_time'].dt.total_seconds()
# Identify comments made by the original poster (OP)
comments_df['is_op'] = np.where(comments_df['post_author'] == comments_df['comment_author'], True, False)

comments_df

# %%
import praw
import os
from datetime import datetime
import pandas as pd

reddit = praw.Reddit(
    client_id=os.getenv("PRAW_CLIENT_ID"),
    client_secret=os.getenv("PRAW_CLIENT_SECRET"),
    user_agent=os.getenv("PRAW_USER_AGENT"),
    username=os.getenv("PRAW_USERNAME"),
    password=os.getenv("PRAW_PASSWORD")
)

unique_users = comments_df['comment_author'].unique().tolist()

user_metrics_cache = {}  # Dictionary to store user metrics

def get_user_metrics(author_name):
    """
    Fetches account age and karma for a given Reddit username.
    Uses a cache to avoid repeated API calls for the same user.

    Args:
        author_name (str): The Reddit username.

    Returns:
        tuple: A tuple containing account age in days and karma score.
               Returns (None, None) if there's an error.
    """
    if author_name in user_metrics_cache:
        return user_metrics_cache[author_name]

    try:
        user = reddit.redditor(author_name)

        # Account Age
        account_creation_time = datetime.fromtimestamp(user.created_utc)
        age_in_days = (datetime.now() - account_creation_time).days

        # Karma
        karma = user.comment_karma + user.link_karma

        user_metrics_cache[author_name] = (age_in_days, karma)  # Store in cache
        return age_in_days, karma

    except praw.exceptions.APIException as e:
        print(f"Error fetching metrics for {author_name}: {e}")
        return None, None
    except Exception as e:
        print(f"Error fetching metrics for {author_name}: {e}")
        return None, None

user_data = {}
for i, user in enumerate(unique_users):
    age, karma = get_user_metrics(user)
    user_data[user] = {'account_age_days': age, 'karma': karma}

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(unique_users)} users")

# Map the user data back to the comments_df
comments_df['account_age_days'] = comments_df['comment_author'].map(lambda x: user_data[x]['account_age_days'])
comments_df['karma'] = comments_df['comment_author'].map(lambda x: user_data[x]['karma'])

comments_df

# %%
# Identify users with missing account age or karma
missing_users = comments_df[comments_df['account_age_days'].isnull()]['comment_author'].tolist()

print(f"Number of rows with missing data: {len(missing_users)}")

# Retry fetching metrics for missing users
if missing_users:
    print("Retrying fetching metrics for missing users...")
    for user in missing_users:
        age, karma = get_user_metrics(user)
        
        # Update the user_data dictionary and DataFrame
        user_data[user] = {'account_age_days': age, 'karma': karma}
        comments_df.loc[comments_df['comment_author'] == user, 'account_age_days'] = age
        comments_df.loc[comments_df['comment_author'] == user, 'karma'] = karma
        
        if age is not None and karma is not None:
            print(f"Fetched data for user: {user} - Age: {age}, Karma: {karma}")

    print("Finished retrying fetching metrics for missing users.")
else:
    print("No users with missing data.")

# Verify that there are no more missing values
print(f"Number of rows with missing data after retry: {comments_df['account_age_days'].isnull().sum()}")

comments_df

# %%
# Drop rows where 'account_age_days' or 'karma' is NaN
comments_df = comments_df.dropna(subset=['account_age_days', 'karma'])

comments_df.to_csv("processed_comments.csv", index=False)

comments_df

# %%
import time

unique_authors = comments_df['comment_author'].unique()

all_comments = []

for i, author in enumerate(unique_authors):
    retries = 3  # Number of retries
    for attempt in range(retries):
        try:
            user = reddit.redditor(author)
            comments = user.comments.new(limit=10)
            for comment in comments:
                comment_data = {
                    'author': author,
                    'comment_id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'subreddit': comment.subreddit.display_name,
                    'created_utc': comment.created_utc,
                    'permalink': comment.permalink
                }
                all_comments.append(comment_data)
            break  # If successful, break out of the retry loop
        except Exception as e:
            print(f"Could not retrieve comments for {author} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(10)  # Wait for 10 seconds before retrying
            else:
                print(f"Failed to retrieve comments for {author} after {retries} attempts.")
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(unique_authors)} authors")

recent_user_comments = pd.DataFrame(all_comments)

recent_user_comments['created_utc'] = pd.to_datetime(recent_user_comments['created_utc'], unit='s')

recent_user_comments.to_csv("recent_user_comments.csv", index=False)

recent_user_comments


