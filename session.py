import pandas as pd
import json
import time

# Load the data from the CSV file
start_time = time.time()
data = pd.read_csv('./twcs/twcs.csv')
print(f"Data loaded in {time.time() - start_time:.2f} seconds")


# Function to build a conversation session from a starting tweet_id
def build_conversation(data, tweet_id):
    session = []
    queue = [tweet_id]

    while queue:
        current_id = queue.pop(0)
        tweet = data.loc[data['tweet_id'] == current_id]

        if tweet.empty:
            continue

        # Convert to dictionary
        tweet_dict = tweet.to_dict(orient='records')[0]

        # Determine if this is a customer or support tweet
        if tweet_dict['inbound']:
            role = 'customer'
        else:
            role = 'support'

        # Add tweet to session
        session.append({
            'author_id': tweet_dict['author_id'],
            'created_at': tweet_dict['created_at'],
            'text': tweet_dict['text'],
            'role': role
        })

        # Add responses to the queue
        if pd.notna(tweet_dict['response_tweet_id']):
            response_ids = str(tweet_dict['response_tweet_id']).split(',')
            response_ids = [int(rid) for rid in response_ids if rid.isdigit()]
            queue.extend(response_ids)

    # Sort the session to maintain the order based on 'created_at'
    session = sorted(session, key=lambda x: x['created_at'])
    return session


# Specify the company author_id to filter sessions
company_author_id = 'sprintcare'  # Example: 'sprintcare'

# Find all unique starting points for conversations where 'in_response_to_tweet_id' is NaN
start_time = time.time()
starting_tweets = data[pd.isna(data['in_response_to_tweet_id'])]
print(f"Identified starting tweets in {time.time() - start_time:.2f} seconds")

# Initialize list to store filtered sessions
filtered_sessions = []

# Process each starting tweet to build full conversation sessions
for index, tweet in starting_tweets.iterrows():
    if index % 100 == 0:
        print(f"Processing tweet {index}/{len(starting_tweets)}...")

    # Check if any response to this starting tweet is from the specified company
    response_tweet_ids = tweet['response_tweet_id']
    if pd.notna(response_tweet_ids):
        response_ids = [int(rid) for rid in str(response_tweet_ids).split(',') if rid.isdigit()]
        # Check if any of these responses is from the specified company
        if data.loc[data['tweet_id'].isin(response_ids) & (data['author_id'] == company_author_id)].empty:
            continue

    # If valid session, build the conversation starting from this tweet
    session_start_time = time.time()
    session = build_conversation(data, tweet['tweet_id'])
    print(f"Built session in {time.time() - session_start_time:.2f} seconds")

    if session:
        filtered_sessions.append(session)

# Output the structured sessions as JSON
output_filename = f'customer_support_sessions_{company_author_id}.json'
start_time = time.time()
with open(output_filename, 'w') as json_file:
    json.dump(filtered_sessions, json_file, indent=4)
print(f"JSON file '{output_filename}' created successfully in {time.time() - start_time:.2f} seconds")


