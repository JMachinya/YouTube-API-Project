from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import datetime

API_KEY = "AIzaSyCW70WzNcZYDOz2-y8yJa7dAJgDke9kCqM"
OAUTH_CREDENTIALS = "client_secret_29238299626-95koqi0a53ageu7d25tuikquappn3l22.apps.googleusercontent.com.json"

# YouTube Data API v3 setup
youtube = build("youtube", "v3", developerKey=API_KEY)
print("YouTube Data API connected successfully!")

SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
    "https://www.googleapis.com/auth/youtube.readonly"
]

flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CREDENTIALS, SCOPES)
credentials = flow.run_local_server(port=0)
youtube_analytics = build("youtubeAnalytics", "v2", credentials=credentials)
print("YouTube Analytics API connected successfully!")

# Function to fetch analytics data with optional filters
def fetch_analytics_data(start_date, end_date, metrics, dimensions, sort, filters=None):
    data = []
    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics=metrics,
        dimensions=dimensions,
        sort=sort,
        filters=filters  # Add filters if provided
    )
    response = request.execute()

    # Extract column headers as names
    column_headers = [header["name"] for header in response["columnHeaders"]]

    for row in response.get("rows", []):
        entry = dict(zip(column_headers, row))
        data.append(entry)
    
    return pd.DataFrame(data)

# Function to fetch video comments
def fetch_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText"
    )
    
    response = request.execute()
    
    # Collect comments from the response
    while response:
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        
        # Check if there are more pages of comments
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
            response = request.execute()
        else:
            break
    
    return comments

# Function to get video IDs from the channel
def get_video_ids(channel_id):
    video_ids = []
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=50,  # Adjust the number as needed
        type="video"
    )
    
    response = request.execute()
    
    for item in response["items"]:
        video_ids.append(item["id"]["videoId"])
    
    return video_ids

# Main Execution
channel_id = "UCiosg1akiDnp4fq513TOmmw"
start_date = "2019-01-01"  # Adjust this based on your channel creation date
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Define the queries based on the documentation
queries = [
    {
        "name": "daily_video_metrics",
        "metrics": "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained",
        "dimensions": "day",
        "sort": "day"
    },
    {
        "name": "daily_annotation_metrics",
        "metrics": "views,likes,annotationClickThroughRate,annotationCloseRate,annotationImpressions",
        "dimensions": "day",
        "sort": "day"
    },
    {
        "name": "daily_country_specific_metrics",
        "metrics": "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained",
        "dimensions": "country",
        "sort": "-estimatedMinutesWatched"
    },
    {
        "name": "traffic_source_metrics",
        "metrics": "views,estimatedMinutesWatched",
        "dimensions": "day,insightTrafficSourceType",
        "sort": "day"
    },
    {
        "name": "revenue_metrics",
        "metrics": "views,estimatedRevenue,estimatedAdRevenue,estimatedRedPartnerRevenue,grossRevenue,adImpressions,cpm,playbackBasedCpm,monetizedPlaybacks",
        "dimensions": "day",
        "sort": "day"
    },
    {
        "name": "ad_type_metrics",
        "metrics": "grossRevenue,adImpressions,cpm",
        "dimensions": "adType",
        "sort": "-adType"
    },
    {
        "name": "sharing_metrics",
        "metrics": "shares",
        "dimensions": "sharingService",
        "sort": "-shares"
    },
    {
        "name": "province_metrics",
        "metrics": "views,estimatedMinutesWatched,averageViewDuration",
        "dimensions": "province",
        "filters": "country==US", 
        "sort": "province"
    }
]

# Fetch and save all datasets
combined_data = {}
for query in queries:
    print(f"Fetching data for {query['name']}...")
    df = fetch_analytics_data(
        start_date=start_date,
        end_date=end_date,
        metrics=query["metrics"],
        dimensions=query["dimensions"],
        sort=query["sort"],
        filters=query.get("filters")  # Pass filters if present
    )
    combined_data[query["name"]] = df
    df.to_csv(f"{query['name']}.csv", index=False)
    print(f"{query['name']} saved to {query['name']}.csv")

# Fetch video IDs from the channel
video_ids = get_video_ids(channel_id)

# Create a list to store all comments with video IDs
all_comments = []

# Fetch and save comments for each video
for video_id in video_ids:
    print(f"Fetching comments for video {video_id}...")
    comments = fetch_video_comments(video_id)
    for comment in comments:
        all_comments.append({"video_id": video_id, "comment": comment})

# Convert all comments into a DataFrame and save to a single CSV
comments_df = pd.DataFrame(all_comments)
comments_df.to_csv("all_comments.csv", index=False)

print("All datasets and comments fetched and saved successfully.")
