from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import datetime

# Replace with your API key and OAuth JSON file
API_KEY = "AIzaSyCW70WzNcZYDOz2-y8yJa7dAJgDke9kCqM"
OAUTH_CREDENTIALS = "client_secret_29238299626-95koqi0a53ageu7d25tuikquappn3l22.apps.googleusercontent.com.json"

# YouTube Data API v3 setup
youtube = build("youtube", "v3", developerKey=API_KEY)
print("YouTube Data API connected successfully!")

# YouTube Analytics API setup
SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
    "https://www.googleapis.com/auth/youtube.readonly"
]
flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CREDENTIALS, SCOPES)
credentials = flow.run_local_server(port=0)
youtube_analytics = build("youtubeAnalytics", "v2", credentials=credentials)
print("YouTube Analytics API connected successfully!")

# Fetch channel creation date
def get_channel_creation_date(channel_id):
    request = youtube.channels().list(
        part="snippet",
        id=channel_id
    )
    response = request.execute()
    creation_date = response["items"][0]["snippet"]["publishedAt"]
    return creation_date.split("T")[0]

# Function to fetch video metadata
def fetch_video_metadata(channel_id):
    video_data = []
    request = youtube.search().list(
        part="id,snippet",
        channelId=channel_id,
        maxResults=50,
        type="video"
    )
    response = request.execute()

    for item in response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        publish_date = item['snippet']['publishedAt']

        stats_request = youtube.videos().list(
            part="statistics,contentDetails",
            id=video_id
        )
        stats_response = stats_request.execute()

        for video in stats_response['items']:
            video_data.append({
                "video_id": video_id,
                "title": video_title,
                "publish_date": publish_date,
                "views": video['statistics'].get("viewCount", 0),
                "likes": video['statistics'].get("likeCount", 0),
                "comments": video['statistics'].get("commentCount", 0),
                "duration": video['contentDetails']['duration']
            })
    return pd.DataFrame(video_data)

# Function to fetch audience demographics
def fetch_audience_insights(start_date, end_date):
    audience_data = []

    # Query: Viewer Percentage by Age and Gender
    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="viewerPercentage",
        dimensions="ageGroup,gender",
        sort="-viewerPercentage"
    )
    response = request.execute()
    for row in response.get("rows", []):
        audience_data.append({
            "age_group": row[0],
            "gender": row[1],
            "viewer_percentage": row[2]
        })

    return pd.DataFrame(audience_data)

# Function to fetch revenue insights
def fetch_revenue_insights(start_date, end_date):
    revenue_data = []

    # API Query for Revenue Insights
    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="estimatedRevenue,playbackBasedCpm",
        dimensions="country",
        sort="-estimatedRevenue"
    )
    response = request.execute()

    for row in response.get("rows", []):
        revenue_data.append({
            "country": row[0],
            "estimated_revenue": float(row[1]),
            "playback_based_cpm": float(row[2])
        })

    return pd.DataFrame(revenue_data)


# Function to fetch traffic sources with valid dimensions and metrics
def fetch_traffic_sources(start_date, end_date):
    traffic_data = []

    # Use the valid dimension `insightTrafficSourceDetail`
    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="views,estimatedRevenue",
        dimensions="country",
        sort="-views"
    )
    response = request.execute()

    for row in response.get("rows", []):
        traffic_data.append({
            "traffic_source_detail": row[0],
            "views": int(row[1]),
            "estimated_revenue": float(row[2]) if len(row) > 2 else None
        })

    return pd.DataFrame(traffic_data)

# Function to fetch device types
def fetch_device_insights(start_date, end_date):
    device_data = []

    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics="views,averageViewDuration",
        dimensions="deviceType",
        sort="-views"
    )
    response = request.execute()

    for row in response.get("rows", []):
        device_data.append({
            "device_type": row[0],
            "views": int(row[1]),
            "average_view_duration": float(row[2])
        })

    return pd.DataFrame(device_data)

# Main Execution
channel_id = "UCiosg1akiDnp4fq513TOmmw"
start_date = get_channel_creation_date(channel_id)
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

video_metadata_df = fetch_video_metadata(channel_id)
audience_df = fetch_audience_insights(start_date, end_date)
revenue_df = fetch_revenue_insights(start_date, end_date)
traffic_sources_df = fetch_traffic_sources(start_date, end_date)
device_insights_df = fetch_device_insights(start_date, end_date)

# Combine datasets
combined_data = {
    "Video Metadata": video_metadata_df,
    "Audience Insights": audience_df,
    "Revenue Details": revenue_df,
    "Traffic Sources": traffic_sources_df,
    "Device Insights": device_insights_df
}

# Save to CSV
for name, df in combined_data.items():
    df.to_csv(f"{name.replace(' ', '_').lower()}.csv", index=False)
    print(f"{name} saved to {name.replace(' ', '_').lower()}.csv")
