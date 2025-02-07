from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import datetime

app = Flask(__name__)

# YouTube API Credentials
API_KEY = "AIzaSyCW70WzNcZYDOz2-y8yJa7dAJgDke9kCqM"  # Replace with your actual API key
OAUTH_CREDENTIALS = "client_secret_29238299626-95koqi0a53ageu7d25tuikquappn3l22.apps.googleusercontent.com.json"  # Replace with your OAuth JSON file

# Initialize YouTube APIs
youtube = build("youtube", "v3", developerKey=API_KEY)

SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
    "https://www.googleapis.com/auth/youtube.readonly"
]
flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CREDENTIALS, SCOPES)
credentials = flow.run_local_server(port=0)
youtube_analytics = build("youtubeAnalytics", "v2", credentials=credentials)


# Function to fetch analytics data
def fetch_analytics_data(start_date, end_date, metrics, dimensions, sort, filters=None):
    request = youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start_date,
        endDate=end_date,
        metrics=metrics,
        dimensions=dimensions,
        sort=sort,
        filters=filters
    )
    response = request.execute()
    data = [dict(zip([col["name"] for col in response["columnHeaders"]], row))
            for row in response.get("rows", [])]
    return pd.DataFrame(data)


# Flask route to fetch analytics data
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    params = request.json
    start_date = params.get("start_date", "2022-01-01")
    end_date = params.get("end_date", datetime.datetime.now().strftime("%Y-%m-%d"))
    metrics = params["metrics"]
    dimensions = params["dimensions"]
    sort = params.get("sort", "day")
    filters = params.get("filters", None)

    try:
        data = fetch_analytics_data(start_date, end_date, metrics, dimensions, sort, filters)
        return data.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
import requests

url = "http://127.0.0.1:5000/fetch_data"
payload = {
    "metrics": "views,estimatedMinutesWatched",
    "dimensions": "day",
    "sort": "day"
}
response = requests.post(url, json=payload)
print(response.json())

@app.route('/fetch_data', methods=['GET', 'POST'])
def fetch_data():
    if request.method == 'GET':
        return jsonify({"message": "Use a POST request with a JSON payload to fetch data."})
    
    params = request.json
    start_date = params.get("start_date", "2022-01-01")
    end_date = params.get("end_date", datetime.datetime.now().strftime("%Y-%m-%d"))
    metrics = params["metrics"]
    dimensions = params["dimensions"]
    sort = params.get("sort", "day")
    filters = params.get("filters", None)

    data = fetch_analytics_data(start_date, end_date, metrics, dimensions, sort, filters)
    return data.to_json(orient="records")
