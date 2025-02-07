# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from prophet import Prophet
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pycountry
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="H&J Academy YouTube Dashboard", layout="wide")


# ---------------------------
# 1. Load Data from PostgreSQL
# ---------------------------
engine = create_engine("postgresql://postgres:1999%40Johannes@localhost:5432/youtube_data")

daily_video_metrics = pd.read_sql('SELECT * FROM daily_video_metrics', engine)
comments = pd.read_sql('SELECT * FROM comments', engine)
province_metrics = pd.read_sql('SELECT * FROM province_metrics', engine)
daily_annotation_metrics = pd.read_sql('SELECT * FROM daily_annotation_metrics', engine)
traffic_source_metrics = pd.read_sql('SELECT * FROM traffic_source_metrics', engine)
revenue_metrics = pd.read_sql('SELECT * FROM revenue_metrics', engine)
ad_type_metrics = pd.read_sql('SELECT * FROM ad_type_metrics', engine)
sharing_metrics = pd.read_sql('SELECT * FROM sharing_metrics', engine)
daily_country_specific_metrics = pd.read_sql('SELECT * FROM daily_country_specific_metrics', engine)

# ---------------------------
# 2. Data Preprocessing & Calculations
# ---------------------------
# Convert date columns to datetime
daily_video_metrics['day'] = pd.to_datetime(daily_video_metrics['day'])
revenue_metrics['day'] = pd.to_datetime(revenue_metrics['day'])

# Calculate additional metrics
daily_video_metrics['views_growth'] = daily_video_metrics['views'].pct_change() * 100
daily_video_metrics['views_7_day_avg'] = daily_video_metrics['views'].rolling(window=7).mean()

# ---------------------------
# 3. Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Filter daily video metrics based on the selected date range
filtered_data = daily_video_metrics[
    (daily_video_metrics['day'] >= pd.to_datetime(start_date)) &
    (daily_video_metrics['day'] <= pd.to_datetime(end_date))
]

# ---------------------------
# 4. Dashboard Tabs
# ---------------------------
tabs = st.tabs([
    "Overview", 
    "Video Trends", 
    "Revenue Analysis", 
    "Geographic Analysis", 
    "Clustering", 
    "Forecasting", 
    "Comments Analysis",
    "Advanced Metrics & Engagement Analysis"
])

# ----- Enhanced Overview Tab -----
with tabs[0]:
    st.header("Overview")
    
    # Original KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_views = daily_video_metrics['views'].sum()
    total_minutes = daily_video_metrics['estimatedMinutesWatched'].sum()
    total_subscribers = daily_video_metrics['subscribersGained'].sum()
    avg_view_duration = daily_video_metrics['averageViewDuration'].mean()
    col1.metric("Total Views", f"{total_views:,.0f}")
    col2.metric("Total Minutes Watched", f"{total_minutes:,.0f}")
    col3.metric("Total Subscribers Gained", f"{total_subscribers:,.0f}")
    col4.metric("Avg. View Duration (s)", f"{avg_view_duration:.2f}")
    
    # Additional Metrics based on the filtered period
    if not filtered_data.empty:
        avg_daily_views = filtered_data['views'].mean()
        peak_day_data = filtered_data.loc[filtered_data['views'].idxmax()]
        peak_day = peak_day_data['day'].strftime('%Y-%m-%d')
        peak_views = peak_day_data['views']
        min_day_data = filtered_data.loc[filtered_data['views'].idxmin()]
        min_day = min_day_data['day'].strftime('%Y-%m-%d')
        min_views = min_day_data['views']
    else:
        avg_daily_views = peak_day = peak_views = min_day = min_views = 0

    st.markdown("### Additional Performance Metrics")
    colA, colB, colC = st.columns(3)
    colA.metric("Avg. Daily Views", f"{avg_daily_views:,.0f}")
    colB.metric("Peak Day", f"{peak_day} ({peak_views:,.0f})")
    colC.metric("Lowest Day", f"{min_day} ({min_views:,.0f})")
    
    # Gauge Chart for Average View Duration
    target_duration = 300  # Example target in seconds
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_view_duration,
        delta={'reference': target_duration, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, target_duration * 1.5]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, target_duration], 'color': "lightgray"},
                {'range': [target_duration, target_duration * 1.5], 'color': "gray"}
            ]},
        title={"text": "Avg. View Duration (s)"}
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # --- Upload Recommendation Based on Day-of-Week Analysis ---
    # Calculate day-of-week averages using the historical daily data
    daily_video_metrics['day_name'] = daily_video_metrics['day'].dt.day_name()
    avg_views_by_day = daily_video_metrics.groupby('day_name')['views'].mean()
    best_day = avg_views_by_day.idxmax()
    best_avg = avg_views_by_day.max()
    
    # Determine if weekends or weekdays yield higher average views
    is_weekend = daily_video_metrics['day_name'].isin(['Saturday', 'Sunday'])
    avg_views_weekend = daily_video_metrics[is_weekend]['views'].mean()
    avg_views_weekday = daily_video_metrics[~is_weekend]['views'].mean()
    
    if avg_views_weekend > avg_views_weekday:
        recommendation = (
            f"Data suggests that **weekends** have higher average views "
            f"({avg_views_weekend:.0f}) compared to weekdays ({avg_views_weekday:.0f}). "
            f"Consider uploading on weekends. The best day overall is **{best_day}** "
            f"with an average of **{best_avg:.0f}** views."
        )
    else:
        recommendation = (
            f"Data suggests that **weekdays** have higher average views "
            f"({avg_views_weekday:.0f}) compared to weekends ({avg_views_weekend:.0f}). "
            f"Consider uploading on weekdays. The best day overall is **{best_day}** "
            f"with an average of **{best_avg:.0f}** views."
        )
    
    st.markdown("### Upload Recommendation")
    st.info(recommendation)

# ----- Video Trends Tab -----
with tabs[1]:
    st.header("Video Trends")
    fig1 = px.line(filtered_data, x='day', y='views', title="Daily Views Over Time")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.line(filtered_data, x='day', y='views_7_day_avg', title="7-Day Rolling Average of Views")
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.line(filtered_data, x='day', y='views_growth', title="Daily Views Growth (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ----- Revenue Analysis Tab -----
with tabs[2]:
    st.header("Revenue Analysis")
    fig4 = px.line(revenue_metrics, x='day', y='estimatedRevenue', title="Estimated Revenue Over Time")
    st.plotly_chart(fig4, use_container_width=True)
    
    # Merge daily_video_metrics and revenue_metrics on 'day'
    merged_rev = pd.merge(daily_video_metrics, revenue_metrics, on='day', how='inner')
    # Use the 'views_x' column from daily_video_metrics after merge
    fig5 = px.scatter(merged_rev, x='views_x', y='estimatedRevenue', 
                      title="Views vs. Estimated Revenue", trendline="ols")
    st.plotly_chart(fig5, use_container_width=True)

# ----- Geographic Analysis Tab -----
with tabs[3]:
    st.header("Geographic Analysis")
    
    def convert_to_iso3(two_letter_code):
        try:
            country = pycountry.countries.get(alpha_2=two_letter_code)
            return country.alpha_3 if country else None
        except:
            return None

    # Convert two-letter country codes to ISO-3 if needed
    if daily_country_specific_metrics['country'].str.len().iloc[0] == 2:
        daily_country_specific_metrics['iso_alpha'] = daily_country_specific_metrics['country'].apply(convert_to_iso3)
    else:
        daily_country_specific_metrics['iso_alpha'] = daily_country_specific_metrics['country']
    
    fig6 = px.choropleth(
        daily_country_specific_metrics,
        locations='iso_alpha',
        locationmode='ISO-3',
        color='views',
        color_continuous_scale='Viridis',
        title='Views by Country'
    )
    st.plotly_chart(fig6, use_container_width=True)

# ----- Clustering Tab -----
with tabs[4]:
    st.header("Clustering Analysis")
    features = daily_video_metrics[['views', 'estimatedMinutesWatched', 'subscribersGained']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    daily_video_metrics['cluster'] = kmeans.fit_predict(scaled_features)
    
    fig7 = px.scatter(daily_video_metrics, x='views', y='estimatedMinutesWatched', color='cluster',
                      title="K-Means Clustering of Videos")
    st.plotly_chart(fig7, use_container_width=True)

# ----- Forecasting Tab -----

with tabs[5]:
    st.header("Forecasting Views with Prophet")
    
    # Allow users to select the forecast period in days
    forecast_period = st.number_input("Forecast Period (days)", min_value=1, value=30, step=1)
    
    # Prepare data for Prophet
    df_forecast = daily_video_metrics[['day', 'views']].copy()
    df_forecast.columns = ['ds', 'y']  # Prophet requires the columns 'ds' (date) and 'y' (value)
    
    # Fit the Prophet model
    model = Prophet()
    model.fit(df_forecast)
    
    # Create a future dataframe for the selected forecast period
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    
    # Plot the forecast
    fig8 = model.plot(forecast)
    
    # Add a vertical line to indicate the end of historical data
    ax = fig8.gca()
    last_date = df_forecast['ds'].max()
    ax.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start')
    ax.legend()
    
    st.pyplot(fig8)
    
    # Display the forecasted values for the future period
    st.markdown("### Forecasted Values for the Next {} Days".format(forecast_period))
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))
    
    # Optionally, plot forecast components for more insights
    st.markdown("### Forecast Components")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

# ----- Comments Analysis Tab -----

with tabs[6]:
    st.header("Comments Analysis")
    
    st.subheader("Sentiment Analysis on Comments")
    # Import TextBlob for sentiment analysis (if not already imported)
    from textblob import TextBlob

    # Function to classify sentiment based on polarity
    def get_sentiment_label(comment):
        polarity = TextBlob(comment).sentiment.polarity
        # Adjust thresholds as needed
        if polarity > 0.1:
            return "POSITIVE"
        elif polarity < 0.0:  # Adjusted threshold to classify as NEGATIVE if polarity is below 0.0
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    # Apply the sentiment function to each comment (if not already done)
    if "sentiment_label" not in comments.columns:
        comments['sentiment_label'] = comments['comment'].apply(get_sentiment_label)
    
    # Display sentiment distribution as a pie chart
    sentiment_counts = comments['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_sentiment = px.pie(sentiment_counts, names='sentiment', values='count',
                           title="Sentiment Distribution of Comments")
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # (Optional) Display the sentiment counts as text for debugging
    st.write("Sentiment Counts:", comments['sentiment_label'].value_counts())
    
    # ---------------------------
    # Word Cloud for Positive Comments
    st.subheader("Word Cloud for Positive Comments")
    positive_comments = " ".join(comments[comments['sentiment_label'] == "POSITIVE"]['comment'].tolist())
    if positive_comments:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
        buf_pos = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title("Positive Comments")
        plt.savefig(buf_pos, format="png")
        st.image(buf_pos, caption="Positive Comments", use_container_width=True)
        plt.clf()
    else:
        st.write("No positive comments to display.")
    
    # ---------------------------
    # Word Cloud for Negative Comments
    st.subheader("Word Cloud for Negative Comments")
    negative_comments = " ".join(comments[comments['sentiment_label'] == "NEGATIVE"]['comment'].tolist())
    if negative_comments:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
        buf_neg = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title("Negative Comments")
        plt.savefig(buf_neg, format="png")
        st.image(buf_neg, caption="Negative Comments", use_container_width=True)
        plt.clf()
    else:
        st.write("No negative comments to display.")


# ----- Advanced Metrics & Engagement Analysis Tab -----
with tabs[7]:
    st.header("Advanced Metrics & Engagement Analysis")
    
    st.subheader("Parallel Coordinates Plot")
    fig9 = px.parallel_coordinates(
        daily_video_metrics,
        dimensions=['views', 'estimatedMinutesWatched', 'averageViewDuration', 'subscribersGained'],
        color='views',
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: Video Performance Metrics"
    )
    st.plotly_chart(fig9, use_container_width=True)
    
    st.subheader("Box Plot: Average View Duration by Cluster")
    fig10 = px.box(daily_video_metrics, x='cluster', y='averageViewDuration',
                   title="Average View Duration by Cluster",
                   labels={'cluster': 'Cluster', 'averageViewDuration': 'Avg. View Duration'})
    st.plotly_chart(fig10, use_container_width=True)
    
    st.subheader("Sharing Metrics by Service")
    sharing_agg = sharing_metrics.groupby('sharingService')['shares'].sum().reset_index()
    fig11 = px.bar(sharing_agg, x='sharingService', y='shares', 
                   title="Total Shares by Sharing Service",
                   labels={'sharingService': 'Sharing Service', 'shares': 'Total Shares'})
    st.plotly_chart(fig11, use_container_width=True)
    
    st.subheader("Ad Type Metrics Treemap")
    fig12 = px.treemap(ad_type_metrics, path=['adType'], 
                       values='adImpressions', color='cpm',
                       color_continuous_scale='RdBu',
                       title="Ad Type Metrics: Ad Impressions and CPM")
    st.plotly_chart(fig12, use_container_width=True)
    
    st.subheader("Traffic Source Distribution")
    traffic_agg = traffic_source_metrics.groupby('insightTrafficSourceType')['views'].sum().reset_index()
    fig13 = px.pie(traffic_agg, names='insightTrafficSourceType', values='views',
                   title="Traffic Source Distribution (by Views)")
    st.plotly_chart(fig13, use_container_width=True)
