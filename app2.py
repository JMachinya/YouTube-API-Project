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

# Set Streamlit page configuration
st.set_page_config(page_title="Advanced YouTube Dashboard", layout="wide")

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

# ----- Overview Tab -----
with tabs[0]:
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_views = daily_video_metrics['views'].sum()
    total_minutes = daily_video_metrics['estimatedMinutesWatched'].sum()
    total_subscribers = daily_video_metrics['subscribersGained'].sum()
    avg_view_duration = daily_video_metrics['averageViewDuration'].mean()
    col1.metric("Total Views", f"{total_views:,.0f}")
    col2.metric("Total Minutes Watched", f"{total_minutes:,.0f}")
    col3.metric("Total Subscribers Gained", f"{total_subscribers:,.0f}")
    col4.metric("Avg. View Duration", f"{avg_view_duration:.2f}")

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
    df_forecast = daily_video_metrics[['day', 'views']].copy()
    df_forecast.columns = ['ds', 'y']  # Prophet requires 'ds' (date) and 'y' (value)
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    fig8 = model.plot(forecast)
    st.pyplot(fig8)

# ----- Comments Analysis Tab -----
with tabs[6]:
    st.header("Comments Analysis")
    st.subheader("Word Cloud of All Comments")
    all_comments = " ".join(comments['comment'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)
    
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of All Comments")
    plt.savefig(buf, format="png")
    st.image(buf)
    plt.clf()

# ----- Advanced Metrics & Engagement Analysis Tab -----
with tabs[7]:
    st.header("Advanced Metrics & Engagement Analysis")
    
    st.subheader("Parallel Coordinates Plot")
    # Parallel Coordinates Plot to explore multiple video performance metrics
    fig9 = px.parallel_coordinates(
        daily_video_metrics,
        dimensions=['views', 'estimatedMinutesWatched', 'averageViewDuration', 'subscribersGained'],
        color='views',
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: Video Performance Metrics"
    )
    st.plotly_chart(fig9, use_container_width=True)
    
    st.subheader("Box Plot: Average View Duration by Cluster")
    # Box Plot to compare average view duration across clusters
    fig10 = px.box(daily_video_metrics, x='cluster', y='averageViewDuration',
                   title="Average View Duration by Cluster",
                   labels={'cluster': 'Cluster', 'averageViewDuration': 'Avg. View Duration'})
    st.plotly_chart(fig10, use_container_width=True)
    
    st.subheader("Sharing Metrics by Service")
    # Bar Chart for total shares aggregated by sharing service
    sharing_agg = sharing_metrics.groupby('sharingService')['shares'].sum().reset_index()
    fig11 = px.bar(sharing_agg, x='sharingService', y='shares', 
                   title="Total Shares by Sharing Service",
                   labels={'sharingService': 'Sharing Service', 'shares': 'Total Shares'})
    st.plotly_chart(fig11, use_container_width=True)
    
    st.subheader("Ad Type Metrics Treemap")
    # Treemap for ad type metrics: size by ad impressions, color by CPM
    fig12 = px.treemap(ad_type_metrics, path=['adType'], 
                       values='adImpressions', color='cpm',
                       color_continuous_scale='RdBu',
                       title="Ad Type Metrics: Ad Impressions and CPM")
    st.plotly_chart(fig12, use_container_width=True)
    
    st.subheader("Traffic Source Distribution")
    # Pie Chart for traffic source metrics (aggregated by insightTrafficSourceType)
    traffic_agg = traffic_source_metrics.groupby('insightTrafficSourceType')['views'].sum().reset_index()
    fig13 = px.pie(traffic_agg, names='insightTrafficSourceType', values='views',
                   title="Traffic Source Distribution (by Views)")
    st.plotly_chart(fig13, use_container_width=True)
