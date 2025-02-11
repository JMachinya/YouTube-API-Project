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
# Access secrets from the [postgresql] table
db_user = st.secrets["postgresql"]["user"]
db_password = st.secrets["postgresql"]["password"]
db_host = st.secrets["postgresql"]["host"]
db_port = st.secrets["postgresql"]["port"]
db_name = st.secrets["postgresql"]["dbname"]

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)


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

daily_video_metrics['day'] = pd.to_datetime(daily_video_metrics['day'])
revenue_metrics['day'] = pd.to_datetime(revenue_metrics['day'])


daily_video_metrics['views_growth'] = daily_video_metrics['views'].pct_change() * 100
daily_video_metrics['views_7_day_avg'] = daily_video_metrics['views'].rolling(window=7).mean()

# ---------------------------
# 3. Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))


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
    "Advanced Metrics & Engagement Analysis",
    "Revenue Prediction Model"  
])

# ----- Overview Tab -----
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
    target_duration = 300  
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
    
    daily_video_metrics['day_name'] = daily_video_metrics['day'].dt.day_name()
    avg_views_by_day = daily_video_metrics.groupby('day_name')['views'].mean()
    best_day = avg_views_by_day.idxmax()
    best_avg = avg_views_by_day.max()
    
    # Determining if weekends or weekdays yield higher average views
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

    # ---- 1. Compute last day views and compare to the previous day ----
    if not filtered_data.empty:
        # Sort by 'day' to ensure last day is at the end
        sorted_data = filtered_data.sort_values('day')
        
        # Identify last day
        last_day = sorted_data['day'].max()
        # Identify the day before the last day (assuming at least 2 rows exist)
        # We'll call it "previous_day"
        
        # Edge case: If there's only 1 day of data, we can't compare
        unique_days = sorted_data['day'].unique()
        if len(unique_days) > 1:
            # The day before the last day is the second to last unique day
            previous_day = unique_days[-2]
            
            # Grab views from the last day
            last_day_views = sorted_data.loc[sorted_data['day'] == last_day, 'views'].sum()
            # Grab views from the previous day
            previous_day_views = sorted_data.loc[sorted_data['day'] == previous_day, 'views'].sum()
            
            # Calculate absolute difference
            diff_views = last_day_views - previous_day_views
            
            # Calculate % difference (avoid dividing by zero)
            if previous_day_views != 0:
                pct_change = (diff_views / previous_day_views) * 100
            else:
                pct_change = 0
        else:
            # If there's only 1 day in the data
            last_day_views = sorted_data['views'].sum()
            diff_views = 0
            pct_change = 0
        
        # Build a string for delta
        # st.metric's delta can be "±X" or "±X%" for a percentage.
        # We'll do a percentage sign:
        delta_str = f"{pct_change:.2f}%"
        
        # Use st.metric to display the last day’s views plus delta
        st.metric(
            label="Last Day Views",
            value=f"{last_day_views:,.0f}",
            delta=delta_str
        )
    else:
        st.write("No data available for metric comparison.")
    
    
    fig1 = px.line(filtered_data, x='day', y='views', title="Daily Views Over Time")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Featured Video")
    st.video("https://www.youtube.com/watch?v=QtfqiBUDVGs&t=647s")
    
    fig2 = px.line(filtered_data, x='day', y='views_7_day_avg', title="7-Day Rolling Average of Views")
    st.plotly_chart(fig2, use_container_width=True)
    
    fig3 = px.line(filtered_data, x='day', y='views_growth', title="Daily Views Growth (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ----- Revenue Analysis Tab -----
with tabs[2]:
    st.header("Revenue Analysis")
    fig4 = px.line(revenue_metrics, x='day', y='estimatedRevenue', title="Estimated Revenue Over Time")
    st.plotly_chart(fig4, use_container_width=True)
    
    # Merging daily_video_metrics and revenue_metrics on 'day'
    merged_rev = pd.merge(daily_video_metrics, revenue_metrics, on='day', how='inner')
    # Using the 'views_x' column from daily_video_metrics after merge
    fig5 = px.scatter(merged_rev, x='views_x', y='estimatedRevenue', 
                     title="Views vs. Estimated Revenue", trendline="ols")
    st.plotly_chart(fig5, use_container_width=True)

with tabs[3]:
    st.header("Geographic Analysis")
    
    def convert_to_iso3(two_letter_code):
        try:
            country = pycountry.countries.get(alpha_2=two_letter_code)
            return country.alpha_3 if country else None
        except:
            return None

    # Converting two-letter country codes to ISO-3 if needed
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
    
    # ----------------------------------------
    # TOP 10 COUNTRIES BY VIEWS
    # ----------------------------------------
    st.subheader("Top 10 Countries by Views")

    top_countries = (
        daily_country_specific_metrics
        .groupby('country', as_index=False)['views']
        .sum()
        .sort_values('views', ascending=False)
        .head(10)
    )

    st.write("Here are the 10 countries with the highest total views:")
    st.dataframe(top_countries)

    fig_top_countries = px.bar(
        top_countries,
        x='views',
        y='country',
        orientation='h',
        color='views',
        color_continuous_scale='Blues',
        title='Top 10 Countries by Total Views',
        labels={'country': 'Country', 'views': 'Total Views'}
    )

    fig_top_countries.update_layout(
        template='plotly_dark',
        xaxis=dict(title='Total Views', showgrid=False),
        yaxis=dict(title='', showgrid=False),
        coloraxis_showscale=False
    )
    fig_top_countries.update_xaxes(tickformat=",")
    st.plotly_chart(fig_top_countries, use_container_width=True)

    # ----------------------------------------
    # DEMOGRAPHICS: AGE GROUP + GENDER
    # ----------------------------------------
    st.subheader("Viewer Demographics (Age & Gender)")

    
    df_demo = pd.read_sql("SELECT * FROM demographic_metrics", engine)
    

    df_demo['viewerPercentage'] = pd.to_numeric(df_demo['viewerPercentage'], errors='coerce')
    
    fig_demo = px.bar(
        df_demo,
        x='ageGroup',
        y='viewerPercentage',
        color='gender',
        barmode='group',
        title="Viewer Percentage by Age Group & Gender"
    )
    st.plotly_chart(fig_demo, use_container_width=True)



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
    
    # Allowing users to select the forecast period in days
    forecast_period = st.number_input("Forecast Period (days)", min_value=1, value=30, step=1)
    
    # Preparing data for Prophet
    df_forecast = daily_video_metrics[['day', 'views']].copy()
    df_forecast.columns = ['ds', 'y']  # Prophet requires the columns 'ds' (date) and 'y' (value)
    
    # Fitting the Prophet model
    model = Prophet()
    model.fit(df_forecast)
    
    # Creating a future dataframe for the selected forecast period
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    
    # Plotting the forecast
    fig8 = model.plot(forecast)
    
    # Adding a vertical line to indicate the end of historical data
    ax = fig8.gca()
    last_date = df_forecast['ds'].max()
    ax.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start')
    ax.legend()
    
    st.pyplot(fig8)
    
    # Displaying the forecasted values for the future period
    st.markdown("### Forecasted Values for the Next {} Days".format(forecast_period))
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))
    
    
    st.markdown("### Forecast Components")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

# ----- Comments Analysis Tab -----

with tabs[6]:
    st.header("Comments Analysis")
    
    st.subheader("Sentiment Analysis on Comments")
    # Importing TextBlob for sentiment analysis (if not already imported)
    from textblob import TextBlob

    # Function to classify sentiment based on polarity
    def get_sentiment_label(comment):
        polarity = TextBlob(comment).sentiment.polarity
        # Adjustting thresholds as needed
        if polarity > 0.1:
            return "POSITIVE"
        elif polarity < 0.0:  # Adjusted threshold to classify as NEGATIVE if polarity is below 0.0
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    # Applying the sentiment function to each comment 
    if "sentiment_label" not in comments.columns:
        comments['sentiment_label'] = comments['comment'].apply(get_sentiment_label)
    
    # Displaying sentiment distribution as a pie chart
    sentiment_counts = comments['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_sentiment = px.pie(sentiment_counts, names='sentiment', values='count',
                           title="Sentiment Distribution of Comments")
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Displaying sentiment counts
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

    # Subsample or filter if you have too many rows (optional)
    # e.g., random sample of 300 rows
    if len(daily_video_metrics) > 300:
        df_sample = daily_video_metrics.sample(n=300, random_state=42)
    else:
        df_sample = daily_video_metrics.copy()

    # Select columns you want in the parallel coordinates plot
    # If you have large numeric ranges (views in thousands, etc.), consider scaling
    features_for_parallel = ['views', 'estimatedMinutesWatched', 'averageViewDuration', 'subscribersGained']

    # Example: min-max scale each selected feature
    df_scaled = df_sample.copy()
    for col in features_for_parallel:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        if col_max != col_min:
            df_scaled[col] = (df_scaled[col] - col_min) / (col_max - col_min)

    st.subheader("Parallel Coordinates Plot (Improved)")

    # We'll color by 'views' after scaling, so lines with higher (scaled) 'views' appear in a different color
    fig9 = px.parallel_coordinates(
        df_scaled,
        dimensions=features_for_parallel,
        color='views',  # now scaled [0..1]
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: Scaled Video Performance Metrics"
    )

    # Optionally reorder the dimension list, or add any custom dimension settings
    # e.g. dimension = dict(range=[0,1], label='Views', values=df_scaled['views']) in a more advanced approach

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
    
# ----- Revenue Prediction Model Tab -----
with tabs[8]:
    st.header("Revenue Prediction Model")
    
    st.markdown("## Data Preparation")
    # Merging daily_video_metrics and revenue_metrics on 'day'
    ml_data = pd.merge(daily_video_metrics, revenue_metrics, on='day', how='inner')
    
    # Droping the duplicate views column from revenue_metrics if it's not needed
    if 'views_y' in ml_data.columns:
        ml_data = ml_data.drop(columns=['views_y'])
    
    # Renaming 'views_x' back to 'views' for clarity
    ml_data = ml_data.rename(columns={'views_x': 'views'})
    
    # Now merging with daily_annotation_metrics (drop its 'views' column if it exists)
    annotation_data = daily_annotation_metrics.copy()
    if 'views' in annotation_data.columns:
        annotation_data = annotation_data.drop(columns=['views'])
    
    annotation_data['day'] = pd.to_datetime(annotation_data['day'])
    ml_data = pd.merge(ml_data, annotation_data, on='day', how='inner')
    
    
    # converting it to one-hot encoded columns
    if 'day_name' in ml_data.columns:
        dummies = pd.get_dummies(ml_data['day_name'], prefix='day')
        ml_data = pd.concat([ml_data, dummies], axis=1)
        ml_data = ml_data.drop(columns=['day_name'])
    
    st.write("### Data Sample")
    st.dataframe(ml_data.tail())
    
    st.markdown("## Model Training and Evaluation")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import seaborn as sns
    
    # Define a list of core features from daily_video_metrics
    selected_features = ['views', 'estimatedMinutesWatched','averageViewDuration', 'cpm','subscribersGained']
    
    
    additional_columns = ['likes']
    for col in additional_columns:
        if col in ml_data.columns:
            selected_features.append(col)
    
    # Optionally, add one-hot encoded day-of-week columns if they exist
    day_columns = [col for col in ml_data.columns if col.startswith('day_')]
    if day_columns:
        selected_features.extend(day_columns)
    
    st.write("### Features Used for Prediction:")
    st.write(selected_features)
    
    # Create the feature set and target variable
    features = ml_data[selected_features]
    target = ml_data['estimatedRevenue']
    
    st.markdown("### Correlation Analysis")

    # Combining features + target into one DataFrame for correlation
    corr_data = ml_data[selected_features + ['estimatedRevenue']].corr()
    
    # Displaying the correlation matrix numerically
    st.write("#### Numerical Correlation Matrix")
    st.write(corr_data)
    
    # Displaying a correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model_rf.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("**Model Performance:**")
    st.write(f"R² Score: {r2:.2f}")
    #st.write(f"Mean Squared Error: {mse:.2f}")
    
    st.markdown("## Feature Importance")
    
    #feature importances
    feature_importances = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.write(feature_importances)
    
    # Plot feature importance using Plotly
    fig_importance = px.bar(feature_importances, x='Feature', y='Importance',
                            title="Feature Importance for Revenue Prediction",
                            labels={'Importance': 'Importance Score'})
    st.plotly_chart(fig_importance, use_container_width=True)