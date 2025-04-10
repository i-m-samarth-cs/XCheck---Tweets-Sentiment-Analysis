import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pytesseract
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Analyzer", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .title-container {
        background-color: #1DA1F2;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .stTextInput, .stFileUploader {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .results-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("""
<div class="title-container">
    <h1>Tweet Sentiment Analyzer</h1>
    <h3>Analyze the sentiment of tweets as positive or negative</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar for app navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
    ["About", "Tweet Analysis", "Model Performance", "Dataset Exploration"])

# Function for text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

# Function to extract text from image
# Function to extract text from image
def extract_text_from_image(image):
    try:
        # Specify Tesseract executable path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        # Try to load data from Kaggle
        try:
            import kagglehub
            path = kagglehub.dataset_download("ruchi798/data-science-tweets")
            data_path = os.path.join(path, "tweets.csv")
            df = pd.read_csv(data_path)
        except:
            # Fallback to using a sample data if Kaggle download fails
            st.warning("Could not load Kaggle dataset. Using sample data instead.")
            # Create sample data
            sample_tweets = [
                "I love data science! It's amazing #datascience #love",
                "This machine learning algorithm is frustrating me so much",
                "Just finished an amazing course on deep learning",
                "The results from my model are terrible, I hate this",
                "Excited about the new Python features! #python #programming",
                "My data analysis project failed again, worst day ever",
                "Happy to share my new data visualization dashboard",
                "Can't stand this data cleaning process, it's too tedious",
                "Successfully deployed my first ML model today!",
                "The conference on AI was disappointing, waste of time"
            ]
            sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
            df = pd.DataFrame({'text': sample_tweets, 'sentiment': sentiments})
        
        # Check if sentiment column exists, if not create one (binary classification)
        if 'sentiment' not in df.columns:
            # This is a placeholder. In reality, you would need to label your data
            # Here we're using a simple rule for demonstration
            df['sentiment'] = df['text'].apply(lambda x: 1 if 'love' in x.lower() or 'good' in x.lower() or 'great' in x.lower() else 0)
        
        # Ensure sentiment is binary (0 or 1)
        if df['sentiment'].nunique() > 2:
            # Convert to binary (assuming positive is > 0.5 on a scale)
            df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x > 0.5 else 0)
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['text', 'sentiment', 'processed_text'])

# Function to train the model
@st.cache_resource
def train_model(df):
    try:
        # Split data
        X = df['processed_text']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return model, vectorizer, accuracy, report, X_test, y_test, y_pred
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, 0, {}, None, None, None

# Function to predict sentiment
def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    return prediction, probability

# Load data and train model
df = load_data()
model_data = train_model(df)
model, vectorizer, accuracy, report, X_test, y_test, y_pred = model_data

# About page
if app_mode == "About":
    st.markdown("""
    ## About This App
    
    This application analyzes the sentiment of tweets as positive or negative. You can input a tweet as text or upload an image containing a tweet.
    
    ### Features:
    - Text-based tweet analysis
    - Image-based tweet analysis (extracts text from images)
    - Model performance metrics
    - Dataset exploration and visualization
    
    ### How to use:
    1. Select "Tweet Analysis" from the sidebar to analyze your tweets
    2. Input text directly or upload an image containing a tweet
    3. View the sentiment prediction and confidence score
    4. Check "Model Performance" to see how accurate our model is
    5. Explore the dataset used for training in "Dataset Exploration"
    
    ### Technology Stack:
    - Streamlit for the web interface
    - NLTK for text preprocessing
    - Scikit-learn for machine learning
    - Tesseract OCR for image text extraction
    - Plotly and Matplotlib for visualizations
    """)
    
    st.markdown("""
    <div class="results-container">
        <h3>Dataset Overview</h3>
        <p>The model is trained on a dataset containing tweets related to data science.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sample tweets
    st.markdown("### Sample Tweets from Dataset")
    if not df.empty:
        sample_df = df.sample(min(5, len(df)))
        for i, row in sample_df.iterrows():
            sentiment = "Positive" if row['sentiment'] == 1 else "Negative"
            st.markdown(f"""
            <div style="background-color: {'#e8f4f8' if row['sentiment'] == 1 else '#f8e8e8'}; 
                        padding: 10px; border-radius: 5px; margin: 5px 0;">
                <p><b>Tweet:</b> {row['text']}</p>
                <p><b>Sentiment:</b> {sentiment}</p>
            </div>
            """, unsafe_allow_html=True)

# Tweet Analysis page
elif app_mode == "Tweet Analysis":
    st.markdown("<h2>Analyze Tweet Sentiment</h2>", unsafe_allow_html=True)
    
    # Create columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Option 1: Enter Tweet Text</h3>", unsafe_allow_html=True)
        tweet_text = st.text_area("Enter the tweet text here", height=150)
        analyze_text_button = st.button("Analyze Text", key="analyze_text")
    
    with col2:
        st.markdown("<h3>Option 2: Upload Tweet Image</h3>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an image of a tweet", type=["jpg", "jpeg", "png"])
        analyze_image_button = st.button("Analyze Image", key="analyze_image")
    
    # Analyze text input
    if analyze_text_button and tweet_text:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing tweet sentiment..."):
            prediction, probability = predict_sentiment(tweet_text, model, vectorizer)
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = probability[1] if prediction == 1 else probability[0]
            
            # Display results
            st.markdown(f"<p><b>Tweet:</b> {tweet_text}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><b>Sentiment:</b> {sentiment}</p>", unsafe_allow_html=True)
            
            # Create gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1DA1F2"},
                    'steps': [
                        {'range': [0, 33], 'color': "#FF4B4B"},
                        {'range': [33, 66], 'color': "#FFA64B"},
                        {'range': [66, 100], 'color': "#4BFF4B"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analyze image input
    if analyze_image_button and uploaded_image is not None:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
        
        with st.spinner("Processing image and analyzing sentiment..."):
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Tweet Image", use_column_width=True)
            
            # Extract text from image
            extracted_text = extract_text_from_image(image)
            
            if extracted_text:
                st.markdown(f"<p><b>Extracted Text:</b> {extracted_text}</p>", unsafe_allow_html=True)
                
                # Predict sentiment
                prediction, probability = predict_sentiment(extracted_text, model, vectorizer)
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = probability[1] if prediction == 1 else probability[0]
                
                # Display results
                st.markdown(f"<p><b>Sentiment:</b> {sentiment}</p>", unsafe_allow_html=True)
                
                # Create gauge chart for confidence
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1DA1F2"},
                        'steps': [
                            {'range': [0, 33], 'color': "#FF4B4B"},
                            {'range': [33, 66], 'color': "#FFA64B"},
                            {'range': [66, 100], 'color': "#4BFF4B"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not extract text from the image. Please try a clearer image or enter text directly.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Model Performance page
elif app_mode == "Model Performance":
    st.markdown("<h2>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
    
    # Display accuracy and classification report
    st.markdown(f"<h3>Model Accuracy: {accuracy:.2%}</h3>", unsafe_allow_html=True)
    
    # Convert classification report to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    if 'support' in report_df.columns:
        report_df['support'] = report_df['support'].astype(int)
    
    # Remove rows that aren't needed
    if 'accuracy' in report_df.index:
        report_df = report_df.drop('accuracy')
    
    # Display classification metrics
    st.markdown("<h3>Classification Metrics</h3>", unsafe_allow_html=True)
    st.table(report_df.style.format({
        'precision': '{:.2%}',
        'recall': '{:.2%}',
        'f1-score': '{:.2%}'
    }))
    
    # Create confusion matrix
    if y_test is not None and y_pred is not None:
        st.markdown("<h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        
        # Calculate confusion matrix values
        tn = sum((y_test == 0) & (y_pred == 0))
        fp = sum((y_test == 0) & (y_pred == 1))
        fn = sum((y_test == 1) & (y_pred == 0))
        tp = sum((y_test == 1) & (y_pred == 1))
        
        # Create confusion matrix figure
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        
        # Prediction distribution
        st.markdown("<h3>Prediction Distribution</h3>", unsafe_allow_html=True)
        fig = px.histogram(
            x=[int(pred) for pred in y_pred], 
            color=[int(pred) for pred in y_pred],
            labels={'x': 'Predicted Sentiment', 'count': 'Count'},
            title='Distribution of Predictions',
            color_discrete_map={0: '#FF4B4B', 1: '#4BFF4B'},
            category_orders={"x": [0, 1]}
        )
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Negative', 'Positive']))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Dataset Exploration page
elif app_mode == "Dataset Exploration":
    st.markdown("<h2>Dataset Exploration</h2>", unsafe_allow_html=True)
    
    if not df.empty:
        st.markdown("<div class='results-container'>", unsafe_allow_html=True)
        
        # Dataset overview
        st.markdown("<h3>Dataset Overview</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Total Tweets:</b> {len(df)}</p>", unsafe_allow_html=True)
        
        # Calculate sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        sentiment_counts['Sentiment'] = sentiment_counts['Sentiment'].map({0: 'Negative', 1: 'Positive'})
        
        # Display sentiment distribution
        st.markdown("<h3>Sentiment Distribution</h3>", unsafe_allow_html=True)
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': '#4BFF4B', 'Negative': '#FF4B4B'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word frequency analysis
        st.markdown("<h3>Word Frequency Analysis</h3>", unsafe_allow_html=True)
        
        # Create word frequency for positive and negative tweets
        positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
        negative_text = ' '.join(df[df['sentiment'] == 0]['processed_text'])
        
        from collections import Counter
        import re
        
        def get_top_words(text, n=10):
            words = re.findall(r'\b\w+\b', text.lower())
            return Counter(words).most_common(n)
        
        # Get top words
        top_positive = get_top_words(positive_text)
        top_negative = get_top_words(negative_text)
        
        # Create DataFrames
        pos_df = pd.DataFrame(top_positive, columns=['Word', 'Frequency'])
        neg_df = pd.DataFrame(top_negative, columns=['Word', 'Frequency'])
        
        # Create tabs for positive and negative words
        tab1, tab2 = st.tabs(["Positive Words", "Negative Words"])
        
        with tab1:
            if not pos_df.empty:
                fig = px.bar(
                    pos_df, 
                    x='Word', 
                    y='Frequency',
                    title='Top Words in Positive Tweets',
                    color_discrete_sequence=['#4BFF4B']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No positive tweets available for word frequency analysis.")
        
        with tab2:
            if not neg_df.empty:
                fig = px.bar(
                    neg_df, 
                    x='Word', 
                    y='Frequency',
                    title='Top Words in Negative Tweets',
                    color_discrete_sequence=['#FF4B4B']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No negative tweets available for word frequency analysis.")
        
        # Display random sample of tweets
        st.markdown("<h3>Sample Tweets</h3>", unsafe_allow_html=True)
        sample_size = min(10, len(df))
        sample_df = df.sample(sample_size)
        
        for i, row in sample_df.iterrows():
            sentiment = "Positive" if row['sentiment'] == 1 else "Negative"
            st.markdown(f"""
            <div style="background-color: {'#e8f4f8' if row['sentiment'] == 1 else '#f8e8e8'}; 
                        padding: 10px; border-radius: 5px; margin: 5px 0;">
                <p><b>Tweet:</b> {row['text']}</p>
                <p><b>Sentiment:</b> {sentiment}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No data available for exploration. Please check the dataset.")