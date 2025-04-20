import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
import plotly.express as px
from visualizations import Visualizer
from nltk_setup import setup_nltk
import re
from collections import Counter
from langdetect import detect
from deep_translator import GoogleTranslator

# Initialize NLTK
setup_nltk()

# Initialize Visualizer
visualizer = Visualizer()

# Function to detect language and translate if needed
def process_text(text):
    try:
        # Detect language
        lang = detect(text)
        
        # If not English, translate to English for analysis
        if lang != 'en':
            translator = GoogleTranslator(source=lang, target='en')
            translated_text = translator.translate(text)
            return {
                'original_text': text,
                'translated_text': translated_text,
                'language': lang,
                'is_translated': True
            }
        else:
            return {
                'original_text': text,
                'translated_text': text,
                'language': 'en',
                'is_translated': False
            }
    except Exception as e:
        st.error(f"Error in language detection or translation: {str(e)}")
        return {
            'original_text': text,
            'translated_text': text,
            'language': 'en',
            'is_translated': False
        }

# Add emotion detection function
def detect_emotions(text):
    # Define emotion keywords dictionary
    emotion_keywords = {
        'joy': ['happy', 'delighted', 'pleased', 'glad', 'thrilled', 'excited', 'love', 'enjoy', 'amazing', 'fantastic', 'wonderful', 'impressed', 'exceeds'],
        'satisfaction': ['satisfied', 'content', 'fulfilled', 'pleased', 'quality', 'perfect', 'excellent', 'great', 'good', 'nice'],
        'trust': ['reliable', 'dependable', 'trustworthy', 'secure', 'confident', 'faith', 'believe', 'trust'],
        'anticipation': ['expect', 'anticipate', 'look forward', 'hope', 'await', 'future'],
        'surprise': ['surprised', 'amazed', 'astonished', 'unexpected', 'wow', 'incredible', 'surprisingly'],
        'anger': ['angry', 'annoyed', 'irritated', 'frustrated', 'mad', 'hate', 'terrible', 'awful'],
        'sadness': ['sad', 'unhappy', 'disappointed', 'upset', 'regret', 'depressed', 'miss', 'unfortunate'],
        'fear': ['afraid', 'scared', 'worried', 'concerned', 'anxious', 'fear', 'dread', 'terrified'],
        'disgust': ['disgusted', 'dislike', 'hate', 'horrible', 'gross', 'unpleasant', 'repulsive']
    }
    
    try:
        # Normalize text
        text_lower = text.lower()
        
        # Count emotion occurrences
        emotion_counts = Counter()
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + keyword + r'\b', text_lower))
                if count > 0:
                    emotion_counts[emotion] += count
        
        # Calculate emotion intensity based on TextBlob sentiment
        blob = TextBlob(text)
        sentiment_intensity = abs(blob.sentiment.polarity)
        
        # Adjust emotion scores based on sentiment intensity
        emotion_scores = {}
        total_count = sum(emotion_counts.values())
        
        # Handle the case when no emotions are detected
        if total_count == 0:
            # Default to sentiment-based emotion
            if blob.sentiment.polarity > 0.1:
                emotion_scores['joy'] = 0.7
                emotion_scores['satisfaction'] = 0.3
            elif blob.sentiment.polarity < -0.1:
                emotion_scores['sadness'] = 0.7
                emotion_scores['disappointment'] = 0.3
            else:
                emotion_scores['neutral'] = 1.0
        else:
            # Calculate scores normally when emotions are detected
            for emotion, count in emotion_counts.items():
                base_score = count / total_count
                # Amplify scores based on sentiment intensity
                emotion_scores[emotion] = base_score * (1 + sentiment_intensity)
        
        return emotion_scores
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        # Return a default emotion in case of error
        return {'neutral': 1.0}

def extract_aspects(text):
    try:
        # Simple sentence splitting
        sentences = text.split('.')
        aspects = {}
        aspect_sentences = {}  # Store sentences containing each aspect
        
        # Define common nouns for aspect extraction
        common_nouns = [
            # Product characteristics
            'price', 'quality', 'value', 'cost', 'worth', 'money',
            # Features
            'feature', 'design', 'style', 'look', 'appearance', 'color', 'size', 'weight',
            'performance', 'speed', 'efficiency', 'power', 'battery', 'screen', 'display',
            # Experience
            'service', 'experience', 'customer', 'support', 'help', 'assistance',
            'delivery', 'shipping', 'packaging', 'arrival', 'condition',
            # Product types
            'product', 'item', 'device', 'model', 'version', 'brand', 'company',
            # Technical aspects
            'functionality', 'reliability', 'durability', 'usability', 'interface',
            'software', 'hardware', 'app', 'application', 'update', 'installation',
            # Emotional aspects
            'satisfaction', 'disappointment', 'expectation', 'recommendation'
        ]
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Convert to lowercase and tokenize simply
            words = [word.strip().lower() for word in sentence.split() if word.strip()]
            
            # Extract aspects without POS tagging
            for word in words:
                # Check if word is a potential aspect (either in common nouns or longer than 3 chars)
                if word in common_nouns or (len(word) > 3 and word.isalpha()):
                    blob = TextBlob(sentence)
                    sentiment = blob.sentiment.polarity
                    
                    if word not in aspects:
                        aspects[word] = []
                        aspect_sentences[word] = []
                    
                    aspects[word].append(sentiment)
                    aspect_sentences[word].append(sentence.strip())
        
        # Filter out non-aspect words (like "the", "and", etc.)
        stopwords = ['the', 'and', 'but', 'for', 'with', 'this', 'that', 'very', 'just', 'from', 'have', 'has', 'had', 'was', 'were']
        filtered_aspects = {}
        filtered_sentences = {}
        
        for k, v in aspects.items():
            if k.strip() and k not in stopwords and len(v) > 0:
                avg_sentiment = sum(v)/len(v)
                filtered_aspects[k] = avg_sentiment
                filtered_sentences[k] = aspect_sentences[k]
        
        # Calculate influence score (combination of sentiment strength and frequency)
        influence_scores = {}
        for aspect, sentiment in filtered_aspects.items():
            # Influence = abs(sentiment) * frequency * sentence_length_factor
            frequency = len(aspect_sentences[aspect])
            
            # Skip aspects with zero frequency (shouldn't happen, but just in case)
            if frequency == 0:
                continue
                
            avg_sentence_length = sum(len(s.split()) for s in aspect_sentences[aspect]) / frequency
            sentence_length_factor = min(1.0, avg_sentence_length / 10)  # Normalize, cap at 1.0
            
            # Ensure we don't get zero influence
            influence = abs(sentiment) * frequency * (1 + sentence_length_factor)
            if influence > 0:
                influence_scores[aspect] = influence
            else:
                influence_scores[aspect] = 0.001  # Small non-zero value
        
        # Return aspects with their sentiment and influence data
        result = {
            'sentiment': filtered_aspects,
            'influence': influence_scores,
            'sentences': filtered_sentences
        }
        
        return result
    except Exception as e:
        st.error(f"Error in aspect extraction: {str(e)}")
        return {'sentiment': {}, 'influence': {}, 'sentences': {}}

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", polarity
    else:
        return "neutral", polarity

# App title and description
st.title("Advanced Sentiment Analysis System")
st.write("Analyze overall sentiment and aspect-based sentiment from your text in any language")

# Text input
text_input = st.text_area("Enter your review here (in any language):", height=150)

if st.button("Analyze"):
    if text_input:
        # Process text (detect language and translate if needed)
        processed_text = process_text(text_input)
        
        # Display language information
        if processed_text['is_translated']:
            language_name = {
                'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian',
                'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian', 'ja': 'Japanese',
                'zh-cn': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'ko': 'Korean'
            }.get(processed_text['language'], processed_text['language'])
            
            st.info(f"Detected language: {language_name}. The review has been translated for analysis.")
            with st.expander("View translation"):
                st.write(processed_text['translated_text'])
        
        # Use translated text for analysis
        analysis_text = processed_text['translated_text']
        
        # Overall sentiment analysis
        sentiment, score = analyze_sentiment(analysis_text)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Overall Sentiment", "Aspect Analysis", "Emotion Analysis"])
        
        # Rest of your tab code remains the same, but using analysis_text instead of text_input
        with tab1:
            st.subheader("Overall Sentiment Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Sentiment", sentiment.capitalize())
            with col2:
                st.metric("Score", f"{score:.2f}")
            
            # Fix the sentiment visualization
            sentiment_data = pd.DataFrame({
                'Category': ['Positive', 'Neutral', 'Negative'],
                'Score': [
                    max(0, score) if sentiment == 'positive' else 0,
                    0.5 if sentiment == 'neutral' else 0,
                    abs(min(0, score)) if sentiment == 'negative' else 0
                ]
            })
            
            fig = px.bar(
                sentiment_data,
                x='Category',
                y='Score',
                color='Category',
                title='Sentiment Analysis',
                color_discrete_map={
                    'Positive': '#00CC96',
                    'Neutral': '#808080',
                    'Negative': '#FF4B4B'
                }
            ).update_layout(showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Aspect-Based Analysis")
            # Use the translated text for aspect analysis instead of original text
            aspect_data = extract_aspects(analysis_text)
            
            if aspect_data['sentiment']:
                # Create aspect sentiment visualization with only significant aspects
                aspects_df = pd.DataFrame({
                    'Aspect': list(aspect_data['sentiment'].keys()),
                    'Sentiment': list(aspect_data['sentiment'].values()),
                    'Influence': list(aspect_data['influence'].values())
                })
                
                # Sort by influence score (most influential first)
                aspects_df = aspects_df.sort_values('Influence', ascending=False)
                
                # Keep only the top influential aspects (top 50% or at least 3, whichever is greater)
                num_aspects = max(3, int(len(aspects_df) * 0.5))
                significant_aspects = aspects_df.head(num_aspects)
                
                # Create color scale based on sentiment for significant aspects only
                fig_aspects = px.bar(
                    significant_aspects,
                    x='Aspect',
                    y='Sentiment',
                    color='Sentiment',
                    color_continuous_scale=['red', 'gray', 'green'],
                    hover_data=['Influence'],
                    title='Key Aspects Influencing the Review',
                    range_y=[-1, 1]  # Fix the y-axis range to show full sentiment scale
                ).update_layout(
                    xaxis_title="Product Aspects",
                    yaxis_title="Sentiment Score (-1 to 1)",
                    height=400,  # Set a fixed height
                    margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins
                    coloraxis_colorbar=dict(
                        title="Sentiment",
                        tickvals=[-1, 0, 1],
                        ticktext=["Negative", "Neutral", "Positive"]
                    )
                )
                
                # Ensure the graph is displayed with proper width
                st.plotly_chart(fig_aspects, use_container_width=True)
                
                # Display key influencing aspects
                st.subheader("Key Influencing Aspects")
                
                for i, (_, row) in enumerate(significant_aspects.iterrows()):
                    aspect = row['Aspect']
                    sentiment = row['Sentiment']
                    influence = row['Influence']
                    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                    
                    # Calculate a star rating to visually indicate influence (1-5 stars)
                    max_influence = significant_aspects['Influence'].max()
                    # Prevent division by zero
                    if max_influence > 0:
                        stars = min(5, max(1, int((influence / max_influence) * 5)))
                    else:
                        stars = 1  # Default to 1 star if max_influence is zero
                    
                    star_display = "★" * stars + "☆" * (5 - stars)
                    
                    st.write(f"**{i+1}. {aspect.capitalize()}**: {sentiment_label} ({sentiment:.2f}) {star_display}")
                    
                    # Show example sentences for this aspect (up to 2)
                    sentences = aspect_data['sentences'][aspect][:2]
                    for sentence in sentences:
                        st.write(f"  - *\"{sentence}\"*")
                    
                    st.write("")
                
                # Add a note about filtered aspects
                if len(aspects_df) > len(significant_aspects):
                    st.info(f"Note: {len(aspects_df) - len(significant_aspects)} less influential aspects were filtered out.")
            else:
                st.info("No specific aspects were identified in the text.")
        
        # Add the emotion analysis tab
        with tab3:
            st.subheader("Emotion Analysis")
            try:
                # Use the translated text for emotion analysis
                emotions = detect_emotions(analysis_text)
                
                if emotions and len(emotions) > 0:
                    # Create emotion visualization
                    emotion_df = pd.DataFrame({
                        'Emotion': list(emotions.keys()),
                        'Score': list(emotions.values())
                    })
                    
                    # Sort by score
                    emotion_df = emotion_df.sort_values('Score', ascending=False)
                    
                    # Create color map for emotions
                    emotion_colors = {
                        'joy': '#FFD700',
                        'satisfaction': '#00CC96', 
                        'trust': '#4682B4',
                        'anticipation': '#FFA500',
                        'surprise': '#9370DB',
                        'anger': '#FF4B4B',
                        'sadness': '#1E90FF',
                        'fear': '#800080',
                        'disgust': '#A0522D',
                        'neutral': '#808080',
                        'disappointment': '#CD5C5C'
                    }
                    
                    # Create color list for the chart
                    colors = [emotion_colors.get(emotion, '#808080') for emotion in emotion_df['Emotion']]
                    
                    # Create bar chart
                    fig_emotions = px.bar(
                        emotion_df,
                        x='Emotion',
                        y='Score',
                        title='Emotions Detected in Review',
                        color='Emotion',
                        color_discrete_map={emotion: color for emotion, color in zip(emotion_df['Emotion'], colors)}
                    )
                    
                    st.plotly_chart(fig_emotions)
                    
                    # Display primary emotion
                    if len(emotion_df) > 0:
                        primary_emotion = emotion_df.iloc[0]['Emotion']
                        primary_score = emotion_df.iloc[0]['Score']
                        
                        st.subheader("Primary Emotion")
                        st.write(f"The dominant emotion in this review is **{primary_emotion.capitalize()}** with a score of {primary_score:.2f}")
                        
                        # Provide emotion-based insights
                        if primary_emotion == 'joy':
                            st.write("This review expresses strong happiness and delight with the product.")
                        elif primary_emotion == 'satisfaction':
                            st.write("The reviewer appears highly satisfied with their purchase.")
                        elif primary_emotion == 'trust':
                            st.write("The review indicates trust and confidence in the product.")
                        elif primary_emotion == 'surprise':
                            st.write("The reviewer was pleasantly surprised by the product's features or performance.")
                        elif primary_emotion in ['anger', 'sadness', 'fear', 'disgust', 'disappointment']:
                            st.write("The review expresses negative emotions, suggesting disappointment with the product.")
                        elif primary_emotion == 'neutral':
                            st.write("The review appears to be factual and neutral in tone.")
                        
                        # Show emotion distribution
                        st.subheader("Emotion Distribution")
                        st.write("The review contains a mix of the following emotions:")
                        
                        for i, (_, row) in enumerate(emotion_df.iterrows()):
                            emotion = row['Emotion']
                            score = row['Score']
                            st.write(f"- **{emotion.capitalize()}**: {score:.2f}")
                else:
                    st.info("No specific emotions were detected in the text.")
            except Exception as e:
                st.error(f"Error analyzing emotions: {str(e)}")
                st.info("Unable to perform emotion analysis on this text.")
    else:
        st.warning("Please enter some text to analyze.")

# Add information about the analysis
st.sidebar.title("About")
st.sidebar.info("""
This advanced sentiment analyzer provides:
- Overall sentiment analysis
- Aspect-based sentiment analysis
- Sentiment scores (-1 to 1)
- Visual representation of results
- Detailed aspect breakdown
- Multilingual support with automatic translation
- Emotion detection
""")