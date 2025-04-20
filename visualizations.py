import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    def __init__(self):
        self.color_scale = ['#FF4B4B', '#808080', '#00CC96']
    
    def plot_sentiment_distribution(self, df):
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.bar(
            sentiment_counts,
            x='Sentiment',
            y='Count',
            title='Sentiment Distribution',
            color='Sentiment',
            color_discrete_map={
                'positive': '#00CC96',
                'neutral': '#808080',
                'negative': '#FF4B4B'
            }
        ).update_layout(showlegend=False)
        
        return fig
    
    def generate_wordcloud(self, df):
        text = ' '.join(df['review_text'].astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.close()
        return fig