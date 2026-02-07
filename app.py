"""
Streamlit App for Royal Enfield Customer Conversation Analysis
Features:
- Interactive data exploration
- Custom data upload
- Automated analysis
- Downloadable reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import io
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Chatbot Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_and_validate_data(uploaded_file, is_sample=False):
    """Load and validate uploaded data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Validate required columns
        required_columns = [
            'TOPIC', 'TOPIC_DESCRIPTION', 'USER_QUERY', 'RESOLUTION',
            'RESOLUTION_STATUS', 'RESOLUTION_STATUS_REASONING',
            'USER_SENTIMENT', 'USER_SENTIMENT_REASONING', 'SESSION_ID'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Required columns: " + ", ".join(required_columns))
            return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def has_enhanced_features(df):
    """Check if dataframe has enhanced features (intent_accuracy, spam_check)"""
    return 'intent_accuracy' in df.columns and 'spam_check' in df.columns


def add_text_features(df):
    """Add text-based features to dataframe"""
    df['query_length'] = df['USER_QUERY'].fillna('').str.len()
    df['query_word_count'] = df['USER_QUERY'].fillna('').str.split().str.len()
    df['resolution_length'] = df['RESOLUTION'].fillna('').str.len()
    df['resolution_word_count'] = df['RESOLUTION'].fillna('').str.split().str.len()
    return df


def extract_keywords(df, top_n=20):
    """Extract top keywords from queries"""
    all_queries = ' '.join(df['USER_QUERY'].fillna('').str.lower())
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'can', 'i', 'you', 'my',
        'what', 'how', 'do', 'does', 'about', 'royal', 'enfield'
    }
    
    words = re.findall(r'\b[a-z]+\b', all_queries)
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_topic_distribution(df):
    """Plot topic distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_counts = df['TOPIC'].value_counts().head(15)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(topic_counts)))
    topic_counts.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_title('Top 15 Topics Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Conversations', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_resolution_status(df):
    """Plot resolution status distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    resolution_counts = df['RESOLUTION_STATUS'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    ax1.pie(resolution_counts.values, labels=resolution_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    ax1.set_title('Resolution Status Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    resolution_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Resolution Status Counts', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Status', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sentiment_analysis(df):
    """Plot sentiment analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sentiment distribution
    sentiment_counts = df['USER_SENTIMENT'].value_counts()
    colors_sent = ['#90EE90', '#FFD700', '#FF6B6B']
    
    sentiment_counts.plot(kind='bar', ax=ax1, color=colors_sent[:len(sentiment_counts)])
    ax1.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Sentiment by resolution status
    sentiment_resolution = pd.crosstab(df['RESOLUTION_STATUS'], df['USER_SENTIMENT'])
    sentiment_resolution.plot(kind='bar', stacked=True, ax=ax2, color=colors_sent)
    ax2.set_title('Sentiment by Resolution Status', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Resolution Status', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_text_analysis(df):
    """Plot text length analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Query word count distribution
    ax1.hist(df['query_word_count'].dropna(), bins=20, color='teal', edgecolor='black', alpha=0.7)
    ax1.axvline(df['query_word_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["query_word_count"].mean():.1f}')
    ax1.set_title('Query Word Count Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Resolution word count distribution
    ax2.hist(df['resolution_word_count'].dropna(), bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax2.axvline(df['resolution_word_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["resolution_word_count"].mean():.1f}')
    ax2.set_title('Resolution Word Count Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Average query length by topic (top 10)
    avg_query_by_topic = df.groupby('TOPIC')['query_word_count'].mean().sort_values(ascending=False).head(10)
    avg_query_by_topic.plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('Avg Query Length by Topic (Top 10)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Average Word Count')
    ax3.grid(axis='x', alpha=0.3)
    
    # Resolution length by status
    avg_resolution_by_status = df.groupby('RESOLUTION_STATUS')['resolution_word_count'].mean()
    avg_resolution_by_status.plot(kind='bar', ax=ax4, color='coral')
    ax4.set_title('Avg Resolution Length by Status', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Resolution Status')
    ax4.set_ylabel('Average Word Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_heatmap_analysis(df):
    """Plot heatmap for topic vs resolution"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    topic_resolution = pd.crosstab(df['TOPIC'], df['RESOLUTION_STATUS'])
    
    # Get top 20 topics by frequency
    top_topics = df['TOPIC'].value_counts().head(20).index
    topic_resolution_filtered = topic_resolution.loc[top_topics]
    
    sns.heatmap(topic_resolution_filtered, annot=True, fmt='d', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5)
    ax.set_title('Topic vs Resolution Status Heatmap (Top 20 Topics)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Resolution Status', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_intent_accuracy_analysis(df):
    """Plot intent accuracy analysis for enhanced sample data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Intent Accuracy Distribution
    intent_counts = df['intent_accuracy'].value_counts()
    colors = ['#4CAF50', '#FF5252']  # Green for accurate, Red for not_accurate
    
    ax1.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    ax1.set_title('Intent Accuracy Distribution', fontsize=14, fontweight='bold')
    
    # 2. Intent Accuracy Bar Chart
    intent_counts.plot(kind='bar', ax=ax2, color=colors)
    ax2.set_title('Intent Accuracy Count', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Accuracy Status', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Intent Accuracy by Resolution Status
    intent_resolution = pd.crosstab(df['RESOLUTION_STATUS'], df['intent_accuracy'])
    intent_resolution.plot(kind='bar', ax=ax3, color=colors, width=0.7)
    ax3.set_title('Intent Accuracy by Resolution Status', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Resolution Status', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Intent Accuracy')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Intent Accuracy by Topic (Top 10)
    topic_accuracy = pd.crosstab(df['TOPIC'], df['intent_accuracy'])
    top_topics = df['TOPIC'].value_counts().head(10).index
    topic_accuracy_filtered = topic_accuracy.loc[top_topics]
    
    topic_accuracy_filtered.plot(kind='barh', stacked=True, ax=ax4, color=colors)
    ax4.set_title('Intent Accuracy by Topic (Top 10)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Count', fontsize=12)
    ax4.set_ylabel('Topic', fontsize=12)
    ax4.legend(title='Intent Accuracy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spam_check_analysis(df):
    """Plot spam check analysis for enhanced sample data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Spam Check Distribution
    spam_counts = df['spam_check'].value_counts()
    colors_spam = ['#2196F3', '#FF9800']  # Blue for not_spam, Orange for spam
    
    ax1.pie(spam_counts.values, labels=spam_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors_spam, textprops={'fontsize': 11})
    ax1.set_title('Spam Check Distribution', fontsize=14, fontweight='bold')
    
    # 2. Spam Check Bar Chart
    spam_counts.plot(kind='bar', ax=ax2, color=colors_spam)
    ax2.set_title('Spam Check Count', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Spam Status', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Spam Check by Sentiment
    spam_sentiment = pd.crosstab(df['USER_SENTIMENT'], df['spam_check'])
    spam_sentiment.plot(kind='bar', ax=ax3, color=colors_spam, width=0.7)
    ax3.set_title('Spam Check by Sentiment', fontsize=14, fontweight='bold')
    ax3.set_xlabel('User Sentiment', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Spam Status')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Combined: Intent Accuracy vs Spam Check
    combined = pd.crosstab(df['intent_accuracy'], df['spam_check'])
    combined.plot(kind='bar', ax=ax4, color=colors_spam, width=0.7)
    ax4.set_title('Intent Accuracy vs Spam Check', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Intent Accuracy', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Spam Status')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header"> Data Analysis </div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dataset")
    st.sidebar.markdown("---")
    
    # File upload option
    upload_option = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Data", "Upload Custom Data"]
    )
    
    df = None
    
    if upload_option == "Upload Custom Data":
        st.sidebar.markdown("### 📤 Upload Your Data")
        st.sidebar.info("""
        **Required Columns:**
        - TOPIC
        - TOPIC_DESCRIPTION
        - USER_QUERY
        - RESOLUTION
        - RESOLUTION_STATUS
        - RESOLUTION_STATUS_REASONING
        - USER_SENTIMENT
        - USER_SENTIMENT_REASONING
        - SESSION_ID

        """)
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file with the required columns"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading and validating data..."):
                df = load_and_validate_data(uploaded_file)
                
            if df is not None:
                st.sidebar.success(f"✅ Data loaded successfully! ({len(df)} rows)")
        else:
            st.info("👈 Please upload a file from the sidebar to begin analysis")
            
    else:  # Use sample data
        try:
            df = pd.read_csv('data.csv')
            st.sidebar.success(f"✅ Sample data loaded! ({len(df)} rows)")
        except:
            st.error("Sample data not found. Please upload your own data.")
            return
    
    # Main content
    if df is not None:
        # Add text features
        df = add_text_features(df)
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Overview", 
            "📊 Topic Analysis", 
            "✅ Resolution Analysis",
            "😊 Sentiment Analysis",
            "📝 Text Analysis",
            "📥 Export Reports"
        ])
        
        # ========================================================================
        # TAB 1: OVERVIEW
        # ========================================================================
        with tab1:
            st.header("Data Overview")
            
            # Check if enhanced features are available
            has_enhanced = has_enhanced_features(df)
            
            # Key Metrics
            if has_enhanced:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
            else:
                col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Conversations", len(df))
            
            with col2:
                unique_topics = df['TOPIC'].nunique()
                st.metric("Unique Topics", unique_topics)
            
            with col3:
                resolved_pct = (df['RESOLUTION_STATUS'] == 'resolved').sum() / len(df) * 100
                st.metric("Resolution Rate", f"{resolved_pct:.1f}%")
            
            with col4:
                unique_sessions = df['SESSION_ID'].nunique()
                st.metric("Unique Sessions", unique_sessions)
            
            with col5:
                negative_pct = (df['USER_SENTIMENT'] == 'negative').sum() / len(df) * 100
                st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
            
            # Enhanced metrics if available
            if has_enhanced:
                with col6:
                    accurate_pct = (df['intent_accuracy'] == 'accurate').sum() / len(df) * 100
                    st.metric("Intent Accuracy", f"{accurate_pct:.1f}%")
            
            st.markdown("---")
            
            # Data Preview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Data Preview")
                # Exclude suggestion column from preview
                display_df = df.drop(columns=['suggestion'], errors='ignore')
                st.dataframe(display_df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📊 Data Statistics")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                
                missing = df.isnull().sum().sum()
                st.write(f"**Missing Values:** {missing}")
                
                duplicates = df.duplicated().sum()
                st.write(f"**Duplicates:** {duplicates}")
                
                st.subheader(" Column Types")
                type_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values
                })
                st.dataframe(type_df, use_container_width=True)
            
            # Missing values details
            if missing > 0:
                st.markdown("---")
                st.subheader("⚠️ Missing Values Details")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
            
            # Enhanced Features Analysis (Intent Accuracy & Spam Check)
            if has_enhanced:
                st.markdown("---")
                st.header("Quality Analysis")
                #st.info("📌 These metrics are available in the sample data to demonstrate bot performance quality.")
                
                # Intent Accuracy & Spam Check Metrics Side by Side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Intent Accuracy Analysis")
                    
                    # Intent accuracy metrics
                    intent_counts = df['intent_accuracy'].value_counts()
                    accurate_count = intent_counts.get('accurate', 0)
                    not_accurate_count = intent_counts.get('not_accurate', 0)
                    total = len(df)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("✅ Accurate", f"{accurate_count} ({accurate_count/total*100:.1f}%)")
                    with metric_col2:
                        st.metric("❌ Not Accurate", f"{not_accurate_count} ({not_accurate_count/total*100:.1f}%)")
                    
                    st.markdown("---")
                    
                    # Intent accuracy visualization
                    fig = plot_intent_accuracy_analysis(df)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("🛡️ Spam Check Analysis")
                    
                    # Spam check metrics
                    spam_counts = df['spam_check'].value_counts()
                    not_spam_count = spam_counts.get('not_spam', 0)
                    spam_count = spam_counts.get('spam', 0)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("✅ Not Spam", f"{not_spam_count} ({not_spam_count/total*100:.1f}%)")
                    with metric_col2:
                        st.metric("⚠️ Spam", f"{spam_count} ({spam_count/total*100:.1f}%)")
                    
                    st.markdown("---")
                    
                    # Spam check visualization
                    fig = plot_spam_check_analysis(df)
                    st.pyplot(fig)
                
                # Combined insights
                st.markdown("---")
                st.subheader("Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy rate by resolution status
                    accuracy_by_resolution = df.groupby('RESOLUTION_STATUS')['intent_accuracy'].apply(
                        lambda x: (x == 'accurate').sum() / len(x) * 100 if len(x) > 0 else 0
                    ).round(2)
                    
                    st.markdown("**📊 Accuracy by Resolution:**")
                    for status, acc in accuracy_by_resolution.items():
                        emoji = "🟢" if acc > 80 else "🟡" if acc > 50 else "🔴"
                        st.write(f"{emoji} {status}: {acc:.1f}%")
                
                with col2:
                    # Spam rate by sentiment
                    spam_by_sentiment = df.groupby('USER_SENTIMENT')['spam_check'].apply(
                        lambda x: (x == 'spam').sum() / len(x) * 100 if len(x) > 0 else 0
                    ).round(2)
                    
                    st.markdown("**🛡️ Spam by Sentiment:**")
                    for sentiment, spam_pct in spam_by_sentiment.items():
                        emoji = "🔴" if spam_pct > 20 else "🟡" if spam_pct > 5 else "🟢"
                        st.write(f"{emoji} {sentiment}: {spam_pct:.1f}%")

        
        # ========================================================================
        # TAB 2: TOPIC ANALYSIS
        # ========================================================================
        with tab2:
            st.header("Topic Analysis")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Topic Distribution")
                fig = plot_topic_distribution(df)
                st.pyplot(fig)
            
            with col2:
                st.subheader("📈 Topic Statistics")
                topic_counts = df['TOPIC'].value_counts()
                
                st.metric("Total Unique Topics", df['TOPIC'].nunique())
                st.write(f"**Most Common:** {topic_counts.index[0]}")
                st.write(f"**Occurrences:** {topic_counts.iloc[0]}")
                
                st.markdown("---")
                st.subheader("Top 10 Topics")
                topic_df = pd.DataFrame({
                    'Topic': topic_counts.head(10).index,
                    'Count': topic_counts.head(10).values,
                    'Percentage': (topic_counts.head(10).values / len(df) * 100).round(2)
                })
                st.dataframe(topic_df, use_container_width=True)
            
            st.markdown("---")
            
            # Topic details table
            st.subheader("📋 Detailed Topic Breakdown")
            
            topic_summary = df.groupby('TOPIC').agg({
                'SESSION_ID': 'count',
                'RESOLUTION_STATUS': lambda x: (x == 'resolved').sum() / len(x) * 100,
                'USER_SENTIMENT': lambda x: (x == 'positive').sum() / len(x) * 100,
                'query_word_count': 'mean'
            }).round(2)
            
            topic_summary.columns = ['Count', 'Resolution Rate %', 'Positive Sentiment %', 'Avg Query Words']
            topic_summary = topic_summary.sort_values('Count', ascending=False)
            
            st.dataframe(topic_summary, use_container_width=True)
        
        # ========================================================================
        # TAB 3: RESOLUTION ANALYSIS
        # ========================================================================
        with tab3:
            st.header("Resolution Analysis")
            
            # Resolution metrics
            col1, col2, col3, col4 = st.columns(4)
            
            resolution_counts = df['RESOLUTION_STATUS'].value_counts()
            
            for i, (status, count) in enumerate(resolution_counts.items()):
                with [col1, col2, col3, col4][i % 4]:
                    pct = count / len(df) * 100
                    st.metric(status.replace('_', ' ').title(), f"{count} ({pct:.1f}%)")
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("Resolution Status Visualizations")
            fig = plot_resolution_status(df)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Heatmap
            st.subheader("Topic vs Resolution Heatmap")
            fig = plot_heatmap_analysis(df)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Resolution by topic
            st.subheader("Resolution Rate by Topic")
            
            resolution_rate = df.groupby('TOPIC').apply(
                lambda x: (x['RESOLUTION_STATUS'] == 'resolved').sum() / len(x) * 100
            ).sort_values(ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            resolution_rate.plot(kind='barh', ax=ax, color='green', alpha=0.7)
            ax.set_xlabel('Resolution Rate (%)')
            ax.set_title('Resolution Rate by Topic (Top 20)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # ========================================================================
        # TAB 4: SENTIMENT ANALYSIS
        # ========================================================================
        with tab4:
            st.header("Sentiment Analysis")
            
            # Sentiment metrics
            col1, col2, col3 = st.columns(3)
            
            sentiment_counts = df['USER_SENTIMENT'].value_counts()
            
            sentiments = ['positive', 'neutral', 'negative']
            colors_metric = ['🟢', '🟡', '🔴']
            
            for i, sentiment in enumerate(sentiments):
                if sentiment in sentiment_counts.index:
                    count = sentiment_counts[sentiment]
                    pct = count / len(df) * 100
                    with [col1, col2, col3][i]:
                        st.metric(
                            f"{colors_metric[i]} {sentiment.title()}", 
                            f"{count} ({pct:.1f}%)"
                        )
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("Sentiment Visualizations")
            fig = plot_sentiment_analysis(df)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Sentiment by topic
            st.subheader("Sentiment Distribution by Topic")
            
            sentiment_by_topic = pd.crosstab(df['TOPIC'], df['USER_SENTIMENT'])
            
            # Show top 15 topics
            top_topics = df['TOPIC'].value_counts().head(15).index
            sentiment_filtered = sentiment_by_topic.loc[top_topics]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sentiment_filtered.plot(kind='barh', stacked=True, ax=ax, 
                                   color=['#90EE90', '#FFD700', '#FF6B6B'])
            ax.set_xlabel('Count')
            ax.set_title('Sentiment by Topic (Top 15)', fontsize=14, fontweight='bold')
            ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        # ========================================================================
        # TAB 5: TEXT ANALYSIS
        # ========================================================================
        with tab5:
            st.header("Text Analysis")
            
            # Text statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Query Statistics")
                st.metric("Average Query Length", f"{df['query_word_count'].mean():.1f} words")
                st.metric("Median Query Length", f"{df['query_word_count'].median():.0f} words")
                st.metric("Max Query Length", f"{df['query_word_count'].max():.0f} words")
                st.metric("Min Query Length", f"{df['query_word_count'].min():.0f} words")
            
            with col2:
                st.subheader("📊 Resolution Statistics")
                st.metric("Average Resolution Length", f"{df['resolution_word_count'].mean():.1f} words")
                st.metric("Median Resolution Length", f"{df['resolution_word_count'].median():.0f} words")
                st.metric("Max Resolution Length", f"{df['resolution_word_count'].max():.0f} words")
                st.metric("Min Resolution Length", f"{df['resolution_word_count'].min():.0f} words")
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📊 Text Length Distributions")
            fig = plot_text_analysis(df)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Keywords
            st.subheader("🔑 Top Keywords in Queries")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                keywords = extract_keywords(df, top_n=30)
                keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
                
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(keyword_df['Keyword'][::-1], keyword_df['Count'][::-1], color='steelblue')
                ax.set_xlabel('Frequency')
                ax.set_title('Top 30 Keywords in User Queries', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(keyword_df, use_container_width=True, height=400)
        
        # ========================================================================
        # TAB 6: EXPORT REPORTS
        # ========================================================================
        with tab6:
            st.header("📥 Export Reports")
            
            st.info("Generate and download comprehensive analysis reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Available Reports")
                
                # Topic Summary Report
                topic_summary = df.groupby('TOPIC').agg({
                    'SESSION_ID': 'count',
                    'RESOLUTION_STATUS': lambda x: (x == 'resolved').sum() / len(x) * 100,
                    'USER_SENTIMENT': lambda x: (x == 'positive').sum() / len(x) * 100,
                    'query_word_count': 'mean',
                    'resolution_word_count': 'mean'
                }).round(2)
                topic_summary.columns = ['Total_Conversations', 'Resolution_Rate_%', 
                                        'Positive_Sentiment_%', 'Avg_Query_Words', 'Avg_Response_Words']
                topic_summary = topic_summary.sort_values('Total_Conversations', ascending=False)
                
                csv1 = topic_summary.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📄 Download Topic Summary",
                    data=csv1,
                    file_name=f"topic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Resolution Analysis Report
                resolution_analysis = pd.crosstab(df['TOPIC'], df['RESOLUTION_STATUS'], margins=True)
                csv2 = resolution_analysis.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📄 Download Resolution Analysis",
                    data=csv2,
                    file_name=f"resolution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Sentiment Analysis Report
                sentiment_analysis = pd.crosstab(df['TOPIC'], df['USER_SENTIMENT'], margins=True)
                csv3 = sentiment_analysis.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📄 Download Sentiment Analysis",
                    data=csv3,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Full Dataset Export
                csv4 = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Full Dataset (with features)",
                    data=csv4,
                    file_name=f"full_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.subheader("💡 Key Insights")
                
                # Calculate insights
                resolved_pct = (df['RESOLUTION_STATUS'] == 'resolved').sum() / len(df) * 100
                dropoff_pct = (df['RESOLUTION_STATUS'] == 'user_drop_off').sum() / len(df) * 100
                negative_pct = (df['USER_SENTIMENT'] == 'negative').sum() / len(df) * 100
                
                st.markdown(f"""
                **📊 Overall Performance:**
                - Resolution Rate: **{resolved_pct:.1f}%**
                - User Drop-off Rate: **{dropoff_pct:.1f}%**
                - Negative Sentiment: **{negative_pct:.1f}%**
                
                **🔝 Top Topic:**
                - {df['TOPIC'].value_counts().index[0]} ({df['TOPIC'].value_counts().iloc[0]} conversations)
                
                **📝 Query Complexity:**
                - Average: **{df['query_word_count'].mean():.1f} words**
                
                **💡 Recommendations:**
                - {'✅ Good resolution rate!' if resolved_pct > 50 else '⚠️ Focus on improving resolution rates'}
                - {'✅ Low drop-off rate!' if dropoff_pct < 15 else '⚠️ Investigate user drop-offs'}
                - {'✅ Positive user sentiment!' if negative_pct < 10 else '⚠️ Address negative sentiment'}
                """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Conversation Analytics Dashboard</p>
        <p>Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
