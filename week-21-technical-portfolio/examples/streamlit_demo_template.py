"""
Streamlit Demo Template for AI/ML Projects

A professional template for creating interactive demos with Streamlit.
Features:
- Multi-page app structure
- Session state management
- Caching
- Professional styling
- Error handling
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import logging
from typing import Optional

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="AI Demo Template",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/yourrepo',
        'Report a bug': "https://github.com/yourusername/yourrepo/issues",
        'About': "# AI Demo Template\nBuilt with Streamlit"
    }
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if 'results' not in st.session_state:
    st.session_state.results = []

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# ============================================================================
# Caching Functions
# ============================================================================

@st.cache_resource
def load_model():
    """Load ML model - cached to avoid reloading"""
    try:
        with st.spinner("Loading model..."):
            model = pipeline("sentiment-analysis")
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data(ttl=3600)
def process_data(data: str) -> dict:
    """Cache processed data for 1 hour"""
    # Your data processing logic here
    return {"processed": data}

# ============================================================================
# Core Functions
# ============================================================================

def predict_sentiment(text: str, model) -> dict:
    """
    Predict sentiment of input text

    Args:
        text: Input text to analyze
        model: Loaded ML model

    Returns:
        Dictionary with results
    """
    if not text or not text.strip():
        return {"error": "Please enter some text"}

    if len(text) > 5000:
        return {"error": "Text too long. Maximum 5000 characters."}

    try:
        result = model(text)[0]
        return {
            "label": result['label'],
            "score": result['score'],
            "text": text,
            "success": True
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/150", width=150)
    st.title("🤖 AI Demo")

    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    model_name = st.selectbox(
        "Model",
        ["distilbert-base", "roberta-base"],
        help="Choose the model for analysis"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values = more randomness"
    )

    max_length = st.slider(
        "Max Length",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum output length"
    )

    st.markdown("---")

    st.markdown("### 📊 Statistics")

    st.metric(
        label="Total Predictions",
        value=len(st.session_state.results),
        delta=1 if st.session_state.results else 0
    )

    if st.session_state.results:
        positive_count = sum(1 for r in st.session_state.results if r.get('label') == 'POSITIVE')
        st.metric(
            label="Positive Sentiment",
            value=f"{positive_count}/{len(st.session_state.results)}"
        )

    st.markdown("---")

    st.markdown("### 🔗 Links")
    st.markdown("- [GitHub](https://github.com/yourusername)")
    st.markdown("- [Documentation](https://your-docs.com)")
    st.markdown("- [Report Issue](https://github.com/yourusername/yourrepo/issues)")

    if st.button("🗑️ Clear History"):
        st.session_state.results = []
        st.rerun()

# ============================================================================
# Main Content
# ============================================================================

st.title("🤖 AI Sentiment Analysis Demo")

st.markdown("""
Welcome to the AI Sentiment Analysis demo! This application uses state-of-the-art
NLP models to analyze the sentiment of text.

[![GitHub](https://img.shields.io/badge/GitHub-View_Code-blue?logo=github)](https://github.com/yourusername/yourrepo)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
""")

# Load model
model = load_model()

if model is None:
    st.error("❌ Failed to load model. Please refresh the page.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Analysis", "📊 Results", "ℹ️ About"])

# ============================================================================
# Tab 1: Analysis
# ============================================================================

with tab1:
    st.header("Text Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        input_text = st.text_area(
            "Enter text to analyze",
            height=200,
            placeholder="Type or paste your text here...",
            help="Maximum 5000 characters",
            key="text_input"
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

        with col_btn2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.text_input = ""
                st.rerun()

    with col2:
        st.info("""
        **How to use:**

        1. Enter your text
        2. Adjust settings (sidebar)
        3. Click "Analyze"
        4. View results below

        **Tips:**
        - Longer text = better analysis
        - Try different models
        - Check the Results tab
        """)

    # Process input
    if analyze_btn and input_text:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(input_text, model)

            if result.get("success"):
                # Store in session state
                st.session_state.results.append(result)

                # Display results
                st.markdown("### 📊 Results")

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.metric(
                        label="Sentiment",
                        value=result['label'],
                        delta=f"{result['score']:.2%} confident"
                    )

                with col_res2:
                    # Progress bar
                    st.progress(result['score'])
                    st.caption(f"Confidence: {result['score']:.4f}")

                # Success message
                st.markdown(f"""
                <div class="success-box">
                    ✅ Analysis complete! The text appears to be <strong>{result['label']}</strong>
                    with {result['score']:.1%} confidence.
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="error-box">
                    ❌ {result.get('error', 'Unknown error occurred')}
                </div>
                """, unsafe_allow_html=True)

    # Examples
    st.markdown("---")
    st.subheader("💡 Try These Examples")

    examples = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Terrible service.",
        "The movie was okay. Not great, not terrible, just average."
    ]

    cols = st.columns(len(examples))

    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.text_input = example
                st.rerun()
            st.caption(example[:50] + "...")

# ============================================================================
# Tab 2: Results
# ============================================================================

with tab2:
    st.header("Analysis History")

    if not st.session_state.results:
        st.info("No results yet. Analyze some text to see results here!")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.results)

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Analyses", len(df))

        with col2:
            positive_pct = (df['label'] == 'POSITIVE').sum() / len(df) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")

        with col3:
            avg_confidence = df['score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

        # Chart
        st.subheader("Sentiment Distribution")

        fig = px.histogram(
            df,
            x='label',
            color='label',
            title='Sentiment Distribution',
            labels={'label': 'Sentiment', 'count': 'Count'},
            color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Detailed Results")

        display_df = df[['text', 'label', 'score']].copy()
        display_df['text'] = display_df['text'].str[:100] + '...'
        display_df['score'] = display_df['score'].apply(lambda x: f"{x:.2%}")

        st.dataframe(
            display_df,
            column_config={
                "text": "Text",
                "label": st.column_config.TextColumn("Sentiment"),
                "score": "Confidence"
            },
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )

# ============================================================================
# Tab 3: About
# ============================================================================

with tab3:
    st.header("About This Demo")

    st.markdown("""
    ### 🎯 Overview

    This demo showcases sentiment analysis using state-of-the-art NLP models.
    It's built with Streamlit and the Transformers library.

    ### 🛠️ Tech Stack

    - **Framework:** Streamlit
    - **ML Library:** Transformers (Hugging Face)
    - **Visualization:** Plotly
    - **Language:** Python 3.11+

    ### 📊 Model Information

    - **Task:** Sentiment Analysis
    - **Architecture:** Transformer-based
    - **Training Data:** Customer reviews, social media
    - **Languages:** English

    ### 🚀 Features

    - Real-time sentiment analysis
    - Interactive visualizations
    - Result history and statistics
    - Export to CSV
    - Multiple model support

    ### ⚠️ Limitations

    - English text only
    - Maximum 5000 characters
    - Results may vary based on context
    - Not suitable for medical/legal decisions

    ### 📝 License

    This project is licensed under the MIT License.

    ### 📬 Contact

    - **GitHub:** [@yourusername](https://github.com/yourusername)
    - **Email:** your.email@example.com
    - **Website:** [your-website.com](https://your-website.com)

    ### 🙏 Acknowledgments

    - Built with [Streamlit](https://streamlit.io)
    - Models from [Hugging Face](https://huggingface.co)
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit | <a href='https://github.com/yourusername/yourrepo'>View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
