import streamlit as st
import requests
import pandas as pd
import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = os.environ.get("API_URL", "http://backend:8000")

# Set page config
st.set_page_config(
    page_title="Serendip Experiential Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Serendip Experiential Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze Sri Lankan Tourism Experiences</div>', unsafe_allow_html=True)

st.markdown("""
This application analyzes tourism reviews to identify key experiential dimensions in Sri Lankan tourism:
* 🌱 **Regenerative & Eco-Tourism**: Travel focused on positive social/environmental impact
* 🧘 **Integrated Wellness**: Journeys combining physical and mental well-being  
* 🍜 **Immersive Culinary**: Experiences centered on authentic local cuisine
* 🌄 **Off-the-Beaten-Path Adventure**: Exploration of less-crowded natural landscapes
""")

# Sidebar
st.sidebar.image("https://placehold.co/200x100/1E88E5/FFFFFF?text=SERENDIP", use_container_width=True)
st.sidebar.markdown("## About")
st.sidebar.info(
    "Serendip Experiential Engine uses NLP and explainable AI to help tourism "
    "stakeholders understand visitor experiences and preferences through advanced text analytics."
)

def fetch_dimensions():
    """Fetch available dimensions from the API"""
    try:
        response = requests.get(f"{API_URL}/dimensions")
        if response.status_code == 200:
            return response.json().get("dimensions", [])
        else:
            st.error(f"Error fetching dimensions: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return [
            "Regenerative & Eco-Tourism",
            "Integrated Wellness",
            "Immersive Culinary",
            "Off-the-Beaten-Path Adventure"
        ]  # Fallback to default dimensions

def analyze_review(review_text):
    """Send review to API for analysis"""
    try:
        response = requests.post(
            f"{API_URL}/classify",
            json={"text": review_text}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error analyzing review: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None

# Main application logic
with st.container():
    st.markdown("## 📝 Enter a Tourism Review")
    
    # Sample reviews
    sample_reviews = [
        "We loved the eco-friendly resort that used solar power and served organic local food. Their conservation efforts were impressive!",
        "The yoga retreat by the beach offered amazing Ayurvedic treatments and meditation sessions that completely refreshed me.",
        "The cooking class taught us how to make authentic Sri Lankan curry and hoppers. We visited the spice market with the chef too!",
        "We hiked through remote villages in the mountains, staying with local families and seeing waterfalls that few tourists visit."
    ]
    
    sample_select = st.selectbox("Try a sample review or write your own:", 
                                ["Write my own"] + sample_reviews)
    
    if sample_select == "Write my own":
        review_text = st.text_area("Enter your review:", 
                                  height=150,
                                  placeholder="Describe your tourism experience in Sri Lanka...")
    else:
        review_text = sample_select
        st.text_area("Review text:", sample_select, height=150)
    
    # Analyze button
    if st.button("Analyze Experience Dimensions", type="primary"):
        if not review_text:
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner('Analyzing review...'):
                # Call backend API
                result = analyze_review(review_text)
                
                if result:
                    st.success("Analysis complete!")
                    
                    # Display results
                    st.markdown("## 📊 Experience Dimension Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a bar chart of dimension scores
                        dimensions = list(result["predictions"].keys())
                        scores = list(result["predictions"].values())
                        
                        # Convert scores to percentages
                        scores_pct = [round(score * 100, 1) for score in scores]
                        
                        # Create dataframe for plotting
                        df_scores = pd.DataFrame({
                            'Dimension': dimensions,
                            'Score': scores_pct
                        })
                        
                        fig = px.bar(
                            df_scores,
                            y='Dimension',
                            x='Score',
                            orientation='h',
                            color='Score',
                            color_continuous_scale='Viridis',
                            labels={'Score': 'Confidence Score (%)'},
                            title='Experience Dimension Confidence Scores'
                        )
                        
                        fig.update_layout(
                            height=400, 
                            yaxis={'categoryorder':'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Key Influencing Factors")
                        st.info("These words and phrases most strongly influenced the classification:")
                        
                        # Show the top dimension
                        top_dim = max(result["predictions"].items(), key=lambda x: x[1])[0]
                        
                        st.markdown(f"#### Top Dimension: {top_dim}")
                        
                        if "explanation" in result and top_dim in result["explanation"]:
                            # Create table of top words and their importance
                            words = [item["word"] for item in result["explanation"][top_dim]]
                            values = [round(item["value"] * 100, 1) for item in result["explanation"][top_dim]]
                            
                            df_words = pd.DataFrame({
                                'Word': words,
                                'Importance': values
                            })
                            
                            st.dataframe(
                                df_words.sort_values(by='Importance', ascending=False),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.write("Explanation data not available")

# Footer
st.markdown("---")
st.markdown('<div class="info-text">Serendip Experiential Engine | Powered by Explainable AI</div>', unsafe_allow_html=True)