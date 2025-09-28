import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import json
import os

# Import our modules
from api_service import check_health, get_dimensions, analyze_review
from config import UI_CONFIG, API_CONFIG, DIMENSIONS, SAMPLE_REVIEWS, SCORE_THRESHOLDS, OPENAI_MODEL_DEFAULT

# Import GenAI benchmark module
try:
    from genai_module import run_genai_benchmark, compare_results
    GENAI_AVAILABLE = UI_CONFIG["genai_enabled"]
except ImportError:
    GENAI_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"],
)

# Load custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Check API health at startup
api_status = check_health()
if not api_status["available"]:
    st.warning("⚠️ Backend API not available. Some features may not work correctly.")
elif not api_status["model_loaded"]:
    st.info("ℹ️ Model is still loading. First analysis may take longer than usual.")

# App header
st.markdown('<div class="main-header">Serendip Experiential Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze Sri Lankan Tourism Experiences</div>', unsafe_allow_html=True)

description = "This application analyzes tourism reviews to identify key experiential dimensions in Sri Lankan tourism:"
for dim in DIMENSIONS:
    description += f"\n* {dim['icon']} **{dim['name']}**: {dim['description']}"

st.markdown(description)

# Sidebar
st.sidebar.image(UI_CONFIG["logo_url"], use_column_width=True)
st.sidebar.markdown("## About")
st.sidebar.info(
    "Serendip Experiential Engine uses NLP and explainable AI to help tourism "
    "stakeholders understand visitor experiences and preferences through advanced text analytics."
)

# Main application logic
with st.container():
    st.markdown("## 📝 Enter a Tourism Review")
    
    # Sample reviews from config
    sample_select = st.selectbox("Try a sample review or write your own:", 
                               ["Write my own"] + SAMPLE_REVIEWS)
    
    if sample_select == "Write my own":
        review_text = st.text_area("Enter your review:", 
                                  height=150,
                                  placeholder="Describe your tourism experience in Sri Lanka...")
    else:
        review_text = sample_select
        st.text_area("Review text:", sample_select, height=150)
    
    # Initialize session state for caching
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}
    
    # Analyze button
    if st.button("Analyze Experience Dimensions", type="primary"):
        if not review_text:
            st.warning("Please enter a review to analyze.")
        else:
            # Check cache first
            if review_text in st.session_state.analysis_cache:
                result = st.session_state.analysis_cache[review_text]
                st.success("Analysis retrieved from cache!")
            else:
                with st.spinner('Analyzing review...'):
                    # Call backend API
                    result = analyze_review(review_text)
                    
                    # Cache the result if successful
                    if result:
                        st.session_state.analysis_cache[review_text] = result
            
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
                        text='Score'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Confidence Score (%)",
                        yaxis_title=None,
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Find top dimension
                    top_dim_name = max(result["predictions"].items(), key=lambda x: x[1])[0]
                    top_dim = top_dim_name
                    top_score = result["predictions"][top_dim_name]
                    
                    # Get the corresponding dimension info
                    top_dim_info = next((d for d in DIMENSIONS if d["name"] == top_dim_name), None)
                    
                    # Display the top dimension card
                    st.markdown(f"### {top_dim_info['icon']} Top Dimension")
                    
                    st.markdown(f"""
                    <div class="dimension-card">
                        <div class="dimension-score">{round(top_score * 100)}%</div>
                        <div class="dimension-name">{top_dim_name}</div>
                        <div class="dimension-desc">{top_dim_info['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display top influential words
                    st.markdown("### 📊 Key Words")
                    
                    if "explanation" in result and "top_words" in result["explanation"] and top_dim in result["explanation"]["top_words"]:
                        # Create table of top words and their importance
                        words = [item["word"] for item in result["explanation"]["top_words"][top_dim]]
                        values = [round(item["value"] * 100, 1) for item in result["explanation"]["top_words"][top_dim]]
                        
                        # Create DataFrame
                        df_words = pd.DataFrame({
                            "Word": words,
                            "Impact %": values
                        })
                        
                        st.dataframe(df_words, width="stretch", hide_index=True)
                
                # Word Impact Data Table Section
                st.markdown("## 🔍 Explainable AI Visualization")
                st.markdown("### Word Impact Data Table")
                st.info("This table shows how each word contributes to the prediction for each dimension.")
                
                if "explanation" in result and "top_words" in result["explanation"]:
                    # Display word impact as a simple data table
                    st.subheader("Word Impact on Prediction")
                    
                    # Get the top dimension
                    top_dim_name = max(result["predictions"].items(), key=lambda x: x[1])[0]
                    
                    # Get all dimension names
                    all_dimensions = list(result["predictions"].keys())
                    
                    # Create tabs for each dimension
                    tabs = st.tabs(all_dimensions)
                    
                    # Loop through dimensions and create a table for each
                    for i, dim in enumerate(all_dimensions):
                        with tabs[i]:
                            if dim in result["explanation"]["top_words"]:
                                # Create a DataFrame for the current dimension's top words
                                words_data = result["explanation"]["top_words"][dim]
                                
                                # Convert to DataFrame for better display
                                word_df = pd.DataFrame([
                                    {
                                        "Word": item["word"],
                                        "Impact Value": round(item["value"] * 100, 2),
                                        "Direction": "Increases score" if item["is_positive"] else "Decreases score"
                                    } for item in words_data
                                ])
                                
                                # Add styling based on impact direction
                                def highlight_direction(val):
                                    if val == "Increases score":
                                        return 'background-color: rgba(255, 68, 68, 0.2)'
                                    else:
                                        return 'background-color: rgba(51, 102, 204, 0.2)'
                                
                                # Display the styled dataframe
                                st.dataframe(
                                    word_df.style.applymap(highlight_direction, subset=["Direction"]), 
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Show explanation
                                st.info(f"These words have the most influence on the '{dim}' dimension score.")
                            else:
                                st.warning(f"No word impact data available for {dim}")
                    
                    # Add explanation for the table
                    st.caption("**How to interpret:** Words with 'Increases score' push the prediction higher, while 'Decreases score' push it lower. The Impact Value shows the magnitude of the effect (larger numbers = stronger influence).")
                else:
                    st.warning("Word impact data not available.")
                
                # GenAI Comparison Section
                st.markdown("## 🤖 GenAI vs BERT Comparison")
                
                if GENAI_AVAILABLE:
                    with st.spinner("Running GPT-4 few-shot classification..."):
                        try:
                            # Run GenAI benchmark
                            genai_results = run_genai_benchmark(review_text)
                            
                            # Compare with BERT results
                            comparison = compare_results(result["predictions"], genai_results)
                            
                            # Show results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### BERT Classification")
                                # Convert scores to categories for easier comparison
                                bert_categories = {}
                                for dim, score in result["predictions"].items():
                                    if score > SCORE_THRESHOLDS["high"]:
                                        category = "HIGH"
                                    elif score > SCORE_THRESHOLDS["medium"]:
                                        category = "MEDIUM"
                                    else:
                                        category = "LOW"
                                    bert_categories[dim] = category
                                
                                # Create dataframe
                                bert_df = pd.DataFrame({
                                    "Dimension": list(result["predictions"].keys()),
                                    "Score": [f"{round(score * 100)}%" for score in result["predictions"].values()],
                                    "Category": list(bert_categories.values())
                                })
                                st.dataframe(bert_df, hide_index=True, width="stretch")
                            
                            with col2:
                                # Get model name from results metadata or from config
                                model_name = genai_results.get("metadata", {}).get("model", os.environ.get("OPENAI_MODEL", OPENAI_MODEL_DEFAULT))
                                st.markdown(f"### GenAI Classification ({model_name})")
                                # Skip metadata and error keys
                                genai_dims = {k: v for k, v in genai_results.items() 
                                            if k not in ["metadata", "error"]}
                                
                                # Create dataframe
                                genai_df = pd.DataFrame({
                                    "Dimension": list(genai_dims.keys()),
                                    "Category": list(genai_dims.values())
                                })
                                st.dataframe(genai_df, hide_index=True, width="stretch")
                            
                            # Agreement metrics
                            agreement = comparison["agreement_percentage"]
                            st.metric("Model Agreement", f"{round(agreement)}%", 
                                     delta=None)
                            
                            # Tokens used
                            st.caption(f"{model_name} tokens used: {genai_results.get('metadata', {}).get('tokens_used', 'N/A')}")
                            
                        except Exception as e:
                            st.error(f"Error in GenAI comparison: {str(e)}")
                else:
                    st.warning("GenAI comparison not available. Make sure you have the OpenAI API key set in the environment.")

# Footer
st.markdown("---")
st.markdown('<div class="info-text">Serendip Experiential Engine | Powered by Explainable AI</div>', unsafe_allow_html=True)
