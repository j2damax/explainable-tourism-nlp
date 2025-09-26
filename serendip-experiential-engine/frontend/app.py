import streamlit as st
import requests
import pandas as pd
import json
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
try:
    import streamlit_shap
    STREAMLIT_SHAP_AVAILABLE = True
except ImportError:
    STREAMLIT_SHAP_AVAILABLE = False

# Import GenAI benchmark module
try:
    from genai_module import run_genai_benchmark, compare_results
    GENAI_AVAILABLE = True and os.environ.get("OPENAI_API_KEY") is not None
except ImportError:
    GENAI_AVAILABLE = False

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
        # First check if backend is available
        try:
            health_response = requests.get(f"{API_URL}/", timeout=5)
            if health_response.status_code != 200:
                st.error(f"Backend API is not available. Status: {health_response.status_code}")
                st.info("Please wait a moment for the backend service to start and try again.")
                return None
            else:
                model_status = health_response.json().get("model_status", "unknown")
                if model_status != "loaded":
                    st.warning("Model is still loading. Your first request may take a minute to process.")
        except requests.exceptions.RequestException as e:
            st.error(f"Cannot connect to backend API: {str(e)}")
            st.info("Please wait a moment for the backend service to start and try again.")
            return None
            
        # Call the predict endpoint with timeout
        predict_response = requests.post(
            f"{API_URL}/predict",
            json={"review_text": review_text},
            timeout=60  # Set a longer timeout for the first request that might load the model
        )
        
        # If predict succeeds, call the explain endpoint
        if predict_response.status_code == 200:
            explain_response = requests.post(
                f"{API_URL}/explain",
                json={"review_text": review_text, "top_n_words": 5},
                timeout=30
            )
            
            if explain_response.status_code == 200:
                # Format the response to match the expected structure
                predictions = predict_response.json()
                explanations = explain_response.json()
                
                # Convert to the format expected by the frontend
                results = {
                    "predictions": {item["label"]: item["score"] for item in predictions},
                    "explanation": explanations.get("explanation", {"top_words": {}, "html": ""})
                }
                return results
            else:
                # If explanation fails, we can still return predictions
                st.warning(f"Could not generate explanations: {explain_response.status_code}")
                predictions = predict_response.json()
                return {
                    "predictions": {item["label"]: item["score"] for item in predictions},
                    "explanation": {}
                }
        else:
            st.error(f"Error analyzing review: {predict_response.status_code}")
            try:
                error_detail = predict_response.json().get("detail", "No details provided")
                st.error(f"Error details: {error_detail}")
            except:
                pass
            return None
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("If this is your first analysis, the model may still be loading. Please try again in a minute.")
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
                        
                        if "explanation" in result and "top_words" in result["explanation"] and top_dim in result["explanation"]["top_words"]:
                            # Create table of top words and their importance
                            words = [item["word"] for item in result["explanation"]["top_words"][top_dim]]
                            values = [round(item["value"] * 100, 1) for item in result["explanation"]["top_words"][top_dim]]
                            is_positive = [item.get("is_positive", True) for item in result["explanation"]["top_words"][top_dim]]
                            
                            df_words = pd.DataFrame({
                                'Word': words,
                                'Importance': values,
                                'Impact': ['Positive' if pos else 'Negative' for pos in is_positive]
                            })
                            
                            st.dataframe(
                                df_words.sort_values(by='Importance', ascending=False),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.write("Explanation data not available")
                    
                    # SHAP Visualization Section
                    st.markdown("## 🔍 Explainable AI Visualization")
                    st.markdown("### SHAP Force Plot")
                    st.info("This visualization shows how each word contributes to the prediction for each dimension.")
                    
                    if "explanation" in result and "html" in result["explanation"]:
                        # Display word impact visualization using HTML
                        shap_html = result["explanation"]["html"]
                        
                        # Directly render the HTML content
                        st.subheader("Word Impact on Prediction")
                        components.html(shap_html, height=500, scrolling=True)
                            
                        # Add some explanation about how to interpret the visualization
                        st.caption("**How to interpret:** Red words push the prediction higher, blue words push it lower. The width of the bar indicates the magnitude of the impact.")
                    else:
                        st.warning("SHAP visualization data not available.")
                    
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
                                        if score > 0.7:
                                            category = "HIGH"
                                        elif score > 0.3:
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
                                    st.dataframe(bert_df, hide_index=True, use_container_width=True)
                                
                                with col2:
                                    st.markdown("### GenAI Classification (GPT-4)")
                                    # Skip metadata and error keys
                                    genai_dims = {k: v for k, v in genai_results.items() 
                                                if k not in ["metadata", "error"]}
                                    
                                    # Create dataframe
                                    genai_df = pd.DataFrame({
                                        "Dimension": list(genai_dims.keys()),
                                        "Category": list(genai_dims.values())
                                    })
                                    st.dataframe(genai_df, hide_index=True, use_container_width=True)
                                
                                # Agreement metrics
                                agreement = comparison["agreement_percentage"]
                                st.metric("Model Agreement", f"{round(agreement)}%", 
                                         delta=None)
                                
                                # Comparison table
                                st.markdown("### 📊 Model Comparison")
                                
                                comp_data = {
                                    "Metric": [
                                        "Inference Time", 
                                        "Cost per 1K Reviews", 
                                        "Reproducibility",
                                        "Explainability",
                                        "Customizability"
                                    ],
                                    "BERT": [
                                        comparison["cost_comparison"]["bert"]["inference_time"],
                                        comparison["cost_comparison"]["bert"]["cost_per_1k_reviews"],
                                        comparison["cost_comparison"]["bert"]["reproducibility"],
                                        "High (SHAP visualization)",
                                        "High (can fine-tune on custom data)"
                                    ],
                                    "GenAI (GPT-4)": [
                                        comparison["cost_comparison"]["genai"]["inference_time"],
                                        comparison["cost_comparison"]["genai"]["cost_per_1k_reviews"],
                                        comparison["cost_comparison"]["genai"]["reproducibility"],
                                        "Medium (rationale but no word-level)",
                                        "Medium (prompt engineering)"
                                    ]
                                }
                                
                                comp_df = pd.DataFrame(comp_data)
                                st.dataframe(comp_df, hide_index=True, use_container_width=True)
                                
                                # Tokens used
                                st.caption(f"GPT-4 tokens used: {genai_results.get('metadata', {}).get('tokens_used', 'N/A')}")
                                
                            except Exception as e:
                                st.error(f"Error in GenAI comparison: {str(e)}")
                    else:
                        st.warning("GenAI comparison not available. Make sure you have the OpenAI API key set in the environment.")

# Footer
st.markdown("---")
st.markdown('<div class="info-text">Serendip Experiential Engine | Powered by Explainable AI</div>', unsafe_allow_html=True)