import streamlit as st
import ollama
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import time

# Page configuration
st.set_page_config(
    page_title="LLM Confidence Visualizer",
    page_icon="🎨",
    layout="wide"
)

def get_color_from_probability(prob: float) -> str:
    """
    Convert probability to a color gradient from red (low) to green (high).

    Args:
        prob: Probability value between 0 and 1

    Returns:
        RGB color string
    """
    # Clamp probability between 0 and 1
    prob = max(0.0, min(1.0, prob))

    # Red to Yellow to Green gradient
    if prob < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (prob * 2))
        b = 0
    else:
        # Yellow to Green
        r = int(255 * (1 - (prob - 0.5) * 2))
        g = 255
        b = 0

    return f"rgb({r}, {g}, {b})"

def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = ollama.list()
        # Verify we got a valid response
        return response is not None
    except Exception as e:
        return False

def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        response = ollama.list()

        # Handle the ListResponse object from ollama library
        # The response has a .models attribute containing the list
        if hasattr(response, 'models'):
            models_list = response.models
        elif isinstance(response, dict):
            models_list = response.get('models', [])
        elif isinstance(response, list):
            models_list = response
        else:
            st.error(f"Unexpected response format from ollama.list(): {type(response)}")
            return []

        # Extract model names with error handling
        model_names = []
        for model in models_list:
            try:
                # Handle model objects with attributes
                # Check .model first (primary field in ollama library)
                if hasattr(model, 'model'):
                    model_names.append(model.model)
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                # Handle dict-based models
                elif isinstance(model, dict):
                    name = model.get('model') or model.get('name') or model.get('id')
                    if name:
                        model_names.append(name)
                # Handle plain strings
                elif isinstance(model, str):
                    model_names.append(model)
            except Exception as e:
                st.warning(f"Could not parse model entry: {model}, error: {e}")
                continue

        return model_names

    except Exception as e:
        st.error(f"Error fetching models: {e}")
        # Show more debug info in expander
        with st.expander("🔍 Debug Information"):
            st.code(f"Exception: {type(e).__name__}\nMessage: {str(e)}")
            st.info("Try running `ollama list` in your terminal to see if models are accessible.")
        return []

def pull_model(model_name: str, progress_bar, status_text) -> bool:
    """
    Download an Ollama model with progress tracking.

    Args:
        model_name: Name of the model to download
        progress_bar: Streamlit progress bar object
        status_text: Streamlit text object for status updates

    Returns:
        True if successful, False otherwise
    """
    try:
        status_text.text(f"Pulling model: {model_name}...")

        # Pull the model with streaming progress
        current_digest = None
        for progress in ollama.pull(model_name, stream=True):
            if 'digest' in progress:
                digest = progress['digest']
                if digest != current_digest:
                    current_digest = digest
                    status_text.text(f"Downloading: {digest[:12]}...")

            if 'completed' in progress and 'total' in progress:
                completed = progress['completed']
                total = progress['total']
                if total > 0:
                    progress_percent = completed / total
                    progress_bar.progress(progress_percent)
                    status_text.text(f"Downloading: {completed / (1024**3):.2f}GB / {total / (1024**3):.2f}GB")

            if 'status' in progress:
                if progress['status'] == 'success':
                    progress_bar.progress(1.0)
                    status_text.text("✅ Model downloaded successfully!")
                    return True

        return True
    except Exception as e:
        status_text.text(f"❌ Error downloading model: {e}")
        return False

def generate_with_logprobs(
    prompt: str = None,
    messages: List[Dict] = None,
    model: str = "llama2",
    use_chat: bool = True,
    base_url: str = "http://localhost:11434"
) -> Optional[Dict]:
    """
    Generate response with logprobs using Ollama REST API directly.

    Args:
        prompt: Prompt for generate API
        messages: Messages for chat API
        model: Model name
        use_chat: Whether to use chat API (vs generate)
        base_url: Ollama API base URL

    Returns:
        Dictionary with response, tokens, and probabilities
    """
    try:
        import requests
        import json

        tokens = []
        probabilities = []
        logprobs_data = []
        full_response = ""

        if use_chat and messages:
            # Use chat API with logprobs enabled via REST API
            url = f"{base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": 500,
                    "num_ctx": 2048,
                    "num_probs": 1,  # Enable logprobs (Ollama uses num_probs instead of logprobs)
                }
            }

            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Extract response text
                    if 'message' in chunk and 'content' in chunk['message']:
                        token_text = chunk['message']['content']
                        full_response += token_text

                        # Extract logprobs if available
                        logprobs_list = None
                        if 'logprobs' in chunk:
                            logprobs_list = chunk['logprobs']
                        elif 'message' in chunk and 'logprobs' in chunk['message']:
                            logprobs_list = chunk['message']['logprobs']

                        if logprobs_list:
                            for logprob_entry in logprobs_list:
                                token = logprob_entry.get('token', token_text)
                                logprob = logprob_entry.get('logprob', 0)
                                probability = math.exp(logprob)

                                tokens.append(token)
                                probabilities.append(probability)
                                logprobs_data.append(logprob)
        else:
            # Use generate API with logprobs enabled via REST API
            url = f"{base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 500,
                    "num_ctx": 2048,
                    "num_probs": 1,  # Enable logprobs (Ollama uses num_probs instead of logprobs)
                }
            }

            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Extract response text
                    if 'response' in chunk:
                        token_text = chunk['response']
                        full_response += token_text

                        # Extract logprobs if available
                        if 'logprobs' in chunk and chunk['logprobs']:
                            for logprob_entry in chunk['logprobs']:
                                token = logprob_entry.get('token', token_text)
                                logprob = logprob_entry.get('logprob', 0)
                                probability = math.exp(logprob)

                                tokens.append(token)
                                probabilities.append(probability)
                                logprobs_data.append(logprob)

        return {
            "response": full_response,
            "tokens": tokens,
            "probabilities": probabilities,
            "logprobs": logprobs_data
        }

    except Exception as e:
        st.error(f"Error generating response: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_colored_tokens(tokens: List[str], probabilities: List[float], logprobs: List[float] = None):
    """
    Display tokens with color coding based on their probabilities.

    Args:
        tokens: List of token strings
        probabilities: List of probability values (0-1)
        logprobs: Optional list of log probabilities
    """
    html_parts = []

    for i, (token, prob) in enumerate(zip(tokens, probabilities)):
        color = get_color_from_probability(prob)

        # Escape HTML special characters
        token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Build tooltip text
        tooltip = f"Probability: {prob:.2%}"
        if logprobs and i < len(logprobs):
            tooltip += f" | Logprob: {logprobs[i]:.4f}"

        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
            f'border-radius: 3px; display: inline-block; font-family: monospace;" '
            f'title="{tooltip}">{token_escaped}</span>'
        )

    html = "".join(html_parts)
    st.markdown(f'<div style="line-height: 2.5; white-space: pre-wrap;">{html}</div>', unsafe_allow_html=True)

def show_model_management():
    """Display model management interface."""
    st.header("📦 Model Management")

    # Check connection
    if not check_ollama_connection():
        st.error("⚠️ Cannot connect to Ollama. Please ensure Ollama is running.")
        st.code("ollama serve", language="bash")
        return None

    # Get available models
    available_models = get_available_models()

    if not available_models:
        st.warning("⚠️ No models found. Please download a model to get started.")
        st.info("""
        **Popular models to try:**
        - `llama3.2` - Latest Llama model (small, fast)
        - `llama2` - Llama 2 model
        - `mistral` - Mistral 7B model
        - `codellama` - Code-focused model
        - `phi3` - Microsoft Phi-3 model

        Or visit https://ollama.com/library for more options.
        """)

        # Model download interface
        col1, col2 = st.columns([3, 1])
        with col1:
            model_to_download = st.text_input(
                "Enter model name to download",
                placeholder="e.g., llama3.2, mistral, phi3",
                key="model_download_input"
            )

        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            download_button = st.button("📥 Download", type="primary", key="download_model_btn")

        if download_button and model_to_download:
            progress_bar = st.progress(0)
            status_text = st.empty()

            success = pull_model(model_to_download, progress_bar, status_text)

            if success:
                st.success(f"✅ Model '{model_to_download}' downloaded successfully!")
                time.sleep(2)
                st.rerun()
            else:
                st.error(f"❌ Failed to download model '{model_to_download}'")

        return None
    else:
        st.success(f"✅ Found {len(available_models)} model(s)")

        # Display available models
        with st.expander("📋 Available Models", expanded=True):
            for model in available_models:
                st.write(f"• {model}")

        # Model selection
        selected_model = st.selectbox(
            "Select a model to use",
            available_models,
            key="model_selector"
        )

        # Option to download more models
        with st.expander("➕ Download Another Model"):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_model = st.text_input(
                    "Model name",
                    placeholder="e.g., llama3.2, mistral",
                    key="new_model_input"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("📥 Download", key="download_new_model_btn"):
                    if new_model:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        success = pull_model(new_model, progress_bar, status_text)

                        if success:
                            st.success(f"✅ Model '{new_model}' downloaded!")
                            time.sleep(2)
                            st.rerun()

        return selected_model

def main():
    st.title("🎨 LLM Confidence Visualizer")
    st.markdown("""
    This app visualizes the confidence of a language model by color-coding tokens based on their probability.
    - 🔴 **Red**: Low confidence (0%)
    - 🟡 **Yellow**: Medium confidence (50%)
    - 🟢 **Green**: High confidence (100%)

    Hover over any token to see its exact probability value.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check Ollama connection
        if not check_ollama_connection():
            st.error("❌ Ollama is not running")
            st.info("Start Ollama with: `ollama serve`")
            st.code("# In a terminal, run:\nollama serve", language="bash")
            st.stop()

        st.success("✅ Ollama is running")

        # Model management
        available_models = get_available_models()

        # Debug mode toggle (hidden by default)
        if st.checkbox("🐛 Debug Mode", value=False, help="Show raw model data for troubleshooting"):
            try:
                raw_response = ollama.list()
                st.json(raw_response)
            except Exception as e:
                st.error(f"Could not fetch raw data: {e}")

        if not available_models:
            st.warning("⚠️ No models available")
            if st.button("🔄 Refresh Models"):
                st.rerun()
            model_name = None
        else:
            model_name = st.selectbox(
                "Select Model",
                available_models,
                help="Choose an Ollama model to use"
            )

            # Option to download more
            if st.button("➕ Download More Models"):
                st.session_state.show_download = True

        use_chat_api = st.checkbox(
            "Use Chat API",
            value=True,
            help="Use /api/chat endpoint instead of /api/generate"
        )

        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Make sure Ollama (v0.12.11+) is running
        2. Download a model if you haven't already
        3. Enter your question or prompt
        4. View the response with color-coded confidence

        ### ℹ️ About Models
        - Models are auto-loaded on first use (no need to "start" them)
        - Downloaded models appear in the dropdown above
        - First request may be slower as the model loads into memory
        - Subsequent requests will be faster

        ### 🔧 Requirements
        - Ollama v0.12.11 or later (with logprobs support)
        - Run: `ollama --version` to check
        - Update: `curl -fsSL https://ollama.com/install.sh | sh`
        """)

    # Show model download interface if no models or requested
    if not available_models or st.session_state.get('show_download', False):
        selected_model = show_model_management()
        if st.session_state.get('show_download', False):
            if st.button("✅ Done"):
                st.session_state.show_download = False
                st.rerun()
        if not selected_model:
            st.stop()
        model_name = selected_model

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "tokens" in message and "probabilities" in message:
                st.markdown("**Response with confidence visualization:**")
                display_colored_tokens(
                    message["tokens"],
                    message["probabilities"],
                    message.get("logprobs", None)
                )
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not model_name:
            st.error("Please select or download a model first.")
            st.stop()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # Call Ollama API
                if use_chat_api:
                    # Build messages for chat API
                    messages = []
                    for msg in st.session_state.messages:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                    response_data = generate_with_logprobs(
                        messages=messages,
                        model=model_name,
                        use_chat=True
                    )
                else:
                    response_data = generate_with_logprobs(
                        prompt=prompt,
                        model=model_name,
                        use_chat=False
                    )

                if response_data:
                    tokens = response_data.get("tokens", [])
                    probabilities = response_data.get("probabilities", [])
                    logprobs = response_data.get("logprobs", [])
                    full_response = response_data.get("response", "")

                    if tokens and probabilities:
                        st.markdown("**Response with confidence visualization:**")
                        display_colored_tokens(tokens, probabilities, logprobs)

                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "tokens": tokens,
                            "probabilities": probabilities,
                            "logprobs": logprobs
                        })

                        # Show probability statistics
                        with st.expander("📊 Probability Statistics"):
                            avg_prob = np.mean(probabilities)
                            min_prob = np.min(probabilities)
                            max_prob = np.max(probabilities)
                            std_prob = np.std(probabilities)

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Average Confidence", f"{avg_prob:.2%}")
                            col2.metric("Minimum Confidence", f"{min_prob:.2%}")
                            col3.metric("Maximum Confidence", f"{max_prob:.2%}")
                            col4.metric("Std Deviation", f"{std_prob:.2%}")

                            # Show distribution
                            st.markdown("**Probability Distribution:**")
                            df = pd.DataFrame({
                                "Probability": probabilities,
                                "Token": tokens[:len(probabilities)]
                            })
                            st.bar_chart(df.set_index("Token")["Probability"])
                    else:
                        st.warning("""
                        ⚠️ No logprobs data received from Ollama.

                        This could mean:
                        1. Your Ollama version doesn't support logprobs (need v0.12.11+)
                        2. The model doesn't support logprobs output
                        3. Logprobs weren't enabled in the response

                        Plain response:
                        """)
                        st.markdown(full_response if full_response else "No response received.")
                else:
                    st.error("Failed to get response from Ollama. Please check your configuration.")

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
