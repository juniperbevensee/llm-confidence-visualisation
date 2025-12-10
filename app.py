import streamlit as st
import requests
import json
import math
import numpy as np
from typing import List, Dict, Tuple

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

def call_ollama_chat_with_logprobs(
    messages: List[Dict],
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    stream: bool = True
) -> Dict:
    """
    Call Ollama chat API with logprobs enabled.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The Ollama model to use
        base_url: Base URL for Ollama API
        stream: Whether to use streaming mode

    Returns:
        Dictionary with response text, tokens, and probabilities
    """
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "logprobs": True,  # Enable log probabilities
        "options": {
            "num_predict": 500,
        }
    }

    try:
        response = requests.post(url, json=payload, stream=stream, timeout=120)
        response.raise_for_status()

        tokens = []
        probabilities = []
        logprobs_data = []
        full_response = ""

        if stream:
            # Streaming mode - process chunks
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Extract token and probability information
                    if "message" in chunk and "content" in chunk["message"]:
                        token_text = chunk["message"]["content"]
                        full_response += token_text

                        # Get logprobs if available
                        if "logprobs" in chunk and chunk["logprobs"]:
                            for logprob_entry in chunk["logprobs"]:
                                token = logprob_entry.get("token", token_text)
                                logprob = logprob_entry.get("logprob", 0)

                                # Convert logprob to probability: p = e^(logprob)
                                probability = math.exp(logprob)

                                tokens.append(token)
                                probabilities.append(probability)
                                logprobs_data.append(logprob)
        else:
            # Non-streaming mode
            data = response.json()

            if "message" in data and "content" in data["message"]:
                full_response = data["message"]["content"]

                # Get logprobs if available
                if "logprobs" in data and data["logprobs"]:
                    for logprob_entry in data["logprobs"]:
                        token = logprob_entry.get("token", "")
                        logprob = logprob_entry.get("logprob", 0)

                        # Convert logprob to probability: p = e^(logprob)
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

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
        return None

def call_ollama_generate_with_logprobs(
    prompt: str,
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    stream: bool = True
) -> Dict:
    """
    Call Ollama generate API with logprobs enabled.

    Args:
        prompt: The user's prompt
        model: The Ollama model to use
        base_url: Base URL for Ollama API
        stream: Whether to use streaming mode

    Returns:
        Dictionary with response text, tokens, and probabilities
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "logprobs": True,  # Enable log probabilities
        "options": {
            "num_predict": 500,
        }
    }

    try:
        response = requests.post(url, json=payload, stream=stream, timeout=120)
        response.raise_for_status()

        tokens = []
        probabilities = []
        logprobs_data = []
        full_response = ""

        if stream:
            # Streaming mode - process chunks
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                    # Extract response text
                    if "response" in chunk:
                        token_text = chunk["response"]
                        full_response += token_text

                        # Get logprobs if available
                        if "logprobs" in chunk and chunk["logprobs"]:
                            for logprob_entry in chunk["logprobs"]:
                                token = logprob_entry.get("token", token_text)
                                logprob = logprob_entry.get("logprob", 0)

                                # Convert logprob to probability: p = e^(logprob)
                                probability = math.exp(logprob)

                                tokens.append(token)
                                probabilities.append(probability)
                                logprobs_data.append(logprob)
        else:
            # Non-streaming mode
            data = response.json()

            if "response" in data:
                full_response = data["response"]

                # Get logprobs if available
                if "logprobs" in data and data["logprobs"]:
                    for logprob_entry in data["logprobs"]:
                        token = logprob_entry.get("token", "")
                        logprob = logprob_entry.get("logprob", 0)

                        # Convert logprob to probability: p = e^(logprob)
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

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
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

        model_name = st.text_input(
            "Ollama Model",
            value="llama2",
            help="Name of the Ollama model to use (e.g., llama2, mistral, codellama)"
        )

        ollama_url = st.text_input(
            "Ollama API URL",
            value="http://localhost:11434",
            help="Base URL for your Ollama instance"
        )

        use_chat_api = st.checkbox(
            "Use Chat API",
            value=True,
            help="Use /api/chat endpoint instead of /api/generate"
        )

        use_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            help="Stream responses token by token"
        )

        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Make sure Ollama (v0.12.11+) is running locally
        2. Enter your question or prompt
        3. View the response with color-coded confidence

        ### ℹ️ Requirements
        - Ollama v0.12.11 or later (with logprobs support)
        - Run: `ollama --version` to check
        - Update: `curl -fsSL https://ollama.com/install.sh | sh`
        """)

        # Version check warning
        st.warning("⚠️ This app requires Ollama v0.12.11+ for logprobs support")

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

                    response_data = call_ollama_chat_with_logprobs(
                        messages=messages,
                        model=model_name,
                        base_url=ollama_url,
                        stream=use_streaming
                    )
                else:
                    response_data = call_ollama_generate_with_logprobs(
                        prompt=prompt,
                        model=model_name,
                        base_url=ollama_url,
                        stream=use_streaming
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
                            import pandas as pd
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
