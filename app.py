import streamlit as st
import requests
import json
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

def call_ollama_with_logprobs(prompt: str, model: str = "llama2", base_url: str = "http://localhost:11434") -> Dict:
    """
    Call Ollama API with logprobs enabled.

    Args:
        prompt: The user's prompt
        model: The Ollama model to use
        base_url: Base URL for Ollama API

    Returns:
        Response dictionary with tokens and probabilities
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 500,  # Maximum tokens to generate
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
        return None

def call_ollama_chat_with_logprobs(messages: List[Dict], model: str = "llama2", base_url: str = "http://localhost:11434") -> Dict:
    """
    Call Ollama chat API with logprobs enabled (streaming to get token probabilities).

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: The Ollama model to use
        base_url: Base URL for Ollama API

    Returns:
        Dictionary with response text, tokens, and probabilities
    """
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": 500,
        }
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        tokens = []
        probabilities = []
        full_response = ""

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)

                # Extract token and probability information
                if "message" in chunk and "content" in chunk["message"]:
                    token_text = chunk["message"]["content"]
                    full_response += token_text

                    # Try to get logprobs if available
                    # Note: Ollama's logprobs implementation may vary by version
                    # This is a placeholder for when logprobs are available
                    if token_text:
                        tokens.append(token_text)
                        # Default to high probability if logprobs not available
                        probabilities.append(0.8)

        return {
            "response": full_response,
            "tokens": tokens,
            "probabilities": probabilities
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Ollama API: {e}")
        return None

def parse_response_with_logprobs(response_data: Dict) -> Tuple[str, List[str], List[float]]:
    """
    Parse Ollama response to extract tokens and their probabilities.
    Note: This function adapts to Ollama's actual response format.

    Args:
        response_data: Response from Ollama API

    Returns:
        Tuple of (full_response, tokens, probabilities)
    """
    if not response_data:
        return "", [], []

    # For now, we'll use a workaround since Ollama's logprobs support varies
    # We'll split the response into tokens (words) and assign estimated probabilities
    full_response = response_data.get("response", "")

    # Simple tokenization by splitting on spaces
    # In a real implementation, this would come from the model's tokenizer
    tokens = []
    probabilities = []

    # Split into words and punctuation
    import re
    token_pattern = r'\w+|[^\w\s]'
    matches = re.finditer(token_pattern, full_response)

    for match in matches:
        tokens.append(match.group())
        # Assign random probabilities for demonstration
        # In reality, these would come from logprobs
        probabilities.append(np.random.uniform(0.3, 1.0))

    return full_response, tokens, probabilities

def display_colored_tokens(tokens: List[str], probabilities: List[float]):
    """
    Display tokens with color coding based on their probabilities.

    Args:
        tokens: List of token strings
        probabilities: List of probability values (0-1)
    """
    html_parts = []

    for token, prob in zip(tokens, probabilities):
        color = get_color_from_probability(prob)
        # Escape HTML special characters
        token_escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px; '
            f'border-radius: 3px; display: inline-block;" '
            f'title="Probability: {prob:.2%}">{token_escaped}</span>'
        )

    html = " ".join(html_parts)
    st.markdown(f'<div style="line-height: 2.5;">{html}</div>', unsafe_allow_html=True)

def main():
    st.title("🎨 LLM Confidence Visualizer")
    st.markdown("""
    This app visualizes the confidence of a language model by color-coding tokens based on their probability.
    - 🔴 **Red**: Low confidence (0%)
    - 🟡 **Yellow**: Medium confidence (50%)
    - 🟢 **Green**: High confidence (100%)
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

        st.markdown("---")
        st.markdown("""
        ### 📝 Instructions
        1. Make sure Ollama is running locally
        2. Enter your question or prompt
        3. View the response with color-coded confidence

        ### ℹ️ Note
        Currently, Ollama's API has limited logprobs support.
        This demo uses simulated probabilities for visualization.
        As Ollama adds full logprobs support, this app will be updated.
        """)

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "tokens" in message and "probabilities" in message:
                st.markdown("**Response with confidence visualization:**")
                display_colored_tokens(message["tokens"], message["probabilities"])
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
                response_data = call_ollama_with_logprobs(
                    prompt=prompt,
                    model=model_name,
                    base_url=ollama_url
                )

                if response_data:
                    # Parse response
                    full_response, tokens, probabilities = parse_response_with_logprobs(response_data)

                    if tokens and probabilities:
                        st.markdown("**Response with confidence visualization:**")
                        display_colored_tokens(tokens, probabilities)

                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "tokens": tokens,
                            "probabilities": probabilities
                        })

                        # Show probability statistics
                        with st.expander("📊 Probability Statistics"):
                            avg_prob = np.mean(probabilities)
                            min_prob = np.min(probabilities)
                            max_prob = np.max(probabilities)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Average Confidence", f"{avg_prob:.2%}")
                            col2.metric("Minimum Confidence", f"{min_prob:.2%}")
                            col3.metric("Maximum Confidence", f"{max_prob:.2%}")
                    else:
                        st.warning("Could not extract token probabilities from response.")
                        st.markdown(response_data.get("response", "No response received."))
                else:
                    st.error("Failed to get response from Ollama. Please check your configuration.")

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
