# LLM Confidence Visualisation

Uses token probabilities to visualise how confident an LLM is about its response.

## Overview

This Streamlit application provides a visual representation of language model confidence by color-coding each token in the response based on its probability. Tokens are colored on a gradient:
- 🔴 **Red**: Low confidence (0%)
- 🟡 **Yellow**: Medium confidence (50%)
- 🟢 **Green**: High confidence (100%)

## Features

- 💬 Interactive chat interface
- 🎨 Color-coded token visualization based on probability
- 📊 Probability statistics (average, min, max, std deviation)
- ⚙️ Configurable Ollama model and API endpoint
- 💾 Chat history management
- 📦 **Built-in model management** - Download models directly from the app
- 📈 Real-time download progress tracking
- 🔍 Automatic model detection and availability checks
- ✅ Connection status monitoring

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) v0.12.11 or later (required for logprobs support)
- [ollama-python](https://pypi.org/project/ollama/) v0.6.1 or later (for logprobs in Python library)
- **No models required initially** - You can download models directly from the app!

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/llm-confidence-visualisation.git
cd llm-confidence-visualisation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Check your Ollama version (must be v0.12.11+):
```bash
ollama --version
```

If you need to update Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

4. Make sure Ollama is running:
```bash
ollama serve
```

That's it! You can download models directly from the app interface.

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. **Download a model** (if you don't have any):
   - The app will detect if you have no models installed
   - Enter a model name (e.g., `llama3.2`, `mistral`, `phi3`)
   - Click "📥 Download" and watch the progress bar
   - Popular models:
     - `llama3.2` - Latest Llama model (small, fast)
     - `mistral` - Mistral 7B (great performance)
     - `phi3` - Microsoft Phi-3 (compact)
     - `codellama` - Specialized for code

4. **Select your model** from the dropdown in the sidebar

5. Type your question in the chat input and press Enter

6. View the response with color-coded tokens showing the model's confidence

## How It Works

1. **User Input**: You enter a prompt in the chat interface
2. **API Call**: The app calls the Ollama API with `"logprobs": true` parameter
3. **Token Extraction**: Each response chunk includes:
   - `token`: The actual token text
   - `logprob`: The log probability of that token
   - `bytes`: Raw byte representation
4. **Probability Calculation**: Log probabilities are converted to standard probabilities using `p = e^(logprob)`
5. **Visualization**: Each token is color-coded based on its probability:
   - Probability < 0.5: Red to Yellow gradient
   - Probability ≥ 0.5: Yellow to Green gradient

## Technical Details

### Probability to Color Mapping

```python
# Red to Yellow (0% to 50%)
if prob < 0.5:
    r = 255
    g = 255 * (prob * 2)
    b = 0

# Yellow to Green (50% to 100%)
else:
    r = 255 * (1 - (prob - 0.5) * 2)
    g = 255
    b = 0
```

### Logprobs Support

This app uses **actual log probabilities** from Ollama's API (available since v0.12.11, released November 2024). The logprobs feature provides the model's confidence for each generated token.

#### API Request Example

```python
payload = {
    "model": "llama2",
    "prompt": "Why is the sky blue?",
    "stream": True,
    "logprobs": True  # Enable log probabilities
}
```

#### Response Format

Each streaming chunk includes logprobs data:

```json
{
  "model": "llama2",
  "response": "The",
  "logprobs": [
    {
      "token": "The",
      "logprob": -0.5234,
      "bytes": [84, 104, 101]
    }
  ]
}
```

## Configuration

The app provides several configuration options in the sidebar:
- **Model Selection**: Choose from your installed models
- **Download More Models**: Add new models anytime via the UI
- **Use Chat API**: Toggle between Chat and Generate endpoints
- **Connection Status**: Real-time Ollama connection monitoring

## Troubleshooting

**Issue**: "Error calling Ollama API"
- Solution: Ensure Ollama is running (`ollama serve`)
- Check that the API URL is correct
- Verify the model name is correct and downloaded

**Issue**: No response appears
- Solution: Check Ollama logs for errors
- Try a different model
- Ensure your model has enough context length for the prompt

**Issue**: No logprobs data received
- Solution: Ensure you're running Ollama v0.12.11 or later (`ollama --version`)
- Update Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- Some models may not support logprobs - try a different model

**Issue**: 404 or 405 error when connecting to Ollama
- Solution: Make sure Ollama is running (`ollama serve`)
- Check that no models are in a broken state (`ollama list`)
- Restart Ollama if needed

**Issue**: Model download stuck or slow
- Solution: Check your internet connection
- Some models are large (several GB) - be patient
- Try a smaller model like `phi3` or `llama3.2`

## Future Enhancements

- [ ] Add support for `top_logprobs` to show alternative token predictions
- [ ] Support for alternative LLM providers (OpenAI, Anthropic, etc.)
- [ ] Customizable color schemes and gradients
- [ ] Export visualizations as images/PDFs
- [ ] Token-level probability inspection with detailed tooltips
- [ ] Probability threshold alerts and warnings
- [ ] Real-time streaming visualization
- [ ] Perplexity calculation and display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM inference
- [Streamlit](https://streamlit.io/) for the web framework
