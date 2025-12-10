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
- 📊 Probability statistics (average, min, max)
- ⚙️ Configurable Ollama model and API endpoint
- 💾 Chat history management

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model downloaded (e.g., `llama2`, `mistral`, `codellama`)

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

3. Make sure Ollama is running:
```bash
ollama serve
```

4. Download an Ollama model if you haven't already:
```bash
ollama pull llama2
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Configure the settings in the sidebar:
   - **Ollama Model**: Enter the name of your model (e.g., `llama2`, `mistral`)
   - **Ollama API URL**: Set the URL where Ollama is running (default: `http://localhost:11434`)

4. Type your question in the chat input and press Enter

5. View the response with color-coded tokens showing the model's confidence

## How It Works

1. **User Input**: You enter a prompt in the chat interface
2. **API Call**: The app calls the Ollama API with logprobs enabled
3. **Probability Calculation**: Log probabilities are converted to standard probabilities using `e^(logprob)`
4. **Visualization**: Each token is color-coded based on its probability:
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

### Note on Logprobs

Currently, Ollama's API has limited native support for returning logprobs. The current implementation uses simulated probabilities for demonstration purposes. As Ollama adds full logprobs support in future versions, this app will be updated to use actual token probabilities from the model.

## Configuration

You can customize the app behavior by modifying these settings in the sidebar:
- **Model Name**: Any Ollama model you have installed
- **API URL**: If Ollama is running on a different host/port

## Troubleshooting

**Issue**: "Error calling Ollama API"
- Solution: Ensure Ollama is running (`ollama serve`)
- Check that the API URL is correct
- Verify the model name is correct and downloaded

**Issue**: No response appears
- Solution: Check Ollama logs for errors
- Try a different model
- Ensure your model has enough context length for the prompt

**Issue**: All tokens show the same color
- Solution: This is expected with the current demo implementation. Real logprobs will be used once Ollama API supports them fully.

## Future Enhancements

- [ ] Use actual logprobs from Ollama when API support is available
- [ ] Support for alternative LLM providers (OpenAI, Anthropic, etc.)
- [ ] Customizable color schemes
- [ ] Export visualizations as images
- [ ] Token-level probability inspection
- [ ] Probability threshold alerts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM inference
- [Streamlit](https://streamlit.io/) for the web framework
