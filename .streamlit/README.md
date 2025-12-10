# Streamlit Configuration

This directory contains Streamlit configuration files.

## Files

- **config.toml**: Main configuration file for Streamlit server, logging, and UI settings
- **secrets.toml**: (Not included in repo) For API keys and sensitive data

## Logging Configuration

The app is configured with INFO level logging by default. To see more detailed logs:

1. Change the logging level in `config.toml`:
   ```toml
   [logger]
   level = "debug"  # Options: "error", "warning", "info", "debug"
   ```

2. Or set it via environment variable:
   ```bash
   export STREAMLIT_LOGGER_LEVEL=debug
   streamlit run app.py
   ```

## Console Output

When running the app, you'll see logs in the terminal showing:
- API requests being sent to Ollama
- Request payloads (including num_probs setting)
- Response chunks (first 3 chunks logged)
- Whether logprobs are found in each chunk
- Total tokens extracted with probabilities

This helps debug logprobs issues with Ollama.

## Customizing

See [Streamlit Configuration Documentation](https://docs.streamlit.io/develop/api-reference/configuration/config.toml) for all available options.
