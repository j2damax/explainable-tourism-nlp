# Setting Up the OpenAI API Integration

This guide explains how to configure the OpenAI API key for the GenAI benchmarking feature in the Serendip Experiential Engine.

## Prerequisites

- An OpenAI account
- An OpenAI API key (can be created at https://platform.openai.com/api-keys)

## Configuration Steps

### 1. Create an API Key on OpenAI Platform

1. Visit [https://platform.openai.com/signup](https://platform.openai.com/signup) and create an account if you don't have one
2. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Click "Create new secret key"
4. Give your key a name (e.g., "Serendip Tourism Project")
5. Copy the generated API key (you won't be able to see it again!)

### 2. Add Your API Key to `.env`

1. Open the `.env` file in the root directory of the project
2. Locate the `OPENAI_API_KEY` line
3. Replace `your_openai_api_key_here` with your actual OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key
```

### 3. Choose Your OpenAI Model (Optional)

The application uses GPT-3.5-Turbo by default, which is much more affordable than GPT-4:

```bash
# Default (more affordable)
OPENAI_MODEL=gpt-3.5-turbo

# Better quality but more expensive
# OPENAI_MODEL=gpt-4
```

### 2. Protect Your API Key

- **IMPORTANT:** Never commit your `.env` file with the real API key to version control
- The `.env` file is already listed in `.gitignore` to prevent accidental commits
- If you're working in a team, share API keys through secure channels, not in code

### 3. Restart Docker Containers

After updating the `.env` file, restart the containers to apply the changes:

```bash
docker-compose down
docker-compose up -d
```

## Verifying the Integration

To verify that the OpenAI API integration is working:

1. Open the Streamlit frontend at http://localhost:8501
2. Enter a review in the text area
3. Click "Classify & Explain"
4. Check that the "GenAI Benchmark Comparison" section shows results

## Troubleshooting

If you encounter issues with the OpenAI API integration:

1. **Invalid API Key Error**: Double-check that your API key is correct
2. **Rate Limiting**: OpenAI applies rate limits to API usage; check if you've exceeded your quota
3. **Connection Errors**: Ensure your server has internet access to reach the OpenAI API
4. **Environment Variables Not Applied**: Make sure you've restarted the containers after updating `.env`

## Cost Considerations

- **Free Tier**: New OpenAI accounts typically receive $5-10 in free credits that expire after 3 months
- **Model Pricing**:
  - GPT-3.5-turbo: ~$0.001 per 1K tokens (default, very affordable)
  - GPT-4: ~$0.06 per 1K tokens (60x more expensive but potentially more accurate)
- The application displays the token usage and estimated cost for each request
- Monitor your OpenAI usage dashboard to keep track of expenses
- For most testing purposes, GPT-3.5-turbo should be sufficient
