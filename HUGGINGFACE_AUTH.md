# Hugging Face Authentication Guide

Some models on Hugging Face are "gated" and require authentication to download. This guide explains how to set up authentication for this project.

## Quick Setup

### Option 1: Environment Variable (Recommended)

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_token_here

# Or add to your .env file (recommended for persistence)
echo "HF_TOKEN=your_token_here" >> .env
```

### Option 2: Config File

Add to your config file (e.g., `config.yaml` or `config-gemma2b.yaml`):

```yaml
huggingface:
  token: your_token_here
```

Or under the model section:

```yaml
model:
  name: "google/gemma-2b-it"
  token: your_token_here
```

### Option 3: Hugging Face CLI Login

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login (will prompt for token)
huggingface-cli login
```

## Getting Your Token

1. Go to [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "simplesql-project")
4. Select "Read" access (sufficient for downloading models)
5. Click "Generate token"
6. Copy the token (you won't be able to see it again!)

## Accepting Model Terms

For gated models like `google/gemma-2b-it`, you also need to:

1. Visit the model page: https://huggingface.co/google/gemma-2b-it
2. Click "Agree and access repository"
3. Accept the terms of use

## Priority Order

The code checks for tokens in this order:
1. Config file (`huggingface.token` or `model.token`)
2. Environment variable (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`)
3. Hugging Face Hub cache (from `huggingface-cli login`)

## Troubleshooting

**Error: "You are trying to access a gated repo"**
- Make sure you've accepted the model's terms on Hugging Face
- Verify your token is set correctly: `echo $HF_TOKEN`
- Try logging in via CLI: `huggingface-cli login`

**Error: "401 Client Error"**
- Your token may be invalid or expired
- Generate a new token and update it

**Error: "Cannot access gated repo"**
- You need to accept the model's terms on the Hugging Face website
- Visit the model page and click "Agree and access repository"
