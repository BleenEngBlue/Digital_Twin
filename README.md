---
title: Monica's Digital Twin
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
python_version: "3.13"
---

# Monica's Digital Twin

A RAG-powered digital twin built with Gradio, OpenAI, and ChromaDB.
Chat with Monica's AI persona — ask about her background, skills, and experience.

## Setup

The following secrets must be configured in the Space settings (not stored here):

- `OPENAI_API_KEY` — OpenAI API key for embeddings and generation
- `DIGITAL_TWIN_TOKEN` — HuggingFace token for accessing the private dataset (`Monica-Wu/digital-twin-data`)
- `PUSHOVER_USER_KEY` — Pushover user key for notifications
- `PUSHOVER_API_TOKEN` — Pushover app token for notifications
