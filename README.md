# Voice AI agent — detailing center

Inbound Russian-language phone agent: answers questions from `config/business.yaml`
and collects booking requests, which are sent to the owner via Telegram.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # then fill in the values
```

## Checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```
