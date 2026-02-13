# 🤖 Nexus Agent

**Combining OpenClaw multi-platform connectivity with Agent Zero multi-agent cooperation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://docker.com)

## 🌟 Overview

Nexus Agent is a local-first framework that combines multi-platform message ingestion with multi-agent orchestration. It provides a single API layer for processing messages, managing tasks, and storing conversational memory.

## ✨ Key Features

- **Local execution** for privacy and control.
- **Multi-platform connectivity** (WhatsApp, Telegram, Slack, Discord, and more).
- **Multi-agent task coordination** for complex workflows.
- **Persistent memory** for context and retrieval.
- **Extensible architecture** for tools and integrations.

## 🏗️ Architecture

- **Platform gateway** receives inbound user messages.
- **Integration API** normalizes messages and exposes service endpoints.
- **Bridge runtime** executes agent workflows and memory operations.
- **Memory store** persists message/response history.

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (recommended), or
- Python 3.11+

### Docker

```bash
docker-compose up -d
```

API: `http://localhost:8080`

### Local development

1. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

2. Install development dependencies (linters/tests):

```bash
pip install -r requirements-dev.txt
```

3. Run the API server:

```bash
python run.py --mode server
```

4. Run the bridge worker (optional, depending on deployment mode):

```bash
python run.py --mode bridge
```

## 🔧 Configuration

Primary settings are defined in `config.json`, including API host/port and supported platforms.

For CORS configuration, set either:

- `CORS_ALLOW_ORIGINS` environment variable (comma-separated), or
- `api.cors_allow_origins` in `config.json`.

Default allowed origins are restricted to local development hosts.

## 📁 Project Structure

```text
nexus-agent/
├── integration/
│   ├── api_server.py
│   └── bridge.py
├── config.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── run.py
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and testing guidance.
