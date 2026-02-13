# 🤖 Nexus Agent

**Combining OpenClaw multi-platform connectivity with Agent Zero multi-agent cooperation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://docker.com)

## 🌟 Overview

The Nexus Agent is an open-source framework that combines the best features of **OpenClaw** and **Agent Zero** to create a powerful, local-first AI assistant with multi-platform connectivity and multi-agent cooperation.

### ✨ Key Features

- **Local Execution**: Runs entirely on your hardware for privacy and control
- **Multi-Platform**: Connects to WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, Microsoft Teams, and WebChat
- **Multi-Agent**: Complex task decomposition and delegation using Agent Zero's hierarchical agent system
- **Persistent Memory**: Long-term learning and knowledge retention with vector database integration
- **Extensible**: Custom tools, instruments, and extensions for unlimited customization
- **Open Source**: MIT licensed, community-driven development

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Agent System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │   OpenClaw      │─────▶│   Agent Zero Core Engine     │  │
│  │  Platform Gateway│      │  (Multi-Agent, Memory, Tools)│  │
│  │                 │      │                              │  │
│  │ - WhatsApp      │      │ - Agent Hierarchy            │  │
│  │ - Telegram      │      │ - Memory System              │  │
│  │ - Slack         │      │ - Knowledge Base             │  │
│  │ - Discord       │      │ - Tool Execution             │  │
│  │ - Google Chat   │      │ - Extension Framework        │  │
│  │ - Signal        │      │                              │  │
│  │ - Microsoft     │      │                              │  │
│  │   Teams         │      │                              │  │
│  │ - WebChat       │      │                              │  │
│  └─────────────────┘      └──────────────────────────────┘  │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │   API Bridge    │      │   Unified Memory Store       │  │
│  │  (Message        │      │   (Vector Database)          │  │
│  │   Transformation)│      │                              │  │
│  └─────────────────┘      └──────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (Windows, macOS, Linux)
- Python 3.11+ (for local development)

### Docker Deployment (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nexus-agent.git
cd nexus-agent
```

2. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

3. **Access the Web UI**
Open your browser and navigate to `http://localhost:8080`

### Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the API server**
```bash
python run.py --mode server
```

3. **Run the integration bridge**
```bash
python run.py --mode bridge
```

## 📁 Project Structure

```
nexus-agent/
├── core/                    # Agent Zero core engine
├── gateway/                 # OpenClaw platform gateway
├── integration/             # API bridge and message transformation
│   ├── bridge.py           # Main integration bridge
│   └── api_server.py       # FastAPI server
├── agents/                  # Multi-agent hierarchy
├── tools/                   # Unified tool interface
├── memory/                  # Shared memory system
├── extensions/              # Modular extensions
├── instruments/             # Custom scripts and functions
├── docs/                    # Documentation
├── docker/                  # Deployment configuration
├── config.json              # System configuration
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
└── run.py                  # Main entry point
```

## 🔧 Configuration

The system is configured via `config.json`:

```json
{
  "project_name": "Nexus Agent",
  "version": "0.1.0",
  "api": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "platforms": ["whatsapp", "telegram", "slack", "discord"],
  "llm_providers": ["anthropic", "openai", "local_llm"]
}
```

## 🤝 Integration Points

### OpenClaw Integration

- **Platform Gateway**: Multi-channel messaging support
- **Session Management**: Unified conversation state
- **Skill/Plugin System**: Extensible functionality

### Agent Zero Integration

- **Agent Hierarchy**: Multi-agent task delegation
- **Memory System**: Persistent learning and knowledge
- **Tool Execution**: Code execution and API integration
- **Extension Framework**: Modular functionality

## 🛠️ Development

### Adding Custom Tools

1. Create a new Python file in `tools/`
2. Implement the tool class
3. Register it in the integration bridge

### Adding New Platforms

1. Create a new gateway module in `gateway/`
2. Implement the platform interface
3. Register it in the configuration

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenClaw](https://github.com/openclaw/openclaw) for the platform gateway architecture
- [Agent Zero](https://github.com/agent0ai/agent-zero) for the multi-agent framework
- All contributors and users of this project

## 📞 Support

- 📧 Email: support@nexus-agent.dev
- 💬 Discord: [Join our community](https://discord.gg/nexus-agent)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/nexus-agent/issues)

---

**Made with ❤️ by the Hybrid Agent Community**