# Nexus Agent - Master Build Prompt

## Project Overview

You are tasked with building **Nexus Agent**, a sophisticated open-source AI assistant framework that combines:
- **OpenClaw**: Multi-platform messaging connectivity
- **Agent Zero**: Hierarchical multi-agent cooperation system

This is a local-first, privacy-focused AI assistant with persistent memory, extensible tools, and support for multiple messaging platforms.

---

## Core Architecture

### System Components

1. **Platform Gateway (OpenClaw Integration)**
   - Multi-platform message handling (WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, Microsoft Teams, WebChat)
   - Unified session management across platforms
   - Message normalization and routing
   - Platform-specific authentication and connection management

2. **Agent Zero Core Engine**
   - Hierarchical multi-agent system for task decomposition
   - Parent-child agent delegation model
   - Autonomous decision-making and task planning
   - Tool execution and code generation capabilities
   - Context-aware agent spawning

3. **Integration Bridge**
   - FastAPI-based REST API server
   - Message transformation between OpenClaw and Agent Zero formats
   - Async request handling
   - WebSocket support for real-time communication

4. **Memory System**
   - Vector database integration (ChromaDB)
   - Long-term knowledge retention
   - Contextual memory retrieval
   - Per-user and per-conversation memory isolation
   - Semantic search capabilities

5. **Tool & Extension Framework**
   - Modular tool registration system
   - Dynamic tool discovery and loading
   - Custom instrument execution
   - API integration capabilities
   - Code execution sandbox

---

## Technical Stack

### Core Dependencies
```python
# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Async & HTTP
httpx>=0.25.0
asyncio>=3.4.3

# Vector Database
chromadb>=0.4.18

# LLM Providers
openai>=1.3.5
anthropic>=0.7.0

# Platform SDKs
python-telegram-bot>=20.6
slack-sdk>=3.21.0
discord.py>=2.3.2

# Utilities
loguru>=0.7.2
python-dotenv>=1.0.0
pyyaml>=6.0.1
pandas>=2.1.3
numpy>=1.26.2
```

### Python Version
- Python 3.11+

### Deployment
- Docker containerization
- Docker Compose orchestration

---

## Directory Structure

```
nexus-agent/
├── core/                       # Agent Zero core engine
│   ├── __init__.py
│   ├── agent.py               # Base agent class
│   ├── agent_manager.py       # Agent lifecycle management
│   ├── task_planner.py        # Task decomposition logic
│   ├── context.py             # Agent context management
│   └── prompts/               # System prompts
│       ├── agent_system.md
│       └── tool_system.md
│
├── gateway/                    # OpenClaw platform gateway
│   ├── __init__.py
│   ├── base_platform.py       # Abstract platform interface
│   ├── platforms/
│   │   ├── telegram.py
│   │   ├── slack.py
│   │   ├── discord.py
│   │   ├── whatsapp.py        # WhatsApp Business API
│   │   ├── google_chat.py
│   │   ├── signal.py
│   │   ├── teams.py           # Microsoft Teams
│   │   └── webchat.py         # Web interface
│   ├── session_manager.py     # Cross-platform sessions
│   └── message_router.py      # Message routing logic
│
├── integration/                # API bridge
│   ├── __init__.py
│   ├── api_server.py          # FastAPI application
│   ├── bridge.py              # OpenClaw↔Agent Zero bridge
│   ├── schemas.py             # Pydantic models
│   └── websocket_handler.py   # WebSocket connections
│
├── agents/                     # Multi-agent hierarchy
│   ├── __init__.py
│   ├── coordinator.py         # Master coordinator agent
│   ├── specialist.py          # Specialist agent types
│   └── worker.py              # Task execution agents
│
├── tools/                      # Tool system
│   ├── __init__.py
│   ├── base_tool.py           # Abstract tool interface
│   ├── registry.py            # Tool registration
│   ├── builtin/               # Built-in tools
│   │   ├── web_search.py
│   │   ├── code_executor.py
│   │   ├── file_operations.py
│   │   ├── calculator.py
│   │   └── api_caller.py
│   └── custom/                # User custom tools
│
├── memory/                     # Memory system
│   ├── __init__.py
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── memory_manager.py      # Memory operations
│   ├── embeddings.py          # Text embedding generation
│   └── retrieval.py           # Context retrieval
│
├── extensions/                 # Modular extensions
│   ├── __init__.py
│   ├── base_extension.py      # Extension interface
│   └── examples/
│       └── weather_extension.py
│
├── instruments/                # Custom scripts
│   ├── __init__.py
│   └── examples/
│       └── data_analysis.py
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment.md
│   └── development_guide.md
│
├── docker/                     # Docker configs
│   ├── app.dockerfile
│   └── nginx.conf
│
├── tests/                      # Test suite
│   ├── test_agents.py
│   ├── test_gateway.py
│   ├── test_memory.py
│   └── test_integration.py
│
├── config.json                 # System configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Main Docker image
├── docker-compose.yml          # Container orchestration
├── run.py                      # Entry point
├── setup.sh                    # Setup script
├── .env.example                # Environment template
└── README.md                   # Documentation
```

---

## Implementation Requirements

### Phase 1: Core Infrastructure

#### 1.1 Agent Zero Core Engine

**File: `core/agent.py`**

Create a base Agent class with:
- Unique agent ID generation
- LLM provider integration (OpenAI, Anthropic, local models)
- Conversation history management
- Tool calling capabilities
- Parent-child agent relationships
- Context awareness (memory, tools, permissions)
- Async message processing

**Key Methods:**
```python
class Agent:
    async def process_message(self, message: str, context: dict) -> str
    async def spawn_child_agent(self, task: str, specialist_type: str) -> Agent
    async def execute_tool(self, tool_name: str, params: dict) -> Any
    async def retrieve_memory(self, query: str, limit: int) -> list
    async def store_memory(self, content: str, metadata: dict) -> None
    def _build_system_prompt(self) -> str
    def _format_tools_for_llm(self) -> str
```

**Agent Types:**
- **Coordinator Agent**: Master agent that delegates tasks
- **Specialist Agents**: Domain-specific (coding, research, data analysis, creative)
- **Worker Agents**: Execute specific subtasks

**File: `core/agent_manager.py`**

Implement agent lifecycle management:
- Agent pool management
- Agent creation and destruction
- Resource allocation
- Performance monitoring

**File: `core/task_planner.py`**

Implement intelligent task decomposition:
- Parse complex requests into subtasks
- Determine optimal agent types for each subtask
- Create execution DAG (Directed Acyclic Graph)
- Handle dependencies between tasks

#### 1.2 Memory System

**File: `memory/vector_store.py`**

Implement ChromaDB integration:
```python
class VectorStore:
    def __init__(self, persist_directory: str)
    async def add_documents(self, texts: list, metadatas: list, ids: list)
    async def query(self, query_text: str, n_results: int, filter: dict) -> list
    async def delete(self, ids: list)
    async def update(self, ids: list, documents: list, metadatas: list)
```

**File: `memory/memory_manager.py`**

Implement memory operations:
- Conversation history storage
- Knowledge base management
- User preference storage
- Context-aware retrieval
- Memory prioritization (recency, relevance, importance)

**Memory Schema:**
```python
{
    "id": "uuid",
    "user_id": "string",
    "conversation_id": "string",
    "content": "string",
    "embedding": [float],  # Vector representation
    "metadata": {
        "timestamp": "datetime",
        "type": "message|knowledge|preference",
        "platform": "string",
        "importance": float,  # 0-1 score
        "tags": ["string"]
    }
}
```

#### 1.3 Tool System

**File: `tools/base_tool.py`**

Define abstract tool interface:
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class BaseTool(ABC):
    name: str
    description: str
    parameters: BaseModel  # Pydantic schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        pass
    
    def get_schema(self) -> dict:
        """Return JSON schema for LLM tool calling"""
        pass
```

**File: `tools/registry.py`**

Implement tool registration system:
```python
class ToolRegistry:
    def __init__(self)
    def register_tool(self, tool: BaseTool) -> None
    def get_tool(self, name: str) -> BaseTool
    def list_tools(self) -> list[str]
    def get_tools_schema(self) -> list[dict]  # For LLM
```

**Built-in Tools to Implement:**

1. **Web Search** (`tools/builtin/web_search.py`)
   - Integration with search APIs (DuckDuckGo, Brave, Google)
   - Result parsing and summarization

2. **Code Executor** (`tools/builtin/code_executor.py`)
   - Sandboxed Python code execution
   - Timeout and resource limits
   - Output capture

3. **File Operations** (`tools/builtin/file_operations.py`)
   - Read/write files
   - Directory operations
   - Permission management

4. **Calculator** (`tools/builtin/calculator.py`)
   - Mathematical expression evaluation
   - Unit conversion

5. **API Caller** (`tools/builtin/api_caller.py`)
   - Generic HTTP request wrapper
   - Authentication support
   - Response parsing

### Phase 2: Platform Gateway (OpenClaw)

#### 2.1 Base Platform Interface

**File: `gateway/base_platform.py`**

```python
from abc import ABC, abstractmethod

class BasePlatform(ABC):
    platform_name: str
    
    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize platform connection"""
        pass
    
    @abstractmethod
    async def send_message(self, chat_id: str, message: str, metadata: dict) -> None:
        """Send message to platform"""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncGenerator:
        """Async generator yielding incoming messages"""
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str) -> dict:
        """Get user profile information"""
        pass
    
    def normalize_message(self, raw_message: Any) -> dict:
        """Convert platform-specific format to standard format"""
        return {
            "message_id": "string",
            "user_id": "string",
            "chat_id": "string",
            "text": "string",
            "timestamp": "datetime",
            "platform": self.platform_name,
            "metadata": {}
        }
```

#### 2.2 Platform Implementations

**Telegram** (`gateway/platforms/telegram.py`)
- Use `python-telegram-bot` library
- Implement webhook or polling
- Handle commands, messages, inline queries
- Support file uploads/downloads

**Discord** (`gateway/platforms/discord.py`)
- Use `discord.py` library
- Bot commands and slash commands
- Channel and DM support
- Role-based permissions

**Slack** (`gateway/platforms/slack.py`)
- Use `slack-sdk` library
- Event subscription handling
- Thread support
- Interactive components

**WebChat** (`gateway/platforms/webchat.py`)
- WebSocket-based real-time chat
- HTTP fallback support
- Session management
- Authentication integration

**WhatsApp** (`gateway/platforms/whatsapp.py`)
- WhatsApp Business API integration
- Message templates
- Media handling

**Google Chat** (`gateway/platforms/google_chat.py`)
- Google Chat API integration
- Card messages
- Space management

**Signal** (`gateway/platforms/signal.py`)
- Signal-CLI integration
- Group message support

**Microsoft Teams** (`gateway/platforms/teams.py`)
- Bot Framework SDK
- Adaptive cards
- Teams-specific features

#### 2.3 Session Management

**File: `gateway/session_manager.py`**

```python
class SessionManager:
    def __init__(self, memory_manager: MemoryManager)
    
    async def get_session(self, user_id: str, platform: str) -> Session
    async def create_session(self, user_id: str, platform: str) -> Session
    async def end_session(self, session_id: str) -> None
    async def get_conversation_history(self, session_id: str, limit: int) -> list
```

**Session Schema:**
```python
{
    "session_id": "uuid",
    "user_id": "string",
    "platform": "string",
    "created_at": "datetime",
    "last_activity": "datetime",
    "context": {
        "current_task": "string",
        "active_agents": ["agent_id"],
        "variables": {}
    }
}
```

### Phase 3: Integration Bridge

#### 3.1 FastAPI Server

**File: `integration/api_server.py`**

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nexus Agent API")

# Endpoints
@app.post("/api/v1/message")
async def process_message(request: MessageRequest) -> MessageResponse:
    """Process incoming message from any platform"""
    pass

@app.post("/api/v1/agent")
async def spawn_agent(request: AgentRequest) -> AgentResponse:
    """Manually spawn an agent for specific task"""
    pass

@app.get("/api/v1/agent/{agent_id}")
async def get_agent_status(agent_id: str) -> AgentStatus:
    """Get agent status and conversation history"""
    pass

@app.post("/api/v1/task")
async def create_task(request: TaskRequest) -> TaskResponse:
    """Create and execute a task"""
    pass

@app.get("/api/v1/memory/search")
async def search_memory(query: str, user_id: str, limit: int) -> list:
    """Search user's memory"""
    pass

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time communication"""
    pass
```

**File: `integration/schemas.py`**

Define Pydantic models for all API requests/responses.

#### 3.2 Integration Bridge

**File: `integration/bridge.py`**

```python
class IntegrationBridge:
    def __init__(self, agent_manager, session_manager, platform_gateway)
    
    async def handle_platform_message(self, message: dict) -> None:
        """
        Main message handler:
        1. Normalize message from platform format
        2. Retrieve or create session
        3. Load conversation context from memory
        4. Route to appropriate agent
        5. Process through agent hierarchy
        6. Store response in memory
        7. Send formatted response to platform
        """
        pass
    
    async def route_to_agent(self, message: dict, context: dict) -> str:
        """Determine which agent should handle message"""
        pass
    
    def transform_message(self, message: dict, target_format: str) -> dict:
        """Transform between OpenClaw and Agent Zero formats"""
        pass
```

### Phase 4: Extension System

**File: `extensions/base_extension.py`**

```python
class BaseExtension(ABC):
    name: str
    version: str
    dependencies: list[str]
    
    @abstractmethod
    async def initialize(self, config: dict) -> None:
        pass
    
    @abstractmethod
    async def on_message(self, message: dict) -> dict:
        """Hook into message processing"""
        pass
    
    @abstractmethod
    async def on_agent_spawn(self, agent: Agent) -> None:
        """Hook into agent creation"""
        pass
    
    def register_tools(self) -> list[BaseTool]:
        """Register extension-specific tools"""
        return []
```

### Phase 5: Configuration & Deployment

#### 5.1 Configuration System

**File: `config.json`**

Already defined in your repository. Ensure runtime loading:

```python
import json
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)
    
    def get(self, key_path: str, default=None):
        """Get nested config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return default
        return value
```

**File: `.env.example`**

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Platform API Keys
TELEGRAM_BOT_TOKEN=
DISCORD_BOT_TOKEN=
SLACK_BOT_TOKEN=
WHATSAPP_API_KEY=
GOOGLE_CHAT_CREDENTIALS=
SIGNAL_CLI_PATH=
TEAMS_APP_ID=
TEAMS_APP_PASSWORD=

# Database
CHROMA_PERSIST_DIR=./data/chroma

# API Server
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4

# Security
JWT_SECRET=
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/nexus-agent.log
```

#### 5.2 Docker Configuration

**File: `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/chroma /app/logs

# Expose API port
EXPOSE 8080

# Run application
CMD ["python", "run.py", "--mode", "server"]
```

**File: `docker-compose.yml`**

```yaml
version: '3.8'

services:
  nexus-agent:
    build: .
    container_name: nexus-agent
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    networks:
      - nexus-network

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: nexus-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - nexus-agent
    networks:
      - nexus-network

networks:
  nexus-network:
    driver: bridge

volumes:
  chroma-data:
  app-logs:
```

#### 5.3 Entry Point

**File: `run.py`**

```python
import asyncio
import argparse
from loguru import logger
from integration.api_server import app
from integration.bridge import IntegrationBridge
from gateway.message_router import MessageRouter
from core.agent_manager import AgentManager
from memory.memory_manager import MemoryManager
import uvicorn

def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Agent")
    parser.add_argument(
        "--mode",
        choices=["server", "bridge", "both"],
        default="both",
        help="Run mode: API server, bridge, or both"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8080, help="API port")
    parser.add_argument("--config", default="config.json", help="Config file")
    return parser.parse_args()

async def start_bridge():
    """Start integration bridge for platform connections"""
    logger.info("Starting Integration Bridge...")
    
    # Initialize components
    memory_manager = MemoryManager()
    agent_manager = AgentManager()
    message_router = MessageRouter()
    bridge = IntegrationBridge(agent_manager, memory_manager, message_router)
    
    # Start platform connections
    await bridge.start()
    
    logger.info("Integration Bridge running")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge...")
        await bridge.stop()

def start_server(host: str, port: int):
    """Start FastAPI server"""
    logger.info(f"Starting API Server on {host}:{port}...")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

async def start_both(host: str, port: int):
    """Start both server and bridge"""
    server_task = asyncio.create_task(
        asyncio.to_thread(start_server, host, port)
    )
    bridge_task = asyncio.create_task(start_bridge())
    
    await asyncio.gather(server_task, bridge_task)

def main():
    args = parse_args()
    
    # Configure logging
    logger.add(
        "logs/nexus-agent.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info("Starting Nexus Agent...")
    logger.info(f"Mode: {args.mode}")
    
    if args.mode == "server":
        start_server(args.host, args.port)
    elif args.mode == "bridge":
        asyncio.run(start_bridge())
    elif args.mode == "both":
        asyncio.run(start_both(args.host, args.port))

if __name__ == "__main__":
    main()
```

---

## Implementation Guidelines

### Code Quality Standards

1. **Type Hints**: Use Python type hints throughout
2. **Async/Await**: Prefer async operations for I/O
3. **Error Handling**: Comprehensive try-except blocks with logging
4. **Documentation**: Docstrings for all classes and functions
5. **Testing**: Unit tests for core functionality
6. **Logging**: Use loguru for structured logging
7. **Configuration**: Environment-based configuration
8. **Security**: API key management, input validation, rate limiting

### Agent System Prompts

**Coordinator Agent System Prompt:**
```
You are the Coordinator Agent in a multi-agent system. Your role is to:
1. Understand user requests and break them into subtasks
2. Delegate subtasks to specialist agents
3. Synthesize results from multiple agents
4. Maintain conversation context
5. Use available tools when needed

Available specialist agents:
- Coding Agent: Programming, debugging, code review
- Research Agent: Web search, information gathering
- Data Agent: Data analysis, visualization, statistics
- Creative Agent: Writing, brainstorming, design ideas

You can spawn specialist agents using spawn_agent() function.
You have access to memory for context retrieval.

Always think step-by-step and explain your reasoning.
```

**Specialist Agent System Prompt Template:**
```
You are a {specialist_type} Agent. Your expertise is in {domain}.

Your capabilities:
{capabilities_list}

Available tools:
{tools_schema}

You are working on a subtask delegated by the Coordinator Agent.
Task: {task_description}

Context from parent agent:
{parent_context}

Execute the task efficiently and return results in a structured format.
```

### Memory Retrieval Strategy

```python
async def retrieve_relevant_context(
    query: str,
    user_id: str,
    conversation_id: str,
    memory_manager: MemoryManager
) -> str:
    """
    Retrieve relevant context from memory:
    1. Recent conversation history (last 10 messages)
    2. Semantic search for relevant knowledge (top 5)
    3. User preferences
    4. Current task context
    """
    
    # Recent history
    recent = await memory_manager.get_recent_history(
        conversation_id=conversation_id,
        limit=10
    )
    
    # Semantic search
    relevant = await memory_manager.semantic_search(
        query=query,
        user_id=user_id,
        limit=5
    )
    
    # User preferences
    prefs = await memory_manager.get_user_preferences(user_id)
    
    # Format context
    context = f"""
    Recent Conversation:
    {format_messages(recent)}
    
    Relevant Knowledge:
    {format_knowledge(relevant)}
    
    User Preferences:
    {format_preferences(prefs)}
    """
    
    return context
```

### Tool Calling Implementation

Implement function calling for both OpenAI and Anthropic:

```python
async def call_llm_with_tools(
    messages: list,
    tools: list[dict],
    model: str = "gpt-4-turbo-preview"
) -> dict:
    """
    Call LLM with tool support.
    Handle tool calls iteratively until completion.
    """
    
    while True:
        response = await llm_client.create_completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Check if LLM wants to call a tool
        if response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                # Execute tool
                tool_result = await execute_tool(tool_name, tool_args)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })
        else:
            # No more tool calls, return final response
            return response
```

---

## Testing Strategy

### Unit Tests

**File: `tests/test_agents.py`**
- Test agent creation and lifecycle
- Test agent communication
- Test task delegation
- Test tool execution

**File: `tests/test_memory.py`**
- Test memory storage and retrieval
- Test semantic search
- Test context management

**File: `tests/test_gateway.py`**
- Test message normalization
- Test platform-specific handling
- Test session management

**File: `tests/test_integration.py`**
- End-to-end message processing
- Test API endpoints
- Test WebSocket connections

### Integration Tests

1. **Multi-Agent Workflow**: Test coordinator → specialist → worker flow
2. **Cross-Platform**: Test message routing across different platforms
3. **Memory Persistence**: Test long-term memory across sessions
4. **Tool Execution**: Test all built-in tools

---

## Deployment Checklist

### Pre-Deployment

- [ ] All environment variables configured
- [ ] API keys secured
- [ ] Database initialized
- [ ] SSL certificates configured (production)
- [ ] Rate limiting configured
- [ ] Logging configured
- [ ] Backup strategy defined

### Deployment Steps

1. **Build Docker Image**
   ```bash
   docker-compose build
   ```

2. **Run Setup Script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Verify Health**
   ```bash
   curl http://localhost:8080/health
   ```

5. **Test Platform Connections**
   - Send test message to each platform
   - Verify agent responses
   - Check logs for errors

### Monitoring

- Set up health check endpoint
- Monitor agent performance metrics
- Track memory usage
- Log aggregation
- Error alerting

---

## Development Workflow

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/dicacid/nexus-agent.git
cd nexus-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run setup
python setup.py

# Run tests
pytest tests/

# Start development server
python run.py --mode both
```

### Git Workflow

1. Create feature branch: `git checkout -b feature/agent-memory`
2. Implement feature with tests
3. Run tests: `pytest tests/`
4. Format code: `black . && flake8`
5. Commit: `git commit -m "feat: implement agent memory system"`
6. Push and create PR

---

## Extension Development Guide

To create a custom extension:

1. **Create Extension File**: `extensions/my_extension.py`

```python
from extensions.base_extension import BaseExtension
from tools.base_tool import BaseTool

class MyExtension(BaseExtension):
    name = "my_extension"
    version = "1.0.0"
    
    async def initialize(self, config: dict):
        self.config = config
        logger.info(f"Initialized {self.name}")
    
    async def on_message(self, message: dict) -> dict:
        # Process message
        return message
    
    def register_tools(self) -> list[BaseTool]:
        return [MyCustomTool()]
```

2. **Register Extension**: In `config.json`

```json
{
  "extensions": [
    {
      "name": "my_extension",
      "enabled": true,
      "config": {}
    }
  ]
}
```

---

## Security Considerations

1. **API Key Management**
   - Use environment variables
   - Never commit keys to git
   - Rotate keys regularly

2. **Input Validation**
   - Sanitize all user inputs
   - Validate message formats
   - Limit message size

3. **Rate Limiting**
   - Per-user rate limits
   - Per-platform rate limits
   - Global rate limits

4. **Sandboxing**
   - Isolate code execution
   - Resource limits for agents
   - Timeout mechanisms

5. **Authentication**
   - JWT tokens for API
   - Platform-specific auth
   - User verification

---

## Performance Optimization

1. **Caching**
   - Cache frequent memory queries
   - Cache tool results
   - Cache LLM responses for identical requests

2. **Async Operations**
   - Parallel agent execution
   - Async tool calls
   - Non-blocking I/O

3. **Resource Management**
   - Agent pool limits
   - Memory cleanup
   - Connection pooling

4. **Database Optimization**
   - Index frequently queried fields
   - Batch operations
   - Query optimization

---

## Troubleshooting Guide

### Common Issues

**Issue: Agent not responding**
- Check LLM API keys
- Verify network connectivity
- Check rate limits
- Review logs for errors

**Issue: Memory not persisting**
- Verify ChromaDB configuration
- Check write permissions
- Ensure database directory exists

**Issue: Platform connection failed**
- Validate platform credentials
- Check webhook configuration
- Verify network accessibility

**Issue: High latency**
- Check LLM response times
- Optimize memory queries
- Review agent hierarchy depth
- Consider caching strategies

---

## Future Enhancements

### Phase 6: Advanced Features

1. **Multi-Modal Support**
   - Image understanding (GPT-4 Vision)
   - Audio processing
   - Video analysis
   - Document parsing

2. **Advanced Memory**
   - Graph-based knowledge representation
   - Automatic knowledge extraction
   - Memory consolidation
   - Forgetting mechanisms

3. **Agent Learning**
   - Reinforcement learning from feedback
   - Performance optimization
   - Personalization

4. **Workflow Automation**
   - Recurring task scheduling
   - Trigger-based actions
   - Integration with automation tools (Zapier, IFTTT)

5. **Analytics Dashboard**
   - Usage statistics
   - Agent performance metrics
   - User engagement analytics
   - Cost tracking

6. **Voice Interface**
   - Speech-to-text
   - Text-to-speech
   - Voice commands

7. **Mobile App**
   - Native iOS/Android apps
   - Push notifications
   - Offline mode

---

## Success Criteria

The implementation is complete when:

✅ **Core Functionality**
- [ ] Multi-agent hierarchy working
- [ ] Task decomposition functioning
- [ ] Memory persistence operational
- [ ] All built-in tools working

✅ **Platform Integration**
- [ ] At least 3 platforms connected
- [ ] Message normalization working
- [ ] Session management functional

✅ **API**
- [ ] All endpoints documented and working
- [ ] WebSocket connections stable
- [ ] Authentication implemented

✅ **Testing**
- [ ] Unit test coverage > 80%
- [ ] Integration tests passing
- [ ] Performance benchmarks met

✅ **Documentation**
- [ ] README comprehensive
- [ ] API documentation complete
- [ ] Deployment guide clear
- [ ] Development guide helpful

✅ **Deployment**
- [ ] Docker images building
- [ ] Docker Compose working
- [ ] Production deployment successful

---

## Support & Resources

### Reference Projects

- **Agent Zero**: https://github.com/agent0ai/agent-zero
- **OpenClaw**: https://github.com/openclaw/openclaw
- **LangChain**: https://github.com/langchain-ai/langchain
- **AutoGPT**: https://github.com/Significant-Gravitas/AutoGPT

### Documentation

- **FastAPI**: https://fastapi.tiangolo.com/
- **ChromaDB**: https://docs.trychroma.com/
- **Python Telegram Bot**: https://python-telegram-bot.org/
- **Discord.py**: https://discordpy.readthedocs.io/
- **Slack SDK**: https://slack.dev/python-slack-sdk/

### LLM Provider Docs

- **OpenAI API**: https://platform.openai.com/docs/
- **Anthropic API**: https://docs.anthropic.com/
- **Ollama** (local LLM): https://ollama.ai/

---

## Final Notes

**Build incrementally:**
1. Start with core agent system
2. Add memory functionality
3. Implement one platform integration
4. Build API server
5. Add more platforms
6. Implement extensions
7. Optimize and scale

**Maintain quality:**
- Write tests as you build
- Document as you code
- Refactor regularly
- Review security practices

**Stay focused:**
- Implement MVP first
- Add advanced features later
- Prioritize stability over features
- Listen to user feedback

This is an ambitious project that combines cutting-edge AI capabilities with practical multi-platform connectivity. Take it step by step, and you'll build something remarkable.

**Good luck building Nexus Agent! 🚀**