
#!/usr/bin/env python3
"""
Nexus Agent Integration Bridge
Connects OpenClaw platform gateway with Agent Zero core engine
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_agent")

@dataclass
class Message:
    """Represents a message in the hybrid system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    platform: str = ""
    sender: str = ""
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Task:
    """Represents a task in the hybrid system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: int = 1
    status: str = "pending"
    assigned_agent: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class IntegrationBridge:
    """Main integration bridge between OpenClaw and Agent Zero"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.platforms = {}
        self.agents = {}
        self.tasks = []
        self.message_queue = asyncio.Queue()
        self.memory_store = {}

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        # Try absolute path first, then relative to script directory, then relative to CWD
        paths_to_try = [
            path,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
            os.path.join(os.getcwd(), "config.json"),
            "config.json"
        ]

        for p in paths_to_try:
            try:
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        logger.info(f"Loaded config from {p}")
                        return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load config from {p}: {e}")

        logger.warning("Config file not found, using defaults")
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "api": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "platforms": ["telegram", "whatsapp"],
            "llm_providers": ["anthropic", "openai"]
        }

    async def receive_message(self, platform: str, sender: str, content: str) -> str:
        """Receive message from platform and process it"""
        message = Message(
            platform=platform,
            sender=sender,
            content=content
        )

        # Add to queue for processing
        await self.message_queue.put(message)

        # Process message and get response
        response = await self._process_message(message)
        return response

    async def _process_message(self, message: Message) -> str:
        """Process incoming message and generate response"""
        logger.info(f"Processing message from {message.sender} on {message.platform}")

        # Route to appropriate agent
        agent_response = await self._route_to_agent(message)

        # Store in memory
        self._store_memory(message, agent_response)

        return agent_response

    async def _route_to_agent(self, message: Message) -> str:
        """Route message to appropriate agent for processing"""
        # For now, use a simple routing mechanism
        # In production, this would use Agent Zero's agent hierarchy

        # Extract intent from message
        intent = self._extract_intent(message.content)

        # Route based on intent
        if intent == "task":
            return await self._handle_task(message)
        elif intent == "query":
            return await self._handle_query(message)
        else:
            return await self._handle_general(message)

    def _extract_intent(self, content: str) -> str:
        """Extract intent from message content"""
        content_lower = content.lower()

        if any(word in content_lower for word in ["find", "search", "look up", "research"]):
            return "query"
        elif any(word in content_lower for word in ["do", "execute", "run", "create", "build"]):
            return "task"
        else:
            return "general"

    async def _handle_task(self, message: Message) -> str:
        """Handle task execution request"""
        task = Task(
            description=message.content,
            priority=1,
            status="pending"
        )

        # Add to task queue
        self.tasks.append(task)

        # Process task (in production, this would use Agent Zero's task execution)
        result = f"Task '{task.description}' has been queued for execution."

        return result

    async def _handle_query(self, message: Message) -> str:
        """Handle query request"""
        # In production, this would use Agent Zero's search and knowledge capabilities
        result = f"I'll search for information about: {message.content}"

        # Store query in memory for future reference
        self._store_memory(message, result, query=True)

        return result

    async def _handle_general(self, message: Message) -> str:
        """Handle general conversation"""
        # In production, this would use Agent Zero's conversation capabilities
        result = f"I received: '{message.content}'. How can I help you with that?"

        return result

    def _store_memory(self, message: Message, response: str, query: bool = False):
        """Store interaction in memory"""
        memory_id = str(uuid.uuid4())
        self.memory_store[memory_id] = {
            "message": message.to_dict(),
            "response": response,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Stored memory entry {memory_id}")

    def get_memory(self, query: str = None) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        if query is None:
            return list(self.memory_store.values())

        # Simple keyword matching (in production, use vector search)
        results = []
        query_lower = query.lower()

        for memory in self.memory_store.values():
            if (query_lower in memory.get("message", {}).get("content", "").lower() or
                query_lower in memory.get("response", "").lower()):
                results.append(memory)

        return results

    async def run(self):
        """Run the integration bridge"""
        logger.info("Starting Nexus Agent Integration Bridge...")

        # In production, this would start the API server and platform connections
        while True:
            try:
                message = await self.message_queue.get()
                response = await self._process_message(message)
                logger.info(f"Response: {response}")
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message: {e}")


async def main():
    """Main entry point"""
    bridge = IntegrationBridge()
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
