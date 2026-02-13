
#!/usr/bin/env python3
"""
Nexus Agent API Server
FastAPI-based server for the hybrid autonomous AI agent system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_agent_api")

# Import integration bridge
import os
import sys
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.bridge import IntegrationBridge, Message, Task

# Create FastAPI app
app = FastAPI(
    title="Nexus Agent API",
    description="API for the hybrid agent system combining OpenClaw and Agent Zero",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize integration bridge
bridge = IntegrationBridge()

# Request/Response models
class MessageRequest(BaseModel):
    platform: str
    sender: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    success: bool
    message: str
    response: Optional[str] = None
    task_id: Optional[str] = None

class TaskRequest(BaseModel):
    description: str
    priority: Optional[int] = 1

class TaskResponse(BaseModel):
    success: bool
    task_id: str
    status: str

class MemoryQueryRequest(BaseModel):
    query: Optional[str] = None
    limit: Optional[int] = 10

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Hybrid Autonomous AI Agent",
        "version": "0.1.0",
        "description": "Combining OpenClaw multi-platform connectivity with Agent Zero multi-agent cooperation",
        "endpoints": {
            "message": "/api/v1/message",
            "agent": "/api/v1/agent",
            "task": "/api/v1/task",
            "memory": "/api/v1/memory",
            "status": "/api/v1/status"
        }
    }

@app.get("/api/v1/status")
async def status():
    """System status endpoint"""
    return {
        "status": "running",
        "version": "0.1.0",
        "platforms_connected": len(bridge.platforms),
        "agents_active": len(bridge.agents),
        "tasks_pending": len([t for t in bridge.tasks if t.status == "pending"]),
        "memory_entries": len(bridge.memory_store)
    }

@app.post("/api/v1/message", response_model=MessageResponse)
async def receive_message(request: MessageRequest, background_tasks: BackgroundTasks):
    """Receive message from platform"""
    try:
        message = Message(
            platform=request.platform,
            sender=request.sender,
            content=request.content,
            metadata=request.metadata or {}
        )

        # Process message in background
        background_tasks.add_task(bridge.message_queue.put, message)

        # Get response (in production, this would be async)
        response = await bridge._process_message(message)

        return MessageResponse(
            success=True,
            message="Message received and processed",
            response=response
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/task", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Create a new task"""
    try:
        task = Task(
            description=request.description,
            priority=request.priority,
            status="pending"
        )

        # Add to task queue
        bridge.tasks.append(task)

        return TaskResponse(
            success=True,
            task_id=task.id,
            status=task.status
        )
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/task", response_model=List[Task])
async def list_tasks(status: Optional[str] = None):
    """List tasks"""
    try:
        if status:
            return [t for t in bridge.tasks if t.status == status]
        return bridge.tasks
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/memory")
async def get_memory(query: Optional[str] = None, limit: int = 10):
    """Retrieve memories"""
    try:
        memories = bridge.get_memory(query)
        return {
            "success": True,
            "count": len(memories),
            "memories": memories[:limit]
        }
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/memory")
async def store_memory(content: str, query: Optional[str] = None):
    """Store new memory"""
    try:
        message = Message(
            platform="api",
            sender="system",
            content=content
        )
        response = f"Memory stored: {content}"
        bridge._store_memory(message, response, query=query is not None)

        return {
            "success": True,
            "message": "Memory stored successfully"
        }
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agent")
async def list_agents():
    """List active agents"""
    return {
        "success": True,
        "agents": list(bridge.agents.keys()),
        "count": len(bridge.agents)
    }

if __name__ == "__main__":
    config = bridge.config
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8080)

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
