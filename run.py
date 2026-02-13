#!/usr/bin/env python3
"""
Nexus Agent - Main Entry Point
Combining OpenClaw multi-platform connectivity with Agent Zero multi-agent cooperation
"""

import sys
import os
import argparse
import logging

# Add integration directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'integration'))

from bridge import IntegrationBridge
from api_server import app, bridge
import uvicorn

def setup_logging(level="INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_server(host="0.0.0.0", port=8080):
    """Run the API server"""
    logging.info(f"Starting Hybrid Agent API Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

def run_bridge():
    """Run the integration bridge"""
    logging.info("Starting Hybrid Agent Integration Bridge...")
    import asyncio
    asyncio.run(bridge.run())

def main():
    parser = argparse.ArgumentParser(
        description="Nexus Agent - Combining OpenClaw and Agent Zero"
    )
    
    parser.add_argument(
        "--mode",
        choices=["server", "bridge", "both"],
        default="server",
        help="Run mode: server (API only), bridge (integration only), or both"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.mode == "server":
        run_server(args.host, args.port)
    elif args.mode == "bridge":
        run_bridge()
    elif args.mode == "both":
        # In production, this would run both concurrently
        logging.info("Running both server and bridge...")
        run_server(args.host, args.port)

if __name__ == "__main__":
    main()
