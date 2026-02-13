#!/bin/bash

# Nexus Agent Setup Script
# This script sets up the hybrid agent system for local development

echo "========================================"
echo "Nexus Agent Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Python version:$(python3 --version)${NC}"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}pip version:$(pip3 --version)${NC}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p data logs memory knowledge

# Set permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chmod +x run.py setup.sh

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOL
# Hybrid Agent Configuration
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8080

# LLM Provider Configuration
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here

# Platform Configuration
PLATFORMS=telegram,whatsapp,slack,discord
EOL
    echo -e "${GREEN}.env file created. Please edit with your configuration.${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the Hybrid Agent API Server, run:"
echo "  python run.py --mode server"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "For more information, see README.md"
echo ""
