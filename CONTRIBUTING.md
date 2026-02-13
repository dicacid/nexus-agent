# Contributing to Nexus Agent

Thank you for your interest in contributing to the Nexus Agent! This document provides guidelines and instructions for contributing to this open-source project.

## 🎯 Project Goals

The Nexus Agent combines the best features of OpenClaw and Agent Zero to create a powerful, local-first AI assistant. Our goals include:

- Maintaining local execution for privacy and control
- Supporting multiple messaging platforms
- Enabling multi-agent cooperation
- Providing persistent memory and knowledge retention
- Creating an extensible framework for custom tools and extensions

## 🚀 Getting Started

1. **Fork the repository**
2. **Create a branch** for your feature or bug fix
3. **Make your changes** following our coding standards
4. **Test your changes** thoroughly
5. **Submit a pull request** with a clear description

## 📝 Coding Standards

- Follow Python best practices (PEP 8)
- Use type hints where possible
- Write clear, descriptive commit messages
- Add documentation for new features
- Include tests for new functionality

## 🧰 Development Environment

Use separate runtime and development dependency sets:

```bash
pip install -r requirements-dev.txt
```

## 🧪 Testing

All contributions should include appropriate tests:

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 📚 Documentation

- Update the README.md if you add new features
- Document any new APIs or configuration options
- Add examples for new functionality

## 🤝 Code Review

All pull requests will be reviewed by the maintainers. We may suggest changes or improvements before merging.

## 📬 Community

Join our community to discuss ideas, ask questions, and collaborate:

- [Discord Server](https://discord.gg/nexus-agent)
- [GitHub Discussions](https://github.com/yourusername/nexus-agent/discussions)
- [Issue Tracker](https://github.com/yourusername/nexus-agent/issues)

## 🙏 Thank You

Your contributions help make this project better for everyone. Thank you for your time and effort!
