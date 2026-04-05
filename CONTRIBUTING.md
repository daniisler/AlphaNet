# Contributing Guide

Thank you for your interest in AlphaNet! We welcome all forms of contributions.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature suggestion, please submit an issue on GitHub:

1. Use a clear title describing the issue
2. Provide reproduction steps
3. Include environment details (OS, Python version, CUDA version, etc.)
4. Attach error logs if applicable

### Submitting Code

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to functions and classes
- Keep code clean and readable
- Add comments for complex logic

### Commit Message Convention

- Use clear English descriptions
- Format: `<type>: <description>`
- Types: feat(new feature), fix(bug fix), docs(documentation), style(formatting), refactor, test, chore(build)

Examples:
```
feat: add multi-GPU training support
fix: fix numerical stability in ZBL potential calculation
docs: update installation instructions in README
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/zmyybc/AlphaNet.git
cd AlphaNet

# Create development environment
conda create -n alphanet-dev python=3.9
conda activate alphanet-dev

# Install dev dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/
```

## Contact

For any questions, please reach out via GitHub Issues or Discussions.
