# Contributing to SynthData

Thank you for your interest in contributing to SynthData! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/WatsonWBlair/SynthData.git
cd SynthData

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Additional dev tools

# Run tests
invoke test
```

## Workflow

1. **Create an Issue**: Before starting work, create or find an issue describing the change
   - Use issue templates when available
   - Tag appropriately (bug, feature, documentation, etc.)

2. **Create a Branch**: Branch directly from `main`
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/issue-number-description
   ```

3. **Develop**: Make your changes following code style guidelines

4. **Test**: Ensure all tests pass
   ```bash
   invoke test
   invoke validate
   ```

5. **Commit**: Use descriptive commit messages
   ```bash
   git commit -m "feat: add new fraud pattern type (#issue-number)"
   ```

6. **Push**: Push your branch to the repository
   ```bash
   git push origin feature/issue-number-description
   ```

7. **Pull Request**: Create a PR linking to the original issue

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting: `black .`
- Run flake8 for linting: `flake8 .`
- Add type hints where appropriate
- Document all public functions with docstrings

## Testing Requirements

- Write tests for new features
- Maintain >80% code coverage
- Validate synthetic data distributions
- Test edge cases and error handling

## Commit Message Format

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `style:` Code style changes

Include issue number in commits when applicable: `fix: correct Benford's Law calculation (#42)`

## Areas for Contribution

- New synthetic data generators
- Additional fraud patterns
- Performance optimizations
- Documentation improvements
- Test coverage expansion
- Visualization tools

## Questions?

Open an issue or discussion on GitHub.