# SynthData - Synthetic Data Generation Utilities

Toolkit for generating high-quality synthetic datasets for data science education and research.

## Overview

SynthData provides realistic synthetic data generators for:
- **Education**: Teaching anomaly detection and machine learning
- **Research**: Testing algorithms without real sensitive data
- **Benchmarking**: Standardized datasets for comparison

Currently includes: **Financial Fraud Detection** - Dark pool trading data with injected fraud patterns

## Quick Start

```bash
# Clone repository
git clone https://github.com/WatsonWBlair/SynthData.git
cd SynthData

# Install dependencies
pip install -r requirements.txt

# Setup project (installs all dependencies)
invoke setup

# Generate financial fraud dataset
invoke generate-fraud

# Or with custom parameters
invoke generate-fraud --transactions 100000 --fraud-rate 0.05
```

This creates synthetic financial transaction data with realistic fraud patterns in `Fin_Fraud/data/`

## Available Commands

```bash
invoke --list         # Show all available tasks
invoke setup         # Complete project setup
invoke generate-fraud # Generate synthetic fraud dataset
invoke generate-student-datasets # Generate unique datasets for multiple teams
invoke validate      # Validate generated data quality
invoke notebook      # Launch interactive Jupyter notebook
invoke clean         # Clean generated data files
```

### Generate Multiple Student Datasets
```bash
# Generate 10 unique datasets for student teams
invoke generate-student-datasets --count 10

# Custom parameters
invoke generate-student-datasets --count 20 --transactions 100000 --fraud-rate 0.03
```

## Project Structure

```
SynthData/
├── README.md                 # This file
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── requirements.txt         # Python dependencies
├── tasks.py                # Invoke task definitions
└── Fin_Fraud/              # Financial fraud detection module
    ├── QUICK_START.md      # Getting started guide
    ├── enhanced_data_generator.py  # Core generation engine
    ├── validate_enhanced_data.py   # Data validation tools
    ├── generate_enhanced.ipynb     # Interactive notebook
    └── docs/               # Detailed documentation
        ├── DATA_SPECIFICATION.md   # Data format details
        ├── ASSIGNMENT.md           # Educational assignments
        ├── GRADING.md             # Grading rubrics
        └── SOLUTIONS.md           # Reference solutions
```

## Features

### Financial Fraud Module
- **Benford's Law Compliance**: Mathematically sound generation following natural distributions
- **10 Market Maker Profiles**: HFT, Block Trading, Arbitrage, Institutional patterns
- **5 Fraud Types**: Structuring, Wash Trading, Layering, Market Manipulation, After-Hours
- **Configurable Parameters**: Dataset size, fraud rate, random seed for reproducibility
- **Educational Design**: Fraud patterns intentionally detectable for learning

## Documentation

- [Quick Start Guide](Fin_Fraud/QUICK_START.md) - Get running in minutes
- [Data Specification](Fin_Fraud/docs/DATA_SPECIFICATION.md) - Detailed data format
- [Educational Resources](Fin_Fraud/docs/ASSIGNMENT.md) - Teaching materials

## Requirements

- Python 3.7+
- See [requirements.txt](requirements.txt) for Python packages

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details

## Author

Watson Blair (2025)

## Acknowledgments

Designed for educational purposes in data science and machine learning courses.