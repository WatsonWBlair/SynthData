"""Invoke tasks for SynthData project management."""

from invoke import task
import os


@task
def install(c):
    """Install project dependencies."""
    print("Installing dependencies...")
    c.run("pip install -r requirements.txt")
    print("Dependencies installed successfully!")


@task
def generate_fraud(c, transactions=216960, fraud_rate=0.02, seed=42):
    """Generate synthetic financial fraud dataset.
    
    Args:
        transactions: Number of transactions to generate (default: 216960)
        fraud_rate: Percentage of fraudulent transactions (default: 0.02)
        seed: Random seed for reproducibility (default: 42)
    """
    print(f"Generating {transactions} transactions with {fraud_rate*100}% fraud rate...")
    os.chdir("Fin_Fraud")
    c.run(f"python enhanced_data_generator.py --transactions {transactions} --fraud-rate {fraud_rate} --seed {seed}")
    os.chdir("..")
    print("Dataset generated successfully!")


@task
def generate_student_datasets(c, count=10, transactions=50000, fraud_rate=0.02, seed=42):
    """Generate multiple unique datasets for different student teams.
    
    Args:
        count: Number of unique datasets to generate (default: 10)
        transactions: Number of transactions per dataset (default: 50000)
        fraud_rate: Percentage of fraudulent transactions (default: 0.02)
        seed: Base random seed (each dataset uses seed + index) (default: 42)
    """
    print(f"Generating {count} unique student datasets...")
    print(f"Each dataset: {transactions} transactions, {fraud_rate*100}% fraud rate")
    os.chdir("Fin_Fraud")
    c.run(f"python enhanced_data_generator.py --student-datasets {count} --transactions {transactions} --fraud-rate {fraud_rate} --seed {seed}")
    os.chdir("..")
    print(f"\n✅ Generated {count} unique datasets for student teams!")
    print("📁 Check Fin_Fraud/data/student_datasets/ for files")


@task
def validate(c):
    """Validate the generated synthetic data."""
    print("Validating synthetic data...")
    os.chdir("Fin_Fraud")
    c.run("python validate_enhanced_data.py")
    os.chdir("..")


@task
def notebook(c):
    """Launch Jupyter notebook for interactive data generation."""
    print("Launching Jupyter notebook...")
    os.chdir("Fin_Fraud")
    c.run("jupyter notebook generate_enhanced.ipynb")
    os.chdir("..")


@task
def test(c):
    """Run all validation tests."""
    print("Running validation tests...")
    validate(c)
    print("All tests passed!")


@task
def clean(c):
    """Clean generated data files."""
    print("Cleaning generated data...")
    c.run("rm -rf Fin_Fraud/data/*.csv")
    print("Cleaned successfully!")


@task
def setup(c):
    """Complete project setup."""
    install(c)
    print("\nSetup complete! Run 'invoke generate-fraud' to create your first dataset.")