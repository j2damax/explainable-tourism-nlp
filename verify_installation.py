#!/usr/bin/env python3
"""
Installation verification script for the Aria-Core-ANN project.
This script checks if all required packages are properly installed.
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and importable.
    
    Args:
        package_name: Name of the package for display
        import_name: Name to use for import (if different from package_name)
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import_name = import_name or package_name
        importlib.import_module(import_name)
        return True, f"✅ {package_name}"
    except ImportError as e:
        return False, f"❌ {package_name} - {str(e)}"

def main():
    """Main verification function."""
    print("🔍 Verifying Aria-Core-ANN Project Installation")
    print("=" * 50)
    
    # Core packages to check
    packages_to_check = [
        # Core data science
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        
        # Visualization
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        
        # Jupyter
        ("jupyterlab", "jupyterlab"),
        ("notebook", "notebook"),
        ("ipython", "ipython"),
        
        # Machine learning
        ("scikit-learn", "sklearn"),
        ("nltk", "nltk"),
        ("spacy", "spacy"),
        
        # Deep learning
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        
        # Hugging Face
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("tokenizers", "tokenizers"),
        ("accelerate", "accelerate"),
        
        # Explainable AI
        ("shap", "shap"),
        ("lime", "lime"),
        ("captum", "captum"),
        
        # Text processing
        ("wordcloud", "wordcloud"),
        ("textblob", "textblob"),
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml"),
        
        # Web deployment
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit-shap", "streamlit_shap"),
        
        # API and services
        ("openai", "openai"),
        ("requests", "requests"),
        
        # Data version control
        ("dvc", "dvc"),
        
        # Development tools
        ("ruff", "ruff"),
        ("pytest", "pytest"),
        ("black", "black"),
        ("pre-commit", "pre_commit"),
        
        # Documentation
        ("mkdocs", "mkdocs"),
        ("mkdocs-material", "mkdocs_material"),
        
        # Utilities
        ("typer", "typer"),
        ("loguru", "loguru"),
        ("tqdm", "tqdm"),
        ("python-dotenv", "dotenv"),
        ("emoji", "emoji"),
        ("contractions", "contractions"),
        ("unidecode", "unidecode"),
    ]
    
    # Check each package
    results = []
    for package_name, import_name in packages_to_check:
        success, message = check_package(package_name, import_name)
        results.append((success, message))
        print(message)
    
    # Summary
    print("\n" + "=" * 50)
    successful = sum(1 for success, _ in results if success)
    total = len(results)
    
    print(f"📊 Installation Summary:")
    print(f"   ✅ Successfully installed: {successful}/{total}")
    print(f"   ❌ Missing packages: {total - successful}/{total}")
    
    if successful == total:
        print("\n🎉 All packages installed successfully!")
        print("🚀 You're ready to start working on the Aria-Core-ANN project!")
        print("\nNext steps:")
        print("1. Run 'jupyter lab' to start Jupyter")
        print("2. Open notebooks/01_comprehensive_eda.ipynb")
        print("3. Run the EDA notebook to explore the dataset")
        return 0
    else:
        print(f"\n⚠️  {total - successful} package(s) are missing.")
        print("Please install missing packages using:")
        print("   conda env update -f environment.yml")
        print("   OR")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())


