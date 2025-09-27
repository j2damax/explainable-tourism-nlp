#!/usr/bin/env python3
"""
Installation verification script for the Tourism Review Classification project.
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
    print("🔍 Verifying Tourism Review Classification Project Installation")
    print("=" * 50)
    
    # Core packages to check
    packages_to_check = [
        # Core data science
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        
        # Visualization
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        
        
        # Deep learning
        ("torch", "torch"),
        
        # Hugging Face
        ("transformers", "transformers"),
        ("huggingface-hub", "huggingface_hub"),
        
        # Explainable AI
        ("shap", "shap"),
        
        # Web deployment
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit-shap", "streamlit_shap"),
        
        # API and services
        ("openai", "openai"),
        ("requests", "requests"),
        ("pydantic", "pydantic"),
        
        # Utilities
        ("python-dotenv", "dotenv"),
        ("loguru", "loguru"),
        
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
        print("🚀 You're ready to start working on the Tourism Review Classification project!")
        print("\nNext steps:")
        print("1. Run the Jupyter notebooks in sequence:")
        print("   jupyter lab notebooks/")
        print("2. Start with the first notebook:")
        print("   notebooks/01_eda.ipynb")
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


