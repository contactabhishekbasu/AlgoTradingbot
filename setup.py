"""
Setup script for AlgoTradingbot
Claude-Powered AI Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="algotrading bot",
    version="0.1.0",
    author="AlgoTradingbot Team",
    author_email="contact@algotrading bot.dev",
    description="AI-powered algorithmic trading system leveraging Claude AI and MCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/contactabhishekbasu/AlgoTradingbot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
        "Framework :: AsyncIO",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "tensorflow>=2.15.0",
        "xgboost>=2.0.0",
        "scikit-learn>=1.4.0",
        "yfinance>=0.2.31",
        "psycopg2-binary>=2.9.9",
        "sqlalchemy>=2.0.23",
        "redis>=5.0.1",
        "aiohttp>=3.9.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.7",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.17.0",
        ],
        "api": [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "algobot=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
