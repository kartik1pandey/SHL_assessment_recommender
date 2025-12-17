from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="assessment-recommender",
    version="0.1.0",
    author="Assessment Recommender Team",
    author_email="team@assessment-recommender.com",
    description="A system for recommending assessments based on job descriptions and skill requirements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/assessment-recommender",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Human Resources",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "causal": [
            "dowhy>=0.8.0",
            "causalml>=0.13.0",
        ],
        "ml": [
            "scikit-optimize>=0.9.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
        ],
        "deep": [
            "torch>=1.10.0",
            "transformers>=4.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "build-catalog=assessments.build_catalog:main",
            "extract-skills=job_skill_extraction.extract_skills:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv", "*.txt", "*.md"],
    },
)