from setuptools import setup, find_packages

setup(
    name="s3cmn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "snntorch>=0.5.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.21.0",
        "pytest>=7.0.0"
    ],
    author="Your Name",
    description="Spiking Neural Network with Memory",
    url="https://github.com/yourusername/s3cmn",
)
