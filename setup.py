from setuptools import setup, find_packages

setup(
    name="gmocoin-backtest",
    version="0.1.1",
    description="gmocoin-backtest is a python library \
        for backtest with gmocoin fx btc trade technical \
            analysis on Python 3.7 and above.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="10mohi6",
    author_email="10.mohi.6.y@gmail.com",
    url="https://github.com/10mohi6/gmocoin-backtest-python",
    keywords="gmocoin trade python backtest fx strategy technical analysis jpy btc",
    packages=find_packages(),
    install_requires=[
        "requests",
        "numpy",
        "pandas",
        "matplotlib",
        "beautifulsoup4",
        "SQLAlchemy",
        "lxml",
    ],
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
    ],
)
