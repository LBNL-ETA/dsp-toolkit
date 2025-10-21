# Distribution Substation Planning Toolkit

## Overview
This repository contains the software and documentation for Distribution Substation Planning (DSP). The goal of this project is to develop a software toolkit for (1) data curation, (2) short-term electric load forecasting, and (3) weather-sensitive load adjustment for future peak demand predictions.

## Features

- Data preprocessing and curation tools
- Data-driven short-term load forecasting pipeline
- Automatic weather-sensitive modeling and analysis

## Installation

### Step 0 (optional, recommended): Use a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. You can create a new virtual environment using `venv` or `conda`.

### Step 1: Install PyTorch

Building Data Copilot depends on PyTorch. Please install the appropriate version of PyTorch based on your system and CUDA setup. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Example for CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:
```
pip install torch
```

### Step 2: Install the `dsp-toolkit`

Once PyTorch is installed, you can install the toolkit:
```
cd dsp_toolkit
pip install .
```

For development mode, you can install the toolkit in editable mode, along with dev tools:
```
cd dsp_toolkit
pip install -e .[dev]
```


## Usage
This library provides a set of tools and APIs for distribution substation planning. Example usage can be found in this [notebook](notebooks/example.ipynb). 

## Contributing
Contributions are welcome! If you'd like to report a bug, suggest a feature, or contribute code, please visit the GitHub repository and open an issue or pull request.

## License
This project is licensed under the PLACE-HOLDER License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, contact the authors:
