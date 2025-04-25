# QuickLook CSV Uploader & Analyzer

A FastAPI application for uploading, analyzing, and visualizing CSV files with machine learning capabilities.

## Features

- Upload CSV files through a modern web interface
- View file statistics (rows, columns)
- Preview the first 5 rows of the uploaded CSV
- Responsive UI with drag-and-drop support
- Machine learning analysis:
  - Descriptive statistics for numeric columns
  - Principal Component Analysis (PCA) with visualization
  - K-Means clustering with cluster information

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/quicklook.git
cd quicklook
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```
python run.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Upload a CSV file using the web interface.
4. Explore the different tabs to see data preview, statistics, and ML analysis.

## Machine Learning Capabilities

The application automatically performs the following analysis on CSV files with numeric data:

- **Descriptive Statistics**: Mean, standard deviation, min/max values, etc.
- **Principal Component Analysis (PCA)**: Dimensionality reduction to identify the most important features.
- **K-Means Clustering**: Automatically groups similar data points together.

## Requirements

- Python 3.7+
- FastAPI
- pandas
- scikit-learn
- numpy
- uvicorn