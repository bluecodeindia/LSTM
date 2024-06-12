# Multivariate Time Series Models

## Overview
This repository contains various Python scripts for implementing and evaluating different multivariate time series models. The models include simple LSTM, stacked LSTM, CNN-LSTM, Conv-LSTM, and bidirectional LSTM.

## Repository Structure
- `README.md`: Project documentation.
- `multivariate-bidirectional-lstm.py`: Implementation of a bidirectional LSTM model for multivariate time series prediction.
- `multivariate-cnn-lstm.py`: Implementation of a CNN-LSTM model for multivariate time series prediction.
- `multivariate-conv-lstm.py`: Implementation of a Conv-LSTM model for multivariate time series prediction.
- `multivariate-simple-lstm.py`: Implementation of a simple LSTM model for multivariate time series prediction.
- `multivariate-stacked-lstm.py`: Implementation of a stacked LSTM model for multivariate time series prediction.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install them using pip:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib
```

### Running the Scripts
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/bluecodeindia/Multivariate-Time-Series.git
    cd Multivariate-Time-Series
    ```

2. **Run the Scripts**:
    Use Python to run the scripts. For example, to run the `multivariate-simple-lstm.py` script:
    ```bash
    python multivariate-simple-lstm.py
    ```

## Scripts Explanation

### Multivariate Time Series Models
- `multivariate-bidirectional-lstm.py`: Implements a bidirectional LSTM model for capturing dependencies in both forward and backward directions.
- `multivariate-cnn-lstm.py`: Combines CNN and LSTM to capture spatial and temporal dependencies in the data.
- `multivariate-conv-lstm.py`: Implements a Conv-LSTM model which integrates convolutional layers with LSTM for better feature extraction.
- `multivariate-simple-lstm.py`: Implements a simple LSTM model for multivariate time series prediction.
- `multivariate-stacked-lstm.py`: Implements a stacked LSTM model with multiple LSTM layers to capture complex patterns in the data.

## Contributing
Feel free to contribute by submitting a pull request. Please ensure your changes are well-documented and tested.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact [bluecodeindia@gmail.com](mailto:bluecodeindia@gmail.com).
