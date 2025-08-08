# Gemini Code Assistant Project Overview: Traffic Sign Detector

This document provides a comprehensive overview of the Traffic Sign Detector project, designed to be understood and used by the Gemini Code Assistant.

## 1. Project Overview

This project is a deep learning application that classifies German traffic signs from a dataset of 43 classes. It utilizes a Convolutional Neural Network (CNN) built with TensorFlow. The project automatically downloads and preprocesses the dataset, trains the model, and evaluates its performance on test images and new images from the web.

The core of the project is implemented in a Jupyter Notebook (`Traffic_Sign_Classifier.ipynb`), which provides a step-by-step guide through the data loading, preprocessing, model architecture, training, and evaluation phases.

## 2. Technologies Used

- **Programming Language:** Python 3
- **Machine Learning:** TensorFlow, Scikit-learn, NumPy
- **Data Handling:** Pandas, joblib, h5py
- **Image Processing:** OpenCV, Pillow, Matplotlib
- **Development Environment:** Jupyter Notebook
- **Testing:** Pytest, Tox

## 3. Setup and Installation

The project uses a `requirements.txt` file to manage dependencies.

**To set up the environment:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shivam-Bhardwaj/TrafficSignDetector-TensorFlow.git
    cd TrafficSignDetector-TensorFlow
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 4. How to Run

### 4.1. Running the Jupyter Notebook

The main application is the Jupyter Notebook.

1.  **Start the Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    In the Jupyter interface in your browser, open the `Traffic_Sign_Classifier.ipynb` file.

3.  **Run the cells:**
    Execute the cells in the notebook sequentially to see the entire process of data loading, training, and evaluation.

### 4.2. Running Tests

The project includes a test suite using `pytest`.

**To run the tests:**

Execute the following command in the root directory of the project:

```bash
pytest
```

Alternatively, you can use `tox` to run the tests in isolated environments:

```bash
tox
```

## 5. Project Structure

Here is a brief overview of the key files and directories in the project:

-   `.`: Root of the project.
-   `README.md`: The main documentation for the project.
-   `gemini.md`: This file, providing an overview for the Gemini Code Assistant.
-   `requirements.txt`: A list of all the Python dependencies for this project.
-   `Traffic_Sign_Classifier.ipynb`: The main Jupyter Notebook containing the core logic for the traffic sign classifier.
-   `secure_model.py`: A Python script that likely contains a secure version of the model.
-   `security_utils.py`: A Python script with security-related utility functions.
-   `tests/`: A directory containing all the tests for the project.
    -   `test_secure_model.py`: Tests for the secure model.
    -   `test_security_utils.py`: Tests for the security utilities.
-   `assets/`: A directory containing images and other assets used in the `README.md` and notebooks.
-   `examples/`: A directory containing example images for testing the model.
-   `signnames.csv`: A CSV file mapping class IDs to traffic sign names.
-   `tox.ini`: Configuration file for the `tox` testing tool.
-   `pytest.ini`: Configuration file for `pytest`.
