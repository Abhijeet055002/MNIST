# MNIST Handwritten Digits Classification

## Overview
This project focuses on analyzing and classifying handwritten digits using the MNIST dataset. Various machine learning and deep learning models are implemented, compared, and evaluated to identify the most effective approach for digit recognition.

---

## Project Structure

### **Files and Directories**:
- **`Numbers.ipynb`**: Jupyter notebook containing the complete analysis, model training, and evaluation process.
- **`README.md`**: Project documentation.
- **`mnist_data/`**: Directory containing the MNIST dataset.

### **Key Components**:
1. **Data Loading and Preprocessing**:
   - The MNIST dataset is loaded, normalized, and prepared for analysis.
2. **Feature Engineering**:
   - Applied Sobel edge detection and Gaussian smoothing.
3. **Model Training**:
   - Implemented Decision Tree, Random Forest, XGBoost, and Neural Network models.
   - Cross-validation is used for some models.
4. **Evaluation**:
   - Models are evaluated using metrics such as accuracy, precision, recall, F1-score, training time, and prediction time.
5. **Insights and Recommendations**:
   - Provided managerial insights and recommendations based on the findings.

---

## Models Implemented

| **Model**          | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Training Time** | **Prediction Time** |
|--------------------|--------------|---------------|------------|--------------|-------------------|---------------------|
| Decision Tree      | 87.80%       | 87.80%        | 87.80%     | 87.79%       | Just under 0.25 min | Under 0.02 sec     |
| Random Forest      | 96.84%       | 96.84%        | 96.84%     | 96.84%       | Under 0.25 min     | Approx. 0.06 sec   |
| XGBoost            | 97.80%       | 97.80%        | 97.80%     | 97.80%       | Just under 0.5 min  | Approx. 0.14 sec   |
| Neural Network     | 97.86%       | 97.86%        | 97.86%     | 97.86%       | Just over 0.1 min   | Under 0.5 sec      |

---

## Insights
- Neural Networks achieve the highest accuracy and are ideal for tasks requiring precision, provided computational resources (GPU) are available.
- XGBoost balances accuracy and speed, making it suitable for projects with moderate computational resources.
- Decision Tree models are quick to train and useful for prototyping but may not deliver high accuracy.
- Cross-validation increases training time significantly but does not always lead to significant performance improvements.

---

## Recommendations
1. **Invest in GPU Hardware**:
   - To maximize the potential of Neural Network models.
2. **Use XGBoost for Balanced Performance**:
   - Ideal for projects requiring moderate accuracy and speed.
3. **Quick Prototyping**:
   - Decision Tree models are recommended for initial prototyping.
4. **Avoid Cross-Validation When Time-Sensitive**:
   - Unless critical, skip cross-validation for faster results.

---

## Requirements

### **Dependencies**:
- Python 3.8+
- TensorFlow 2.18
- Scikit-learn
- XGBoost
- NumPy
- Matplotlib

### **Installation**:
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/mnist-classification.git
   cd mnist-classification
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MNIST dataset and place it in the `mnist_data/` directory.

---

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Numbers.ipynb
   ```
2. Follow the steps in the notebook to:
   - Load and preprocess the data.
   - Train and evaluate models.
   - Analyze the results.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact
For any inquiries, please contact:
- **Name**: Abhijeet
- **Email**: [abhijeet1472@gmail.com]

