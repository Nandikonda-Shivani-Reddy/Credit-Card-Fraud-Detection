# Credit-Card-Fraud-Detection
Machine Learning model that can detect fraudulent credit card transactions from a dataset containing transaction details such as amount, merchant information, timestamps, user details, and more.  The goal is to maximize fraud detection accuracy while minimizing false positives, with an analysis of misclassifications.

## Description
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project is sourced from Kaggle and includes transaction details that help in identifying fraudulent activities.

## Dataset
The dataset is split into two files:
- `fraudTrain.csv`: Training data
- `fraudTest.csv`: Test data

Due to size limitations, the datasets are not included in this repository. You can download them from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/). Please ensure the files are placed in the same directory as the Jupyter notebook.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
 How to Run (Using Google Colab)
Open the project notebook in Google Colab:


Upload your dataset:

# python

from google.colab import files
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')

!pip install pandas scikit-learn matplotlib seaborn
Load the model and make predictions:

The trained model creditcard_fraud_model.pkl can be uploaded to the Colab environment similarly.

Follow the notebook steps to preprocess data, make predictions, and evaluate performance.

## Model Information
The project uses a `RandomForestClassifier` with balanced class weights to handle class imbalance in the dataset. The model is trained on preprocessed transaction data to predict fraudulent activities.

## Results
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations are provided in the notebook to illustrate the results.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
