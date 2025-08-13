# Customer Churn Prediction Using Artificial Neural Network (ANN)

This project predicts **customer churn** (whether a customer will leave the service or not) using an Artificial Neural Network (ANN) built with **TensorFlow/Keras**. The model is trained on customer data to classify churn as a binary outcome.

## 📌 Project Overview
Customer churn is a critical business problem, especially for subscription-based companies. By predicting churn in advance, companies can take proactive actions to retain customers.

This notebook implements:
- Data preprocessing (handling categorical & numerical data)
- Feature scaling
- Building and training an ANN
- Evaluating model performance

## 📂 Dataset
- The dataset contains customer details such as demographics, account information, and service usage.
- The target variable is `Exited` (1 = churn, 0 = not churn).
- DataSet : https://drive.google.com/file/d/1dQDZhdaQdrKOqW64YtPEq3vzwBSI2yRP/view?usp=drive_link

> **Note:** The dataset should be placed in the same directory or updated in the code.

## ⚙️ Technologies Used
- **Python**
- **Google Colab**
- **TensorFlow / Keras**
- **Pandas & NumPy**
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-learn** (for preprocessing & evaluation)


## 🧠 Model Architecture
1. **Input Layer** — Number of neurons = number of features after preprocessing.
2. **Hidden Layers** — Fully connected (Dense) layers with ReLU activation.
3. **Output Layer** — 1 neuron with Sigmoid activation (for binary classification).

**Loss Function:** `binary_crossentropy`  
**Optimizer:** `adam`  
**Evaluation Metric:** `accuracy`


## 🚀 How to Run
1. Open the notebook in **Google Colab**.
2. Upload the dataset.
3. Run all cells sequentially.
4. View the final accuracy and performance metrics.


## 📊 Results
- Model achieves a good accuracy on test data.
- Can be further improved by hyperparameter tuning, adding dropout, or using more complex architectures.

## 🔮 Future Improvements
- Try different optimizers (SGD, RMSProp)
- Implement **Dropout layers** to prevent overfitting
- Use **GridSearchCV** for hyperparameter tuning
- Deploy the model as a web app using Flask or Streamlit
