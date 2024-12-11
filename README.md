
## Introduction
Customer churn prediction is a critical aspect of customer relationship management, aimed at identifying customers who are likely to leave or discontinue a service. Understanding and predicting customer churn can help businesses take proactive steps to retain valuable customers, optimize marketing strategies, and reduce revenue loss.

This project leverages machine learning to predict customer churn based on various factors such as demographics, account details, and transaction patterns. By using historical data, we develop a predictive model that classifies whether a customer is likely to churn or remain loyal.

Key Features of the Project
* Data Preprocessing: Includes encoding categorical variables, scaling numerical features, and handling missing data.
* Machine Learning Model: Built using TensorFlow/Keras, providing high accuracy and robust performance.
* Scikit-learn Pipelines: Used for encoding, scaling, and feature engineering.
* Streamlit Dashboard: An interactive and user-friendly interface for churn prediction based on user inputs.
* Interpretability: Understand the factors contributing to customer churn with easy-to-read visualizations and insights.
  
This project serves as a practical implementation of AI in business analytics, demonstrating how predictive modeling can aid in decision-making and customer management. Whether you're looking to explore data science, machine learning, or streamline business strategies, this project provides a hands-on approach to tackling real-world challenges in customer churn.


## Implementation
1. Create a virtual Environment: conda create -p venv python==2.11 -y || python -m venv environ_name
2. Activate the virtual environment: environ_name\Scripts\Activate
3. create a requirements.txt file and use this command to install the libraries: pip install -r requiements.txt
4. If you'r using your device, install ipykernel and run then run the experiments.ipynb and prediction.ipynb.
5. You will get the .pkl files as the output and  model.ht saved as model.
6. I have used tensorboard for visualization the training process, after executing the session in experiments.ipynb, click the launch tensorboard session and select the folder fit which is saved in logs, tensorboard will be launched.
7. In the terminal navigate to your project directory and hit the command to run the app: streamlit run app.py

## Output
![App Screenshot](https://github.com/allu0786ansari/Customer_Churn_Prediction/blob/main/Customer_Churn_Prediction_output.png)
