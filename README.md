## Customer Churn Prediction
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


## Project Directory Structure

Here is an overview of the project's directory structure:

```plaintext
Project_Name/
├── app.py                         # Main application file
├── Churn_Modelling.csv            # Dataset for churn modeling
├── Customer_Churn_Prediction_output.png # Output visualization
├── experiments.ipynb              # Jupyter notebook for experiments
├── label_encoder_gender.pkl       # Encoded gender data
├── LICENSE                        # License information
├── model.h5                       # Trained ML model
├── onehot_encoder_geo.pkl         # One-hot encoder for geography
├── prediction.ipynb               # Prediction-related notebook
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── scaler.pkl                     # Scaler for feature scaling
```

## Model Details

The project employs a machine learning model to predict customer churn based on various features. Below are the details of the models and techniques used:

- **Trained Model:** The model is stored as `model.h5`.
- **Data Preprocessing:** 
  - Encoded gender using `label_encoder_gender.pkl`.
  - One-hot encoded geography using `onehot_encoder_geo.pkl`.
  - Applied feature scaling using `scaler.pkl`.
- **Evaluation Metrics:** 
  - Accuracy, Precision, Recall, F1-score.
  - Confusion Matrix for performance analysis.
- **Input Dataset:** The dataset used for training and evaluation is `Churn_Modelling.csv`.

---

## Tech Stack Used

The following technologies and tools were utilized to build this project:

- **Programming Language:** Python
- **Libraries/Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow/Keras
  - TensorBoard (for visualization)
  - Streamlit
- **Development Tools:**
  - Jupyter Notebook
  - Git for version control
- **IDE/Editor:** Visual Studio Code / Jupyter Notebook
- **Environment:** Python Virtual Environment (`churnvenv`)
- **Operating System:** Compatible with Windows, macOS, and Linux


# Credits and Acknowledgements

This project would not have been possible without the guidance, tools, and resources provided by the following:

- **Guidance and Support:**
  - Krish Naik: for project guidance and support.
  - ----------- Name for providing a platform and resources for development.

- **Tools and Libraries:**
  - [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for building and training the machine learning model.
  - [Scikit-learn](https://scikit-learn.org/) for data preprocessing and evaluation.
  - [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation and analysis.
  - [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization.

- **Dataset:**  
  - The dataset used in this project, `Churn_Modelling.csv`, was sourced from [Kaggle](https://www.kaggle.com/).

- **Community and Tutorials:**  
  - Special thanks to the Python and data science community for sharing valuable tutorials and insights.
  - [GitHub](https://github.com/) for version control and collaboration.

---

**Acknowledgement:**  
This project is a result of collaborative learning and the application of knowledge from various online resources, courses, and tutorials. Thank you to everyone who contributed directly or indirectly to its success.

## Future Improvements

This project serves as a foundational model for predicting customer churn. However, there are several areas for enhancement and further development:

1. **Improving Model Performance:**
   - Explore advanced machine learning algorithms such as Gradient Boosting, XGBoost, or LightGBM for improved accuracy.
   - Perform hyperparameter tuning using techniques like Grid Search or Bayesian Optimization.

2. **Feature Engineering:**
   - Add domain-specific features that can provide deeper insights into customer behavior.
   - Incorporate time-series features for customers with recurring transactions or interactions.

3. **Deep Learning Integration:**
   - Experiment with deep learning models such as Recurrent Neural Networks (RNN) or Transformers for sequential or behavioral data analysis.

4. **Real-time Prediction:**
   - Develop an API or integrate the model into a production environment for real-time customer churn prediction.

5. **Explainability and Interpretability:**
   - Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to provide insights into model decisions.

6. **Broader Dataset:**
   - Use a larger, more diverse dataset from multiple industries to generalize the model's applicability.

7. **User Interface:**
   - Create an interactive dashboard to visualize customer churn predictions and metrics for end-users.

8. **Scalability and Deployment:**
   - Host the application on a cloud platform (e.g., AWS, Azure, or GCP) to enable scalability.
   - Implement containerization using Docker and orchestration with Kubernetes for efficient deployment.

9. **Data Privacy and Security:**
   - Ensure the model adheres to data privacy regulations like GDPR, CCPA, etc.
   - Use encryption and secure practices for sensitive data handling.

10. **Integration with CRM Systems:**
    - Integrate the model with Customer Relationship Management (CRM) systems to help organizations take immediate action based on predictions.

By addressing these areas, the project can evolve into a more robust, scalable, and user-friendly system.

## Contact Information


## Output
![App Screenshot](https://github.com/allu0786ansari/Customer_Churn_Prediction/blob/main/Customer_Churn_Prediction_output.png)
