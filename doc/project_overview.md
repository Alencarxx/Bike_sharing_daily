# Project Overview: Bike Sharing Prediction

## 1. Project Description
This project uses a **neural network** built with TensorFlow to predict the number of bicycles rented daily based on various factors like weather conditions, season, and temperature. The dataset used comes from a bike-sharing system and includes features like temperature, humidity, wind speed, and more.

---

## 2. Objectives
The main goals of this project are:
1. Analyze and clean the dataset to prepare it for training.
2. Build and train a neural network to predict daily bike rentals.
3. Visualize the results and evaluate the model's performance using common metrics.
4. Compare the predicted values with the true values to assess the model's accuracy.

---

## 3. Project Structure

BikeSharingPrediction/ 
│ 
├── data/ 
│ └── bike-sharing-daily.csv # Input dataset 
│ 
├── src/ 
│ └── main.py # Main project script 
│ 
├── doc/ 
│ └── project_overview.md # Detailed project documentation 
│ 
├── README.md # Main project document 
├── requirements.txt # Project dependencies 
└── .gitignore # Files to be ignored by Git


---

## 4. Workflow

### 4.1 Data Loading and Cleaning
The dataset is loaded from `data/bike-sharing-daily.csv`. Key steps in this phase include:
- Removing unnecessary columns (`instant`, `casual`, `registered`).
- Converting the date column (`dteday`) into a datetime format.
- Setting the date column as the index for better time-based analysis.

### 4.2 Exploratory Data Analysis (EDA)
- Visualizing trends in bike rentals:
  - Weekly, monthly, and quarterly bike usage trends.
- Exploring correlations between numerical variables like temperature, humidity, wind speed, and daily rentals.
- Generating a heatmap to identify the strength of relationships between features.

### 4.3 Data Preprocessing
- Splitting data into numerical and categorical features:
  - **Categorical**: Season, year, month, holiday, weekday, working day, weather situation.
  - **Numerical**: Temperature, humidity, wind speed.
- Applying one-hot encoding to categorical features.
- Scaling the target variable (`cnt`) using **MinMaxScaler** for better performance during training.
- Splitting the data into training and testing sets using an 80/20 ratio.

### 4.4 Model Building and Training
A neural network is constructed with the following architecture:
- **Input Layer**: Matches the number of input features.
- **Hidden Layers**: Three dense layers with 100 neurons each and `relu` activation.
- **Output Layer**: One neuron with linear activation (to predict a continuous value).
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam optimizer.

The model is trained for 25 epochs with a batch size of 50.

### 4.5 Model Evaluation
- **Loss Plot**: Visualizing training and validation loss over epochs to assess performance.
- **Metrics**:
  - **Mean Absolute Error (MAE)**: Average absolute difference between true and predicted values.
  - **Mean Squared Error (MSE)**: Average squared difference between true and predicted values.
  - **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretability.
  - **R-Squared (R²)**: Proportion of variance in the target variable explained by the model.
  - **Adjusted R²**: Adjusted for the number of predictors in the model.

### 4.6 Predictions
The model predicts the number of bikes rented for unseen data (test set). Predicted values are compared with true values:
- **Scatter Plot**: Visualization of predicted vs. true values to assess the model's performance.

---

## 5. Key Functions in the Code

### 5.1 `load_and_clean_data(file_path)`
- **Purpose**: Load and clean the dataset by removing unnecessary columns and formatting the date.
- **Input**: File path to the dataset.
- **Output**: Cleaned Pandas DataFrame.

### 5.2 `visualize_data(bike)`
- **Purpose**: Perform exploratory data analysis, including trends and correlations.
- **Input**: Cleaned Pandas DataFrame.

### 5.3 `preprocess_data(bike)`
- **Purpose**: Prepares data for training by encoding categorical variables, scaling the target variable, and splitting data into training and testing sets.
- **Input**: Cleaned Pandas DataFrame.
- **Output**: Training and testing sets for features (`X`) and target variable (`y`), and the scaler object.

### 5.4 `build_and_train_model(X_train, y_train)`
- **Purpose**: Build and train a neural network using TensorFlow.
- **Input**: Training features (`X_train`) and target variable (`y_train`).
- **Output**: Trained model and training history.

### 5.5 `evaluate_model(model, history, X_test, y_test, scaler)`
- **Purpose**: Evaluate the model's performance using loss metrics, prediction accuracy, and visualizations.
- **Input**: Trained model, training history, test data, and scaler.
- **Output**: Evaluation metrics and plots.

---

## 6. Results and Insights

### Key Observations:
1. **Neural Network Performance**:
   - The model achieves a reasonable fit with low training and validation loss.
   - Evaluation metrics like RMSE and R² show that the model explains a significant portion of the variance in the target variable.

2. **Correlation Analysis**:
   - Temperature has a strong positive correlation with bike rentals, while humidity shows a moderate negative correlation.
   - Wind speed has a weaker influence on the target variable.

3. **Feature Importance**:
   - Features like season, weather situation, and temperature play a crucial role in predicting bike rentals.

---

## 7. Technologies Used
- **Python**: Programming language.
- **TensorFlow**: For building and training the neural network.
- **scikit-learn**: For preprocessing and evaluation metrics.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.

---

## 8. Author
- **Name**: Alencar Porto   
- **Date**: 04/01/2025
- **Contact**: alencarporto2008@gmail.com

---

## 9. Next Steps
1. Experiment with different neural network architectures and hyperparameters.
2. Incorporate additional features like weather forecasts for improved predictions.
3. Deploy the model using a web framework (e.g., Flask or FastAPI) for real-time predictions.
