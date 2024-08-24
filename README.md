# Project Title: Price Plan Recommendation System

## Project Overview
This Django project implements a Price Plan Recommendation System that uses a deep learning model to recommend the best price plan based on customer clusters. The model uses the "wide and deep" architecture to combine deep learning layers with a linear component for improved prediction performance. The dataset used in this project is preprocessed and clustered, and the model is trained to classify customers into 25 clusters.


## Requirements
To run this project, you will need the following packages installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `joblib`

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used is stored as `Reprocessed_Clustered_data.csv` and contains features related to customer behavior and a target column `Cluster` that represents the cluster to which the customer belongs.

## Model Architecture
The project uses a "wide and deep" neural network model. The model is built using the TensorFlow Keras API, and the architecture includes:
1. **Deep component**: Multiple dense layers with BatchNormalization and Dropout for regularization.
2. **Wide component**: A linear layer for simple interactions.
3. **Combined output**: The output layer uses Softmax activation to predict one of 25 customer clusters.

### Model Compilation
- **Optimizer**: Adam with a learning rate of 0.0005.
- **Loss function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

### Early Stopping
The model is trained with early stopping to prevent overfitting. It monitors the validation loss and stops training if the model doesn't improve for 5 consecutive epochs, restoring the best weights.

## Data Preprocessing
- The dataset is split into features (X) and the target (y).
- The target variable `Cluster` is one-hot encoded.
- The data is split into training (80%) and testing (20%) sets.
- Numerical features are scaled using `StandardScaler`.

## Training the Model
The model is trained for up to 50 epochs with a batch size of 64. Validation is performed using the test set, and early stopping is applied to stop training when validation performance stagnates.

## Saving the Model
- **Model**: The trained model is saved as `model.h5` in the `priceplan/recommendations/models/` directory.
- **Scaler**: The `StandardScaler` object used for feature scaling is saved as `scaler.pkl` for future use in scaling new data.

## How to Run the Project

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Place your dataset in the root directory of the project.
4. Run the Django server using `python manage.py runserver`.
5. Access the recommendations through the provided endpoints.

## Future Improvements
- Integration with a frontend for real-time customer plan recommendations.
- Model optimization and hyperparameter tuning for better performance.
- Deployment of the model as a web service using Django and REST APIs.

## License
This project is licensed under the MIT License.
