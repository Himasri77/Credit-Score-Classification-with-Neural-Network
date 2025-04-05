# Credit-Score-Classification-with-Neural-Network
This script loads and preprocesses a credit score dataset to train a deep learning model that predicts credit scores as Good, Standard, or Poor. The key steps include:

Data Visualization & Analysis: Heatmaps, boxplots, and histograms are used to explore feature correlations and distributions.

Data Cleaning & Feature Engineering:

Outlier detection via IQR.

Encoding categorical variables and extracting loan type info.

Mapping ordinal categorical values to numerical.

Imbalance Handling: Uses SMOTE to balance the target classes.

Data Scaling: Applies StandardScaler and RobustScaler through a ColumnTransformer.

Modeling:

Trains a deep neural network with dropout and batch normalization layers.

Uses EarlyStopping to prevent overfitting.

Evaluation: Confusion matrix, classification report for both train and test sets.

Final Model Training: Retrains on the entire balanced dataset and saves the scaler.

Prediction:

Applies the trained model to new data (df_prediction), outputs class labels and normalized class probabilities.

Saves final predictions to final_output.csv.
