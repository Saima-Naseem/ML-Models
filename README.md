# ML Models

Ongoing machine learning assignment repo models, code, and explanations submitted for ML coaching.

## Contents
- **linear-regression/** Linear Regression model predicting diabetes progression

---

##  Multi Linear Regression (Diabetes Dataset)

### Goal
Predict disease progression in diabetes patients using health measurements (age, BMI, blood pressure, etc.) with a simple linear regression model.

### Dataset
Built-in `diabetes` dataset from `sklearn.datasets` consisting of 442 patients, 10 health features, and 1 target (disease progression score).

### Steps Followed

1. **Load data**: Imported the diabetes dataset from sklearn.
2. **Structure data**: Converted raw data into a labeled pandas table (`X` = features, `y` = target).
3. **Train/test split**: Split data into 80% training, 20% testing (`random_state=42` for reproducibility).
4. **Train model**: Fit a `Multi LinearRegression` model on the training data.
5. **Predict**: Tested the trained model to predict target values on unseen test data.
6. **Evaluate**: Measured performance using:
   - **MSE (Mean Squared Error)**: average squared difference between predicted and actual values (lower = better)
   - **R² Score**: how much variance in the target is explained by the model (closer to 1 = better)
7. **Visualize**: Plotted actual vs predicted values, with a red dashed diagonal line representing perfect predictions.

### Result Summary
The model shows a clear positive trend (predictions follow actual values), with moderate accuracy, typical for a multilinear model on real-world medical data with non-linear relationships.

### How to Run this Model
1. Download `multilinear_regression.ipynb` from this repo
2. Upload it to your Google Colab
3. Run all cells in order (top to bottom)
4. View printed MSE/R² scores and the final plot
