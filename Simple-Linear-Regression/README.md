# ML Models

# 1- Simple Linear Regression - Mango Weight and Price Dataset
## Simple Linear Regression — Mango Weight vs Price

### Goal
Predict mango price based on its weight using a simple linear regression model (1 input → 1 output).

### Dataset
`mango_data.xlsx`: contains two columns: **Weight** and **Price**.

### Steps Followed

1. **Import libraries**: numpy, pandas, matplotlib, and sklearn's `linear_model`.
2. **Load data**: Read `mango_data.xlsx` into a pandas DataFrame.
3. **Visualize raw data**: Scatter plot of Weight (x-axis) vs Price (y-axis) to see the relationship before modeling.
4. **Train model**: Created a `LinearRegression` object and fit it using Weight as the single input feature and Price as the target.
5. **Predict**: Used the trained model to predict price for a new weight value (e.g., weight = 11).
6. **Check score**: Used `.score()` to get the R² value — how well the line fits the data (closer to 1 = better fit).
7. **Inspect the line equation**: Extracted:
   - **Coefficient** (`lr.coef_`) — the slope (how much price changes per unit of weight)
   - **Intercept** (`lr.intercept_`) — the starting price when weight = 0
8. **Manual prediction check**: Verified the model's prediction using the line formula directly:
9. **Visualize the fitted line**: Re-plotted the scatter data with the model's regression line overlaid in green, showing the best-fit straight line through the points.

### Why This Is "Simple" Linear Regression
Only **one** input feature (Weight) is used to predict the target (Price), unlike multiple linear regression, which uses several inputs at once.
### How to Run
1. Upload `mango_data.xlsx` to your Colab environment (or update the file path)
2. Open `simple_linear_regression_Sir_ZI.ipynb` in Google Colab
3. Run all cells in order (top to bottom)
4. View the R² score, coefficient/intercept, and the final fitted-line plot
