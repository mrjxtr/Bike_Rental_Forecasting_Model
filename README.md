# ONGOING PROJECT

## Project Summary: Bike Rental Prediction Model

**Objective:**
To develop a predictive model that forecasts daily bike rental demand based on various factors. This model aims to optimize staffing, inventory management, and revenue for bike rental operations.

![](/reports/figures/fig_003_Line_plot_actual_vs_predicted.png)

![](/reports/figures/fig_004_Line_plot_full_actual_vs_predicted.png)

**Tools and Technologies Used:**

- **Programming Languages:** Python
- **Libraries/Frameworks:**
  - **Data Manipulation & Analysis:** Pandas, NumPy
  - **Data Visualization:** Matplotlib
  - **Machine Learning:** Scikit-Learn
  - **Data Preprocessing:** Scikit-Learn, Pandas
  - **Model Export:** Joblib
  - **Model Evaluation:** Scikit-Learn
- **Data Storage:** CSV files
- **Version Control:** Git
- **Development Environment:** VSCode

**Project Steps:**

1. **Data Collection:**
   - Download the data from the data source:
      Original Source: <http://capitalbikeshare.com/system-data>
      Weather Information: <http://www.freemeteo.com>
      Holiday Schedule: <http://dchr.dc.gov/page/holiday-schedule>

      Data Downloaded for Model Development:
      <https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset?select=hour.csv>

2. **Exploratory Data Analysis (EDA):**
   - Perform EDA to understand the distribution of bike rentals and the impact of different features.
   - Visualize correlations and trends using Matplotlib and Seaborn.
   - Identify patterns and relationships between variables.
   - Identify target variable and features for model development.

3. **Model Development:**
   - Split the data into training and testing sets.
   - Train model using RandomForestRegressor model with tuned hyperparameters
   - Tune hyperparameters and evaluate model performance using metrics such as Mean Squared Error (MSE), and R-squared.

4. **Save Model for Future Use:**
   - Save the model for future deployment joblib.

5. **Documentation and Reporting:**
   - Document the entire process, including data sources, preprocessing steps, model details, and evaluation results.
   - Provide a final report summarizing findings and recommendations for optimizing bike rental operations.

6. **Future Work:**
   - Create a more up to date data set and use the saved model for predictions.
   - Experiment with using different models and hyperparameters for for potentially better performance metrics.
   - Consider testing in a deployment scenario where the model is used to make real-time predictions.
