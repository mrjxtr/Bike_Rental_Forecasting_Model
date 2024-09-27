# BIKE RENTAL PREDICTION MODEL 🚲📊

## Project Summary 📝

The **Bike Rental Prediction Model** (ONGOING PROJECT) aims to create a model using historical data and Scikit-Learn's RandomForestRegressor that forecasts daily bike rental demand based on various factors. Helping the business optimize staffing, inventory management, and revenue.

<br />

<div align="center">
  
  [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mrjxtr)
  [![Upwork](https://img.shields.io/badge/-Upwork-6fda44?style=flat-square&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~01f2fd0e74a0c5055a?mp_source=share)
  [![Facebook](https://img.shields.io/badge/-Facebook-1877F2?style=flat-square&logo=facebook&logoColor=white)](https://www.facebook.com/mrjxtr)
  [![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/mrjxtr)
  [![Threads](https://img.shields.io/badge/-Threads-000000?style=flat-square&logo=threads&logoColor=white)](https://www.threads.net/@mrjxtr)
  [![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/mrjxtr)
  [![Gmail](https://img.shields.io/badge/-Gmail-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mr.jesterlumacad@gmail.com)

</div>

### Report Outline 🧾

- [Project Summary](#ProjectSummary)
  - [Features](#Features)
  - [Requirements](#Requirements)
  - [Installation](#Installation)
  - [Usage](#Usage)
  - [Project Structure](#ProjectStructure)
    - [Notes](#Notes)

<br />

### Tools and Technologies 🛠️

- **Programming Languages:** Python
- **Libraries/Frameworks:**
  - **Data Manipulation & Analysis:** Pandas, NumPy
  - **Data Visualization:** Matplotlib, Seaborn
  - **Machine Learning:** Scikit-Learn (RandomForestRegressor)
  - **Data Preprocessing:** Scikit-Learn, Pandas
  - **Model Export:** Joblib
  - **Model Evaluation:** Scikit-Learn (MSE, R-Squared)
- **Data Storage:** CSV files
- **Version Control:** Git
- **Development Environment:** VSCode

<br />

### Project Steps 🛤 <a name="Installation"></a>

1. **Data Collection:**
   - Download data from the following sources:
     - Original Source: <http://capitalbikeshare.com/system-data>
     - Weather Info: <http://www.freemeteo.com>
     - Holiday Schedule: <http://dchr.dc.gov/page/holiday-schedule>
     - Data Downloaded for Model Development: <https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset?select=hour.csv>

2. **Exploratory Data Analysis (EDA):**
   - Perform EDA to explore the distribution of bike rentals.
   - Visualize correlations and trends using Matplotlib and Seaborn.
   - Identify key features impacting bike rentals, such as weather, holidays, and seasonal changes.
  
3. **Model Development:**
   - Split the data into training and testing sets.
   - Train the **RandomForestRegressor** model and tune hyperparameters.
   - Evaluate model performance using metrics like Mean Squared Error (MSE) and R-squared.

4. **Save Model for Future Use:**
   - Use `joblib` to save the model for future deployment and predictions.

5. **Documentation and Reporting:**
   - Document the entire process, from data sources to model details.
   - Provide visualizations for actual vs. predicted values.

6. **Future Work:**
   - Update the dataset periodically and test the model with new data.
   - Experiment with different models and hyperparameters for better performance.
   - Consider real-time predictions in a deployment scenario.

<br />

### Visualizations 📊

![](/reports/figures/fig_003_Line_plot_actual_vs_predicted.png)
![](/reports/figures/fig_004_Line_plot_full_actual_vs_predicted.png)

<br />

#### Notes 📌 <a name="Notes"></a>

- **Data Sources**: Original bike-sharing data, weather data, and holiday schedules were used to build this model.
- **Model Updates**: The model is designed for future improvements, such as using a more up-to-date dataset and experimenting with different models.
- **Deployment**: The model is exportable for deployment in production environments using `joblib`.
- **Evaluation**: Metrics such as MSE and R-squared were used to measure the model's performance.
- **Maintenance**: Future work will include refining the model with additional data and testing deployment scenarios for real-time predictions.
