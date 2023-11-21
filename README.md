<p align="center">
  <img src="https://github.com/samueldepaul/EcoForecast-SE-2023/blob/main/imgs/1.jpg?raw=true"/>
</p>

<p align="center">
  <img src="https://github.com/samueldepaul/EcoForecast-SE-2023/blob/main/imgs/2.jpg?raw=true"/>
</p>

From November 18th to 21st, 2023, the Schneider Electric Hackathon 2023, hosted by NUWE, unfolded its innovative challenges. I delved into the Data Science challenge, where multiple teams grappled with a common goal:

Given data on the energy consumption of 9 European countries and their generation of renewable or 'green' energies, the task was to predict, for each hour, the country with the highest surplus of green energy in the upcoming hour.

Explore my solution to this challenge below. I hope you find it engaging and recognize the considerable effort I've invested over the past few days.

<p align="center">
  <img src="https://github.com/samueldepaul/EcoForecast-SE-2023/blob/main/imgs/3.jpg?raw=true"/>
</p>

The problem was tackled by building a robust pipeline composed of the following stages:
- Data Ingestion
- Data Processing
- Model Training
- Prediction Generation

Let's delve into a general overview of each phase:
### Data Ingestion:
  - The ENTSO-E API was used for data ingestion.
  - Functions in utils.py were modified to introduce a 2-second delay between API calls to avoid exceeding call limits.
  - Data was ingested for specific PsrType values, filtering information related to green/renewable energies.
  - We generated data within the required timeframe as per the instructions: from 2022-01-01 to 2023-01-01.

### Data Processing:
  - Two optional functions were created to enable users to perform an initial diagnosis with specific information on the datasets just ingested using the API.
  - A custom interpolation function was developed to only interpolate intra-hourly values when there is at least one observation per hour. In such cases, bidirectional interpolation was performed, as set by the organizers.
  - A function was implemented to perform hourly resampling of data with finer periodicity based on their original periodicity.
  - A basic dataframe with essential information was constructed, handling NANs and outliers automatically. However, a significant number of warnings are displayed if necessary.
  - Feature engineering was conducted, considering variables with lags of 1, 2, 3, and 24 hours, monthly and daily grouped variables, and trigonometric transformations (sin, cos) of the date.
  - Before proceeding to computationally intensive tasks like feature selection or model training, a memory reduction of the dataset was executed by efficiently managing data types, resulting in a 66% size reduction.
  - Finally, the 25 most relevant features were selected through a process based on model feature importance, specifically leveraging 5 iterations of feature_importance from LGBM and XGBoost models.

### Model Training:
  - The problem is framed as a tabular data classification problem. We did not utilize any recursive deep learning algorithms or ARIMA. Current state-of-the-art suggests that ensemble models based on gradient boosting (like LGBM and XGBoost), contrary to intuition, perform better in these tasks than ARIMA or RNN models.
  - As the problem seems relatively straightforward (due to limited data variability and most countries not usually having any surplus of renewable energies), before fitting any models, we conducted a baseline prediction. This involved predicting that the country with the highest surplus of green energies in the next hour would be the same as the one in the previous hour.
  - The split used for train-test division is 80-20, as recommended by the organizers.
  - For ML model training, we performed hyperparameter tuning using k = 7-fold cross-validation. Generally, the models performed well on the test set and showed no signs of overfitting.
  - The best model with the optimal hyperparameters is saved in .pkl format for use in a later phase.

### Surplus Prediction:
  - We generated predictions for both the baseline model and the best ML model.
  - Promising metrics were obtained in both cases.
  - Comparing the predictions, we noticed they were very similar, suggesting that the simpler solution might be more suitable for solving this problem.
  - Predictions were saved in a JSON file in the format specified by the organizers.

<p align="center">
  <img src="https://github.com/samueldepaul/EcoForecast-SE-2023/blob/main/imgs/4.jpg?raw=true"/>
</p>

  - The code is written with a focus on efficiency, ease of maintenance, and recyclability for future projects. Functions created are extensively documented and commented for good readability, ensuring anyone can understand their functionality. Moreover, warnings are generated throughout the code to provide users with insights into potential issues in their data or models.
  
  - When cleaning the data, the decision was made to exclude UK data for several reasons:
    - As a consequence of the terms of the post-Brexit Trade and Cooperation Agreement (TCA) between the UK and the EU, there is no longer any obligation for GB to publish information on the Transparency Platform. This results in a significant amount of missing information.
    - The amount of missing data was so substantial that any effort to fill it was unfeasible.
    - The available UK data was too low to be plausible. According to the data, the UK has by far the lowest energy consumption among the 9 considered countries, even below countries with much smaller land areas and populations, such as Denmark.
    - The missing data significantly affected the dataset, making it very confusing to calculate the test set. Without a test set, evaluating the solutions we built was impossible.

  - Trigonometric transformations of time variables (hour, day, month) were applied during feature engineering to better capture their cyclical nature.
  - In the generation of daily or monthly aggregated data, utmost care was taken to avoid data leakage. Under no circumstances did we use future information to calculate values of the current observation. Aggregated variables such as average monthly consumption should be interpreted as the average monthly consumption UP TO THE DATE.

