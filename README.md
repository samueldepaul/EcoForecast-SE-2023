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

<p align="center">
  <img src="https://github.com/samueldepaul/EcoForecast-SE-2023/blob/main/imgs/4.jpg?raw=true"/>
</p>






# Repositorio de la WebApp Vendimia360

## Link a la WebApp:
Clicando aquí puede accederse a la plataforma :  https://vendimia360.streamlit.app/


##  Secciones de Vendimia 360:

### Home
![](https://github.com/samueldepaul/Vendimia360_webapp/blob/main/img_readme/home.gif?raw=true)

### Base de Datos
![](https://github.com/samueldepaul/Vendimia360_webapp/blob/main/img_readme/base_de_datos.gif?raw=true)

### Predicción
##### Subiendo un CSV con información de múltiples fincas:
![](https://github.com/samueldepaul/Vendimia360_webapp/blob/main/img_readme/prediccion_csv.gif?raw=true)
##### Introduciendo la información de una finca manualmente:
![](https://github.com/samueldepaul/Vendimia360_webapp/blob/main/img_readme/prediccion_manual.gif?raw=true)
