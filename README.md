\# CodeAlpha Task 4 - Sales Prediction using Python



\## Project Overview



This project is part of the CodeAlpha Data Science Internship.



The objective of this project is to predict product sales based on advertising expenditure across different platforms such as TV, Radio, and Newspaper.



A Linear Regression machine learning model is used to learn the relationship between advertising expenditure and sales.



\## Dataset



The dataset used is `advertising.csv`.



It contains 200 records and 4 columns:



\- TV - Advertising expenditure on TV

\- Radio - Advertising expenditure on Radio

\- Newspaper - Advertising expenditure on Newspaper

\- Sales - Product sales



There are no missing values in the dataset.



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Scikit-learn



\## Machine Learning Method



Linear Regression is used for predicting sales.



The dataset is divided into:



\- 80% training data

\- 20% testing data



The model uses the following input features:



\- TV

\- Radio

\- Newspaper



Target variable:



\- Sales



\## Model Evaluation



The model produced the following results:



\- Mean Absolute Error (MAE): 1.27

\- Mean Squared Error (MSE): 2.91

\- Root Mean Squared Error (RMSE): 1.71

\- R-squared (R2): 0.9059



The R-squared value indicates that the model explains a substantial portion of the variation in sales in this dataset.



\## Advertising Impact



The regression coefficients obtained from the model are:



\- TV: 0.0545

\- Radio: 0.1009

\- Newspaper: 0.0043



These coefficients describe the fitted linear relationship between each advertising feature and predicted sales while considering the other features in the model.



\## Sample Prediction



For the following advertising expenditure:



\- TV = 150

\- Radio = 30

\- Newspaper = 20



The predicted sales value is:



16.01



\## Visualizations



The project generates two visualizations:



1\. `actual\_vs\_predicted\_sales.png`

&#x20;  - Compares actual sales with predicted sales.



2\. `advertising\_impact.png`

&#x20;  - Displays the regression coefficients for the advertising platforms.



\## Project Files



```text

CodeAlpha\_SalesPrediction/

│

├── advertising.csv

├── sales\_prediction.py

├── actual\_vs\_predicted\_sales.png

├── advertising\_impact.png

├── README.md

└── requirements.txt

