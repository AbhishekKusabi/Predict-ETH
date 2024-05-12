import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Function to fetch historical Ethereum price data from CoinGecko API
def fetch_ethereum_data():
    # Make a request to the CoinGecko API to fetch historical Ethereum price data
    response = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365')
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response JSON data
        ethereum_data = response.json()
        
        # Extract relevant data from the response
        prices = ethereum_data['prices']
        ethereum_df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        
        # Convert timestamp to datetime
        ethereum_df['Timestamp'] = pd.to_datetime(ethereum_df['Timestamp'], unit='ms')
        
        return ethereum_df
    else:
        # If the request failed, return None
        print("Error in api")
        return None

# Load historical Ethereum price data from the CoinGecko API
ethereum_data = fetch_ethereum_data()

if ethereum_data is not None:
    # Data preprocessing
    ethereum_data['Year'] = ethereum_data['Timestamp'].dt.year
    ethereum_data['Month'] = ethereum_data['Timestamp'].dt.month
    ethereum_data['Day'] = ethereum_data['Timestamp'].dt.day
    ethereum_data['Weekday'] = ethereum_data['Timestamp'].dt.weekday
    
    # Split data into features (X) and target variable (y)
    X = ethereum_data[['Year', 'Month', 'Day', 'Weekday']]  # Features
    y = ethereum_data['Price']  # Target variable

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter tuning for Random Forest
    rf_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
    rf_grid_search.fit(X_scaled, y)
    best_rf_model = rf_grid_search.best_estimator_

    # Hyperparameter tuning for Extra Trees
    et_param_grid = {'n_estimators': [100, 200, 300],
                     'max_depth': [None, 10, 20, 30],
                     'min_samples_split': [2, 5, 10]}
    et_grid_search = GridSearchCV(ExtraTreesRegressor(), et_param_grid, cv=5)
    et_grid_search.fit(X_scaled, y)
    best_et_model = et_grid_search.best_estimator_

    # Function to predict Ethereum price for future dates using best Random Forest model
    def predict_prices_for_future_random_forest(date):
    # Preprocess input date provided by the user
     date = pd.to_datetime(date)
     year = date.year
     month = date.month
     day = date.day
     weekday = date.weekday()

    # Feature scaling for prediction
     scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

    # Make prediction
     predicted_price = best_rf_model.predict(scaled_features_for_date)[0]

    # Return tuple of predicted price and default value for predicted high and low
     return predicted_price, predicted_price+100, 0.0



    # Function to predict Ethereum price for future dates using best Extra Trees model
    def predict_prices_for_future_extra_trees(date):
        # Preprocess input date provided by the user
        date = pd.to_datetime(date)
        year = date.year
        month = date.month
        day = date.day
        weekday = date.weekday()

        # Feature scaling for prediction
        scaled_features_for_date = scaler.transform([[year, month, day, weekday]])

        # Make prediction
        predicted_price = best_et_model.predict(scaled_features_for_date)[0]
        return predicted_price
else:
    print("Failed to fetch Ethereum data from the CoinGecko API.")















# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV

# # Load historical Ethereum price data
# ethereum_data = pd.read_csv('ETH-USD.csv')

# # Data preprocessing
# ethereum_data['Date'] = pd.to_datetime(ethereum_data['Date'], dayfirst=True)
# ethereum_data['Price'] = ethereum_data['Price'].astype(float)
# ethereum_data['High'] = ethereum_data['High'].astype(float)
# ethereum_data['Low'] = ethereum_data['Low'].astype(float)

# # Extract date features
# ethereum_data['Year'] = ethereum_data['Date'].dt.year
# ethereum_data['Month'] = ethereum_data['Date'].dt.month
# ethereum_data['Day'] = ethereum_data['Date'].dt.day
# ethereum_data['Weekday'] = ethereum_data['Date'].dt.weekday

# # Split data into features (X) and target variable (y)
# X = ethereum_data[['High', 'Low', 'Year', 'Month', 'Day', 'Weekday']]  # Features
# y = ethereum_data['Price']  # Target variable

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Hyperparameter tuning for Random Forest
# rf_param_grid = {'n_estimators': [100, 200, 300],
#                  'max_depth': [None, 10, 20, 30],
#                  'min_samples_split': [2, 5, 10]}
# rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5)
# rf_grid_search.fit(X_scaled, y)
# best_rf_model = rf_grid_search.best_estimator_

# # Hyperparameter tuning for Extra Trees
# et_param_grid = {'n_estimators': [100, 200, 300],
#                  'max_depth': [None, 10, 20, 30],
#                  'min_samples_split': [2, 5, 10]}
# et_grid_search = GridSearchCV(ExtraTreesRegressor(), et_param_grid, cv=5)
# et_grid_search.fit(X_scaled, y)
# best_et_model = et_grid_search.best_estimator_

# # Function to predict Ethereum price, high, and low for future dates using best Random Forest model
# def predict_prices_for_future_random_forest(date):
#     # Preprocess input date provided by the user
#     date = pd.to_datetime(date, dayfirst=True)
#     year = date.year
#     month = date.month
#     day = date.day
#     weekday = date.weekday()
    
#     # Feature scaling for prediction
#     scaled_features_for_date = scaler.transform([[ethereum_data.iloc[-1]['High'], ethereum_data.iloc[-1]['Low'], year, month, day, weekday]])
    
#     # Make prediction
#     predicted_price = best_rf_model.predict(scaled_features_for_date)[0]
#     predicted_high = predicted_price + 100  # Example adjustment for high
#     predicted_low = predicted_price - 100  # Example adjustment for low
#     return predicted_price, predicted_high, predicted_low

# # Function to predict Ethereum price, high, and low for future dates using best Extra Trees model
# def predict_prices_for_future_extra_trees(date):
#     # Preprocess input date provided by the user
#     date = pd.to_datetime(date, dayfirst=True)
#     year = date.year
#     month = date.month
#     day = date.day
#     weekday = date.weekday()
    
#     # Feature scaling for prediction
#     scaled_features_for_date = scaler.transform([[ethereum_data.iloc[-1]['High'], ethereum_data.iloc[-1]['Low'], year, month, day, weekday]])
    
#     # Make prediction
#     predicted_price = best_et_model.predict(scaled_features_for_date)[0]
#     predicted_high = predicted_price + 100  # Example adjustment for high
#     predicted_low = predicted_price - 100  # Example adjustment for low
#     return predicted_price, predicted_high, predicted_low
