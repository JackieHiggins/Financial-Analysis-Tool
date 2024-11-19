import pandas as pd
from prophet import Prophet

def prepare_data_for_prophet(df):
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['date'] = pd.to_datetime(df['date'])
    df['Transaction Amount'] = df['Transaction Amount'].astype(float)
    prophet_df = df.rename(columns={'date': 'ds', 'Transaction Amount': 'y'})
    return prophet_df

def train_prophet_model(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model

def make_prophet_predictions(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def predict_future_transactions(df):
    # Prepare data for Prophet: positive for income, negative for expenses
    prophet_income_df = prepare_data_for_prophet(df[df['Transaction Amount'] > 0])
    prophet_expense_df = prepare_data_for_prophet(df[df['Transaction Amount'] < 0])

    # Train Prophet models for income and expenses
    income_model = train_prophet_model(prophet_income_df)
    expense_model = train_prophet_model(prophet_expense_df)

    # Make predictions
    future_income = make_prophet_predictions(income_model)
    future_expenses = make_prophet_predictions(expense_model)

    # Convert negative expense predictions to absolute values
    future_expenses['yhat'] = future_expenses['yhat'].abs()

    return future_income, future_expenses

# Suppress cmdstanpy logs
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
