from typing import Optional
import pandas as pd
import numpy as np

class StockAnalysisError(Exception):
    """Custom exception for stock analysis errors."""
    pass

class DataPreprocessor:
    def __init__(self, handle_missing='interpolate'):
        self.handle_missing = handle_missing
    
    def process(self, df):
        print(f"Initial data shape: {df.shape}")
        # Handle missing data or apply any preprocessing steps
        if self.handle_missing == 'interpolate':
            df['Close'] = df['Close'].interpolate(method='linear', axis=0)
        
        print(f"Processed data shape: {df.shape}")
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data structure and contents."""
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        if missing := required_columns - set(df.columns):
            raise StockAnalysisError(f"Missing required columns: {missing}")
        if df.empty:
            raise StockAnalysisError("Dataset is empty")

        # Check for invalid values
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if not pd.to_numeric(df[col], errors='coerce').notnull().all():
                raise StockAnalysisError(f"Invalid numeric values in column: {col}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to the specified method."""
        if self.handle_missing == 'interpolate':
            df = df.interpolate(method='linear', axis=0)
        elif self.handle_missing == 'drop':
            df = df.dropna()
        elif self.handle_missing == 'fill':
            df = df.fillna(method='ffill')
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df