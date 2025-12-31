"""
Data cleaner for handling missing values, duplicates, and outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats

from .base_transformer import BaseTransformer


class DataCleaner(BaseTransformer):
    """Transformer for cleaning data - handling missing values, duplicates, and outliers."""

    def __init__(self):
        """Initialize the data cleaner."""
        super().__init__("data_cleaner", "cleaner")

        # Load specific cleaning configuration
        self.remove_duplicates = True
        self.fill_missing_strategy = "interpolate"  # interpolate, mean, median, mode, drop
        self.outlier_removal = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input DataFrame.

        Args:
            df: Input DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Step 1: Remove duplicates
        if self.remove_duplicates:
            cleaned_df = self._remove_duplicates(cleaned_df)

        # Step 2: Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)

        # Step 3: Remove columns with too much missing data
        cleaned_df = self._remove_high_missing_columns(cleaned_df)

        # Step 4: Handle outliers
        if self.outlier_removal:
            cleaned_df = self._handle_outliers(cleaned_df)

        # Step 5: Standardize data types
        cleaned_df = self._standardize_data_types(cleaned_df)

        self.logger.info("Data cleaning completed")
        return cleaned_df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()

        removed_count = initial_rows - len(df_cleaned)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")

        return df_cleaned

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()

        # For each column, apply appropriate missing value strategy
        for column in df_cleaned.columns:
            if df_cleaned[column].isnull().any():
                if self._is_numeric_column(df_cleaned[column]):
                    df_cleaned[column] = self._fill_numeric_missing(df_cleaned[column])
                elif self._is_datetime_column(df_cleaned[column]):
                    df_cleaned[column] = self._fill_datetime_missing(df_cleaned[column])
                else:
                    df_cleaned[column] = self._fill_categorical_missing(df_cleaned[column])

        self.logger.info(f"Handled missing values for {df.isnull().any().sum()} columns")
        return df_cleaned

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column is numeric.

        Args:
            series: Pandas Series to check

        Returns:
            True if column is numeric
        """
        try:
            pd.to_numeric(series.dropna(), errors='coerce')
            return True
        except:
            return False

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a column is datetime.

        Args:
            series: Pandas Series to check

        Returns:
            True if column is datetime
        """
        try:
            pd.to_datetime(series.dropna(), errors='coerce')
            return True
        except:
            return False

    def _fill_numeric_missing(self, series: pd.Series) -> pd.Series:
        """Fill missing values in numeric column.

        Args:
            series: Numeric Series with missing values

        Returns:
            Series with missing values filled
        """
        if self.fill_missing_strategy == "interpolate":
            return series.interpolate(method='linear', limit_direction='both')
        elif self.fill_missing_strategy == "mean":
            return series.fillna(series.mean())
        elif self.fill_missing_strategy == "median":
            return series.fillna(series.median())
        else:
            # Drop rows with missing values for this column
            return series.dropna()

    def _fill_datetime_missing(self, series: pd.Series) -> pd.Series:
        """Fill missing values in datetime column.

        Args:
            series: Datetime Series with missing values

        Returns:
            Series with missing values filled
        """
        # For datetime columns, forward fill is usually most appropriate
        return series.fillna(method='ffill')

    def _fill_categorical_missing(self, series: pd.Series) -> pd.Series:
        """Fill missing values in categorical column.

        Args:
            series: Categorical Series with missing values

        Returns:
            Series with missing values filled
        """
        if self.fill_missing_strategy == "mode":
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else "Unknown")
        else:
            return series.fillna("Unknown")

    def _remove_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with too much missing data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with high-missing columns removed
        """
        missing_percentages = df.isnull().sum() / len(df)
        columns_to_drop = missing_percentages[missing_percentages > self.missing_data_threshold].index

        if len(columns_to_drop) > 0:
            self.logger.info(f"Removing {len(columns_to_drop)} columns with >{self.missing_data_threshold*100}% missing data: {list(columns_to_drop)}")
            df_cleaned = df.drop(columns=columns_to_drop)
        else:
            df_cleaned = df
            self.logger.info("No columns removed for high missing data")

        return df_cleaned

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        df_cleaned = df.copy()

        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if self.outlier_method == "tukey":
                df_cleaned[column] = self._remove_outliers_tukey(df_cleaned[column])
            elif self.outlier_method == "zscore":
                df_cleaned[column] = self._remove_outliers_zscore(df_cleaned[column])
            # isolation_forest would require sklearn, skipping for now

        self.logger.info(f"Handled outliers in {len(numeric_columns)} numeric columns")
        return df_cleaned

    def _remove_outliers_tukey(self, series: pd.Series, k: float = 1.5) -> pd.Series:
        """Remove outliers using Tukey's method.

        Args:
            series: Numeric Series
            k: Multiplier for IQR

        Returns:
            Series with outliers removed (replaced with NaN)
        """
        if series.dropna().empty:
            return series

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        # Instead of removing, cap the outliers
        series_cleaned = series.clip(lower=lower_bound, upper=upper_bound)

        outliers_removed = ((series < lower_bound) | (series > upper_bound)).sum()
        if outliers_removed > 0:
            self.logger.debug(f"Clipped {outliers_removed} outliers in {series.name}")

        return series_cleaned

    def _remove_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Remove outliers using Z-score method.

        Args:
            series: Numeric Series
            threshold: Z-score threshold

        Returns:
            Series with outliers removed (replaced with NaN)
        """
        if series.dropna().empty:
            return series

        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = z_scores > threshold

        series_cleaned = series.copy()
        series_cleaned.loc[series_cleaned.index[outliers.reindex(series_cleaned.index, fill_value=False)]] = np.nan

        outliers_removed = outliers.sum()
        if outliers_removed > 0:
            self.logger.debug(f"Removed {outliers_removed} outliers in {series.name}")

        return series_cleaned

    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types in DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized data types
        """
        df_standardized = df.copy()

        # Try to convert string columns that look like dates
        for column in df_standardized.columns:
            if df_standardized[column].dtype == 'object':
                # Try to convert to datetime
                try:
                    df_standardized[column] = pd.to_datetime(df_standardized[column], errors='ignore')
                except:
                    pass

                # Try to convert to numeric
                if df_standardized[column].dtype == 'object':
                    try:
                        df_standardized[column] = pd.to_numeric(df_standardized[column], errors='ignore')
                    except:
                        pass

        self.logger.info("Standardized data types")
        return df_standardized
