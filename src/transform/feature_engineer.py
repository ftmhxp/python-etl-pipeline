"""
Feature engineering transformer for creating derived metrics and features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import math

from .base_transformer import BaseTransformer


class FeatureEngineer(BaseTransformer):
    """Transformer for feature engineering - creating derived metrics and features."""

    def __init__(self):
        """Initialize the feature engineer."""
        super().__init__("feature_engineer", "feature_engineer")

        # Load feature engineering configuration
        self.features_to_create = config.get("pipeline.transform.features", [
            "cases_per_million",
            "deaths_per_million",
            "mortality_rate",
            "recovery_rate",
            "doubling_time"
        ])

        # COVID-specific column mappings
        self.column_mappings = {
            'confirmed': ['confirmed', 'Confirmed', 'cases', 'Cases'],
            'deaths': ['deaths', 'Deaths', 'death', 'Death'],
            'recovered': ['recovered', 'Recovered', 'recovery', 'Recovery'],
            'population': ['population', 'Population', 'pop', 'Pop'],
            'date': ['date', 'Date', 'dt', 'DT'],
            'country': ['country', 'Country', 'nation', 'Nation', 'location', 'Location']
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for the input DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering process")

        # Make a copy to avoid modifying original
        engineered_df = df.copy()

        # Detect column types and standardize names
        column_map = self._detect_columns(engineered_df)

        # Convert date columns to datetime
        if column_map.get('date'):
            engineered_df[column_map['date']] = pd.to_datetime(engineered_df[column_map['date']], errors='coerce')

        # Create features based on configuration
        for feature in self.features_to_create:
            try:
                if feature == "cases_per_million":
                    engineered_df = self._create_cases_per_million(engineered_df, column_map)
                elif feature == "deaths_per_million":
                    engineered_df = self._create_deaths_per_million(engineered_df, column_map)
                elif feature == "mortality_rate":
                    engineered_df = self._create_mortality_rate(engineered_df, column_map)
                elif feature == "recovery_rate":
                    engineered_df = self._create_recovery_rate(engineered_df, column_map)
                elif feature == "doubling_time":
                    engineered_df = self._create_doubling_time(engineered_df, column_map)
                elif feature == "active_cases":
                    engineered_df = self._create_active_cases(engineered_df, column_map)
                elif feature == "case_fatality_rate":
                    engineered_df = self._create_case_fatality_rate(engineered_df, column_map)

            except Exception as e:
                self.logger.warning(f"Failed to create feature '{feature}': {e}")

        # Create additional derived features
        engineered_df = self._create_additional_features(engineered_df, column_map)

        self.logger.info("Feature engineering completed")
        return engineered_df

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect and map column names to standard types.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping column types to actual column names
        """
        column_map = {}

        for col_type, possible_names in self.column_mappings.items():
            for col_name in df.columns:
                if any(name.lower() in col_name.lower() for name in possible_names):
                    column_map[col_type] = col_name
                    break

        # Log detected columns
        detected = [f"{k}: {v}" for k, v in column_map.items()]
        self.logger.info(f"Detected columns: {', '.join(detected)}")

        return column_map

    def _create_cases_per_million(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create cases per million feature.

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with cases_per_million column added
        """
        confirmed_col = column_map.get('confirmed')
        pop_col = column_map.get('population')

        if confirmed_col and pop_col:
            df['cases_per_million'] = (df[confirmed_col] / df[pop_col]) * 1_000_000
            self.logger.info("Created cases_per_million feature")

        return df

    def _create_deaths_per_million(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create deaths per million feature.

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with deaths_per_million column added
        """
        deaths_col = column_map.get('deaths')
        pop_col = column_map.get('population')

        if deaths_col and pop_col:
            df['deaths_per_million'] = (df[deaths_col] / df[pop_col]) * 1_000_000
            self.logger.info("Created deaths_per_million feature")

        return df

    def _create_mortality_rate(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create mortality rate feature (deaths/confirmed).

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with mortality_rate column added
        """
        deaths_col = column_map.get('deaths')
        confirmed_col = column_map.get('confirmed')

        if deaths_col and confirmed_col:
            df['mortality_rate'] = (df[deaths_col] / df[confirmed_col]) * 100
            # Handle division by zero
            df['mortality_rate'] = df['mortality_rate'].replace([np.inf, -np.inf], np.nan)
            self.logger.info("Created mortality_rate feature")

        return df

    def _create_recovery_rate(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create recovery rate feature (recovered/confirmed).

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with recovery_rate column added
        """
        recovered_col = column_map.get('recovered')
        confirmed_col = column_map.get('confirmed')

        if recovered_col and confirmed_col:
            df['recovery_rate'] = (df[recovered_col] / df[confirmed_col]) * 100
            # Handle division by zero
            df['recovery_rate'] = df['recovery_rate'].replace([np.inf, -np.inf], np.nan)
            self.logger.info("Created recovery_rate feature")

        return df

    def _create_active_cases(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create active cases feature (confirmed - recovered - deaths).

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with active_cases column added
        """
        confirmed_col = column_map.get('confirmed')
        recovered_col = column_map.get('recovered')
        deaths_col = column_map.get('deaths')

        if confirmed_col and recovered_col and deaths_col:
            df['active_cases'] = df[confirmed_col] - df[recovered_col] - df[deaths_col]
            self.logger.info("Created active_cases feature")

        return df

    def _create_case_fatality_rate(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create case fatality rate feature (deaths/(recovered + deaths)).

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with case_fatality_rate column added
        """
        recovered_col = column_map.get('recovered')
        deaths_col = column_map.get('deaths')

        if recovered_col and deaths_col:
            df['case_fatality_rate'] = (df[deaths_col] / (df[recovered_col] + df[deaths_col])) * 100
            # Handle division by zero
            df['case_fatality_rate'] = df['case_fatality_rate'].replace([np.inf, -np.inf], np.nan)
            self.logger.info("Created case_fatality_rate feature")

        return df

    def _create_doubling_time(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create doubling time feature for cases.

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with doubling_time column added
        """
        confirmed_col = column_map.get('confirmed')
        date_col = column_map.get('date')
        country_col = column_map.get('country')

        if confirmed_col and date_col and country_col:
            df = self._calculate_doubling_times(df, confirmed_col, date_col, country_col)
            self.logger.info("Created doubling_time feature")

        return df

    def _calculate_doubling_times(self, df: pd.DataFrame, confirmed_col: str,
                                date_col: str, country_col: str) -> pd.DataFrame:
        """Calculate doubling times for each country/location.

        Args:
            df: Input DataFrame
            confirmed_col: Name of confirmed cases column
            date_col: Name of date column
            country_col: Name of country column

        Returns:
            DataFrame with doubling_time column added
        """
        df = df.copy()
        df['doubling_time'] = np.nan

        # Sort by country and date
        df = df.sort_values([country_col, date_col])

        # Group by country and calculate doubling times
        for country, group in df.groupby(country_col):
            if len(group) < 2:
                continue

            # Get cumulative cases
            cases = group[confirmed_col].values
            dates = group[date_col].values

            # Calculate daily growth rates
            doubling_times = []
            for i in range(1, len(cases)):
                if cases[i-1] > 0 and cases[i] > cases[i-1]:
                    # Calculate how many days to double
                    growth_rate = cases[i] / cases[i-1]
                    if growth_rate > 1:
                        dt = math.log(2) / math.log(growth_rate)
                        doubling_times.append(dt)
                    else:
                        doubling_times.append(np.nan)
                else:
                    doubling_times.append(np.nan)

            # Add NaN for first row
            doubling_times.insert(0, np.nan)

            # Update the dataframe
            mask = df[country_col] == country
            df.loc[mask, 'doubling_time'] = doubling_times

        return df

    def _create_additional_features(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Create additional derived features.

        Args:
            df: Input DataFrame
            column_map: Mapping of column types to names

        Returns:
            DataFrame with additional features
        """
        # Create daily new cases if we have cumulative data
        confirmed_col = column_map.get('confirmed')
        date_col = column_map.get('date')
        country_col = column_map.get('country')

        if confirmed_col and date_col and country_col:
            df = self._create_daily_new_cases(df, confirmed_col, date_col, country_col)

        # Create 7-day moving averages
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['confirmed', 'deaths', 'recovered', 'active_cases']:
                ma_col = f"{col}_7d_ma"
                df[ma_col] = df.groupby(country_col)[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )

        self.logger.info("Created additional features (daily changes, moving averages)")
        return df

    def _create_daily_new_cases(self, df: pd.DataFrame, confirmed_col: str,
                              date_col: str, country_col: str) -> pd.DataFrame:
        """Create daily new cases from cumulative data.

        Args:
            df: Input DataFrame
            confirmed_col: Name of confirmed cases column
            date_col: Name of date column
            country_col: Name of country column

        Returns:
            DataFrame with new_cases column added
        """
        df = df.copy()

        # Sort by country and date
        df = df.sort_values([country_col, date_col])

        # Calculate daily new cases
        df['new_cases'] = df.groupby(country_col)[confirmed_col].diff().fillna(0)

        # Ensure non-negative values
        df['new_cases'] = df['new_cases'].clip(lower=0)

        return df
