"""
Data validator for quality checks and validation rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .base_transformer import BaseTransformer


class DataValidator(BaseTransformer):
    """Transformer for data validation and quality checks."""

    def __init__(self):
        """Initialize the data validator."""
        super().__init__("data_validator", "validator")

        # Load validation configuration
        self.validation_rules = {
            'date_format': True,
            'numeric_ranges': True,
            'logical_consistency': True,
            'completeness': True,
            'uniqueness': True
        }

        # COVID-specific validation rules
        self.covid_validation_rules = {
            'deaths_should_not_exceed_cases': True,
            'recovered_should_not_exceed_cases': True,
            'active_cases_calculation': True,
            'population_realistic': True,
            'date_range_valid': True
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the input DataFrame and return filtered/cleaned data.

        Args:
            df: Input DataFrame to validate

        Returns:
            Validated and cleaned DataFrame
        """
        self.logger.info("Starting data validation process")

        # Make a copy to avoid modifying original
        validated_df = df.copy()

        validation_results = {
            'total_records': len(validated_df),
            'validation_passed': 0,
            'validation_failed': 0,
            'issues_found': []
        }

        # Run validation checks
        validation_results.update(self._validate_date_formats(validated_df))
        validation_results.update(self._validate_numeric_ranges(validated_df))
        validation_results.update(self._validate_logical_consistency(validated_df))
        validation_results.update(self._validate_completeness(validated_df))
        validation_results.update(self._validate_uniqueness(validated_df))

        # COVID-specific validations
        if self._is_covid_data(validated_df):
            covid_results = self._validate_covid_specific_rules(validated_df)
            validation_results.update(covid_results)

            # Apply COVID-specific cleaning based on validation results
            validated_df = self._apply_covid_corrections(validated_df, covid_results)

        # Log validation summary
        self._log_validation_results(validation_results)

        self.logger.info("Data validation completed")
        return validated_df

    def _is_covid_data(self, df: pd.DataFrame) -> bool:
        """Check if the data appears to be COVID-related.

        Args:
            df: Input DataFrame

        Returns:
            True if data appears to be COVID-related
        """
        covid_indicators = ['confirmed', 'deaths', 'recovered', 'cases', 'corona', 'covid']
        columns_lower = [col.lower() for col in df.columns]

        return any(any(indicator in col for col in columns_lower) for indicator in covid_indicators)

    def _validate_date_formats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate date formats in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {'date_format_issues': 0}

        date_columns = self._detect_date_columns(df)

        for col in date_columns:
            try:
                # Try to convert to datetime
                pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isna().sum() - df[col].isnull().sum()
                results['date_format_issues'] += invalid_dates
            except Exception as e:
                results['date_format_issues'] += len(df)
                self.logger.warning(f"Date validation failed for column {col}: {e}")

        return results

    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that contain dates.

        Args:
            df: Input DataFrame

        Returns:
            List of column names that likely contain dates
        """
        date_columns = []

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'dt', 'time']):
                date_columns.append(col)
            else:
                # Try to infer from sample values
                sample = df[col].dropna().head(5)
                if not sample.empty:
                    try:
                        pd.to_datetime(sample, errors='coerce')
                        # Check if most values can be converted to dates
                        converted = pd.to_datetime(sample, errors='coerce')
                        if converted.notna().sum() >= len(sample) * 0.8:
                            date_columns.append(col)
                    except:
                        pass

        return date_columns

    def _validate_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate numeric ranges in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {'numeric_range_issues': 0}

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # Check for negative values in columns that shouldn't have them
            if any(keyword in col.lower() for keyword in ['confirmed', 'deaths', 'cases', 'population']):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    results['numeric_range_issues'] += negative_count
                    self.logger.warning(f"Found {negative_count} negative values in {col}")

            # Check for unrealistic values
            if 'population' in col.lower():
                unrealistic = ((df[col] < 1000) | (df[col] > 1_500_000_000)).sum()
                if unrealistic > 0:
                    results['numeric_range_issues'] += unrealistic
                    self.logger.warning(f"Found {unrealistic} unrealistic population values in {col}")

        return results

    def _validate_logical_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate logical consistency between columns.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {'logical_consistency_issues': 0}

        # This will be expanded based on specific data types
        # For now, just check that numeric columns make sense relative to each other

        return results

    def _validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {'completeness_issues': 0}

        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            results['completeness_issues'] += empty_rows
            self.logger.warning(f"Found {empty_rows} completely empty rows")

        # Check for rows missing critical information
        critical_columns = ['country', 'date']  # Adjust based on data type
        existing_critical = [col for col in critical_columns if col in df.columns]

        if existing_critical:
            critical_missing = df[existing_critical].isnull().any(axis=1).sum()
            if critical_missing > 0:
                results['completeness_issues'] += critical_missing
                self.logger.warning(f"Found {critical_missing} rows missing critical information")

        return results

    def _validate_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate uniqueness constraints.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {'uniqueness_issues': 0}

        # Check for expected unique combinations
        # For COVID data, country-date combinations should typically be unique
        if self._is_covid_data(df):
            country_cols = [col for col in df.columns if 'country' in col.lower()]
            date_cols = [col for col in df.columns if 'date' in col.lower()]

            if country_cols and date_cols:
                country_col = country_cols[0]
                date_col = date_cols[0]

                if df[date_col].dtype != 'datetime64[ns]':
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

                duplicates = df.duplicated(subset=[country_col, date_col]).sum()
                if duplicates > 0:
                    results['uniqueness_issues'] += duplicates
                    self.logger.warning(f"Found {duplicates} duplicate country-date combinations")

        return results

    def _validate_covid_specific_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate COVID-specific rules.

        Args:
            df: COVID DataFrame

        Returns:
            Dictionary with COVID-specific validation results
        """
        results = {}

        # Detect column names
        confirmed_col = self._find_column(df, ['confirmed', 'cases'])
        deaths_col = self._find_column(df, ['deaths'])
        recovered_col = self._find_column(df, ['recovered'])
        active_col = self._find_column(df, ['active'])
        population_col = self._find_column(df, ['population', 'pop'])

        # Rule 1: Deaths should not exceed confirmed cases
        if confirmed_col and deaths_col:
            deaths_exceed_cases = (df[deaths_col] > df[confirmed_col]).sum()
            results['deaths_exceed_cases'] = deaths_exceed_cases
            if deaths_exceed_cases > 0:
                self.logger.warning(f"Found {deaths_exceed_cases} records where deaths exceed confirmed cases")

        # Rule 2: Recovered should not exceed confirmed cases
        if confirmed_col and recovered_col:
            recovered_exceed_cases = (df[recovered_col] > df[confirmed_col]).sum()
            results['recovered_exceed_cases'] = recovered_exceed_cases
            if recovered_exceed_cases > 0:
                self.logger.warning(f"Found {recovered_exceed_cases} records where recovered exceed confirmed cases")

        # Rule 3: Active cases validation
        if confirmed_col and deaths_col and recovered_col and active_col:
            calculated_active = df[confirmed_col] - df[deaths_col] - df[recovered_col]
            active_mismatch = (abs(df[active_col] - calculated_active) > 1).sum()  # Allow small rounding differences
            results['active_cases_mismatch'] = active_mismatch
            if active_mismatch > 0:
                self.logger.warning(f"Found {active_mismatch} records with inconsistent active cases calculation")

        # Rule 4: Population realistic values
        if population_col:
            unrealistic_pop = ((df[population_col] < 1000) | (df[population_col] > 1_500_000_000)).sum()
            results['unrealistic_population'] = unrealistic_pop
            if unrealistic_pop > 0:
                self.logger.warning(f"Found {unrealistic_pop} records with unrealistic population values")

        return results

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find a column containing any of the given keywords.

        Args:
            df: DataFrame to search
            keywords: List of keywords to search for

        Returns:
            Column name if found, None otherwise
        """
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in keywords):
                return col
        return None

    def _apply_covid_corrections(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> pd.DataFrame:
        """Apply corrections based on COVID validation results.

        Args:
            df: Input DataFrame
            validation_results: Results from COVID validation

        Returns:
            Corrected DataFrame
        """
        corrected_df = df.copy()

        # Fix deaths exceeding cases
        if validation_results.get('deaths_exceed_cases', 0) > 0:
            deaths_col = self._find_column(corrected_df, ['deaths'])
            confirmed_col = self._find_column(corrected_df, ['confirmed', 'cases'])

            if deaths_col and confirmed_col:
                mask = corrected_df[deaths_col] > corrected_df[confirmed_col]
                corrected_df.loc[mask, deaths_col] = corrected_df.loc[mask, confirmed_col]
                self.logger.info(f"Corrected {mask.sum()} records where deaths exceeded confirmed cases")

        # Fix recovered exceeding cases
        if validation_results.get('recovered_exceed_cases', 0) > 0:
            recovered_col = self._find_column(corrected_df, ['recovered'])
            confirmed_col = self._find_column(corrected_df, ['confirmed', 'cases'])

            if recovered_col and confirmed_col:
                mask = corrected_df[recovered_col] > corrected_df[confirmed_col]
                corrected_df.loc[mask, recovered_col] = corrected_df.loc[mask, confirmed_col]
                self.logger.info(f"Corrected {mask.sum()} records where recovered exceeded confirmed cases")

        return corrected_df

    def _log_validation_results(self, results: Dict[str, Any]) -> None:
        """Log validation results summary.

        Args:
            results: Validation results dictionary
        """
        total_issues = sum([
            results.get('date_format_issues', 0),
            results.get('numeric_range_issues', 0),
            results.get('logical_consistency_issues', 0),
            results.get('completeness_issues', 0),
            results.get('uniqueness_issues', 0),
            results.get('deaths_exceed_cases', 0),
            results.get('recovered_exceed_cases', 0),
            results.get('active_cases_mismatch', 0),
            results.get('unrealistic_population', 0)
        ])

        self.logger.info("=== Validation Results ===")
        self.logger.info(f"Total records: {results.get('total_records', 0)}")
        self.logger.info(f"Total issues found: {total_issues}")

        if total_issues > 0:
            self.logger.warning("Issues by category:")
            for key, value in results.items():
                if key.endswith('_issues') and value > 0:
                    self.logger.warning(f"  {key}: {value}")
        else:
            self.logger.info("No validation issues found")
