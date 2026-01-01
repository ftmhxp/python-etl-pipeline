"""
Data loading logic for COVID-19 ETL pipeline tables.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_loader import BaseLoader
from .sql_loader import SQLLoader
from .database_schema import get_table_schema, get_table_indexes, get_foreign_key_constraints


class COVIDDataLoader:
    """Handles loading of COVID-19 data into database tables."""

    def __init__(self, sql_loader: SQLLoader):
        """Initialize the COVID data loader.

        Args:
            sql_loader: SQL loader instance for database operations
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.sql_loader = sql_loader

        # Country code mapping for standardization
        self.country_code_mapping = self._load_country_code_mapping()

    def _load_country_code_mapping(self) -> Dict[str, str]:
        """Load country name to ISO code mapping.

        Returns:
            Dictionary mapping country names to ISO codes
        """
        # Basic mapping - in a real implementation, this would be loaded from a comprehensive dataset
        return {
            'United States': 'USA',
            'US': 'USA',
            'China': 'CHN',
            'India': 'IND',
            'Brazil': 'BRA',
            'Russia': 'RUS',
            'United Kingdom': 'GBR',
            'UK': 'GBR',
            'France': 'FRA',
            'Germany': 'DEU',
            'Italy': 'ITA',
            'Spain': 'ESP',
            'Canada': 'CAN',
            'Australia': 'AUS',
            'Japan': 'JPN',
            'South Korea': 'KOR',
            'Mexico': 'MEX',
            'Indonesia': 'IDN',
            'Turkey': 'TUR',
            'Saudi Arabia': 'SAU',
            'United Arab Emirates': 'ARE',
            'Israel': 'ISR',
            'South Africa': 'ZAF',
            'Egypt': 'EGY',
            'Nigeria': 'NGA',
            'Kenya': 'KEN',
            'Ethiopia': 'ETH',
            'Ghana': 'GHA',
            'Morocco': 'MAR',
            'Algeria': 'DZA',
            'Tunisia': 'TUN',
            'Libya': 'LBY',
            'Sudan': 'SDN',
            'Chad': 'TCD',
            'Niger': 'NER',
            'Mali': 'MLI',
            'Burkina Faso': 'BFA',
            'Senegal': 'SEN',
            'Gambia': 'GMB',
            'Guinea': 'GIN',
            'Sierra Leone': 'SLE',
            'Liberia': 'LBR',
            'Cote d\'Ivoire': 'CIV',
            'Togo': 'TGO',
            'Benin': 'BEN',
            'Cameroon': 'CMR',
            'Central African Republic': 'CAF',
            'Democratic Republic of the Congo': 'COD',
            'Republic of the Congo': 'COG',
            'Gabon': 'GAB',
            'Equatorial Guinea': 'GNQ',
            'Angola': 'AGO',
            'Namibia': 'NAM',
            'Botswana': 'BWA',
            'Zimbabwe': 'ZWE',
            'Mozambique': 'MOZ',
            'Malawi': 'MWI',
            'Zambia': 'ZMB',
            'Tanzania': 'TZA',
            'Uganda': 'UGA',
            'Rwanda': 'RWA',
            'Burundi': 'BDI',
            'Somalia': 'SOM',
            'Djibouti': 'DJI',
            'Eritrea': 'ERI',
            'Thailand': 'THA',
            'Vietnam': 'VNM',
            'Malaysia': 'MYS',
            'Singapore': 'SGP',
            'Philippines': 'PHL',
            'Pakistan': 'PAK',
            'Bangladesh': 'BGD',
            'Afghanistan': 'AFG',
            'Iran': 'IRN',
            'Iraq': 'IRQ',
            'Jordan': 'JOR',
            'Lebanon': 'LBN',
            'Syria': 'SYR',
            'Yemen': 'YEM',
            'Oman': 'OMN',
            'Qatar': 'QAT',
            'Bahrain': 'BHR',
            'Kuwait': 'KWT',
            'Uzbekistan': 'UZB',
            'Kazakhstan': 'KAZ',
            'Turkmenistan': 'TKM',
            'Kyrgyzstan': 'KGZ',
            'Tajikistan': 'TJK',
            'Azerbaijan': 'AZE',
            'Georgia': 'GEO',
            'Armenia': 'ARM',
            'Ukraine': 'UKR',
            'Belarus': 'BLR',
            'Moldova': 'MDA',
            'Poland': 'POL',
            'Czech Republic': 'CZE',
            'Slovakia': 'SVK',
            'Hungary': 'HUN',
            'Romania': 'ROU',
            'Bulgaria': 'BGR',
            'Serbia': 'SRB',
            'Croatia': 'HRV',
            'Bosnia and Herzegovina': 'BIH',
            'Montenegro': 'MNE',
            'Kosovo': 'XKX',
            'Albania': 'ALB',
            'North Macedonia': 'MKD',
            'Greece': 'GRC',
            'Portugal': 'PRT',
            'Netherlands': 'NLD',
            'Belgium': 'BEL',
            'Luxembourg': 'LUX',
            'Switzerland': 'CHE',
            'Austria': 'AUT',
            'Slovenia': 'SVN',
            'Ireland': 'IRL',
            'Denmark': 'DNK',
            'Sweden': 'SWE',
            'Norway': 'NOR',
            'Finland': 'FIN',
            'Iceland': 'ISL',
            'Estonia': 'EST',
            'Latvia': 'LVA',
            'Lithuania': 'LTU',
            'Chile': 'CHL',
            'Argentina': 'ARG',
            'Uruguay': 'URY',
            'Paraguay': 'PRY',
            'Bolivia': 'BOL',
            'Peru': 'PER',
            'Colombia': 'COL',
            'Venezuela': 'VEN',
            'Ecuador': 'ECU',
            'Guyana': 'GUY',
            'Suriname': 'SUR',
            'French Guiana': 'GUF',
            'Panama': 'PAN',
            'Costa Rica': 'CRI',
            'Nicaragua': 'NIC',
            'Honduras': 'HND',
            'El Salvador': 'SLV',
            'Guatemala': 'GTM',
            'Belize': 'BLZ',
            'Cuba': 'CUB',
            'Haiti': 'HTI',
            'Dominican Republic': 'DOM',
            'Jamaica': 'JAM',
            'Trinidad and Tobago': 'TTO',
            'Barbados': 'BRB',
            'Bahamas': 'BHS',
            'Antigua and Barbuda': 'ATG',
            'Saint Lucia': 'LCA',
            'Grenada': 'GRD',
            'Saint Vincent and the Grenadines': 'VCT',
            'Dominica': 'DMA',
            'Saint Kitts and Nevis': 'KNA'
        }

    def load_countries_data(self, data: Union[pd.DataFrame, str],
                           if_exists: str = 'replace') -> Dict[str, Any]:
        """Load countries/demographic data into the countries table.

        Args:
            data: Countries data to load
            if_exists: Action if table exists ('replace', 'append', 'fail')

        Returns:
            Dictionary with loading results
        """
        self.logger.info("Loading countries data")

        try:
            # Convert data to DataFrame
            df = self._validate_data(data) if isinstance(data, (str, dict)) else data

            # Preprocess countries data
            processed_df = self._preprocess_countries_data(df)

            # Load to database
            result = self.sql_loader.load_data(
                processed_df,
                'countries',
                if_exists=if_exists,
                index=False
            )

            self.logger.info(f"Loaded {result['rows_loaded']} countries")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load countries data: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rows_loaded": 0,
                "timestamp": pd.Timestamp.now().timestamp()
            }

    def load_covid_cases_data(self, data: Union[pd.DataFrame, str],
                             if_exists: str = 'append') -> Dict[str, Any]:
        """Load COVID-19 cases data into the covid_cases table.

        Args:
            data: COVID-19 cases data to load
            if_exists: Action if table exists ('replace', 'append', 'fail')

        Returns:
            Dictionary with loading results
        """
        self.logger.info("Loading COVID-19 cases data")

        try:
            # Convert data to DataFrame
            df = self._validate_data(data) if isinstance(data, (str, dict)) else data

            # Preprocess COVID cases data
            processed_df = self._preprocess_covid_cases_data(df)

            # Load to database
            result = self.sql_loader.load_data(
                processed_df,
                'covid_cases',
                if_exists=if_exists,
                index=False
            )

            self.logger.info(f"Loaded {result['rows_loaded']} COVID-19 case records")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load COVID-19 cases data: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rows_loaded": 0,
                "timestamp": pd.Timestamp.now().timestamp()
            }

    def load_vaccinations_data(self, data: Union[pd.DataFrame, str],
                              if_exists: str = 'append') -> Dict[str, Any]:
        """Load vaccinations data into the vaccinations table.

        Args:
            data: Vaccinations data to load
            if_exists: Action if table exists ('replace', 'append', 'fail')

        Returns:
            Dictionary with loading results
        """
        self.logger.info("Loading vaccinations data")

        try:
            # Convert data to DataFrame
            df = self._validate_data(data) if isinstance(data, (str, dict)) else data

            # Preprocess vaccinations data
            processed_df = self._preprocess_vaccinations_data(df)

            # Load to database
            result = self.sql_loader.load_data(
                processed_df,
                'vaccinations',
                if_exists=if_exists,
                index=False
            )

            self.logger.info(f"Loaded {result['rows_loaded']} vaccination records")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load vaccinations data: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rows_loaded": 0,
                "timestamp": pd.Timestamp.now().timestamp()
            }

    def load_testing_data(self, data: Union[pd.DataFrame, str],
                         if_exists: str = 'append') -> Dict[str, Any]:
        """Load testing data into the testing table.

        Args:
            data: Testing data to load
            if_exists: Action if table exists ('replace', 'append', 'fail')

        Returns:
            Dictionary with loading results
        """
        self.logger.info("Loading testing data")

        try:
            # Convert data to DataFrame
            df = self._validate_data(data) if isinstance(data, (str, dict)) else data

            # Preprocess testing data
            processed_df = self._preprocess_testing_data(df)

            # Load to database
            result = self.sql_loader.load_data(
                processed_df,
                'testing',
                if_exists=if_exists,
                index=False
            )

            self.logger.info(f"Loaded {result['rows_loaded']} testing records")
            return result

        except Exception as e:
            self.logger.error(f"Failed to load testing data: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "rows_loaded": 0,
                "timestamp": pd.Timestamp.now().timestamp()
            }

    def _preprocess_countries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess countries data for loading.

        Args:
            df: Raw countries DataFrame

        Returns:
            Processed DataFrame ready for loading
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Standardize column names
        column_mapping = {
            'Country/Region': 'country_name',
            'Country': 'country_name',
            'Country Name': 'country_name',
            'Country_Code': 'country_code',
            'ISO_Code': 'country_code',
            'Population': 'population',
            'GDP per capita': 'gdp_per_capita',
            'Continent': 'continent',
            'Region': 'continent'
        }

        processed_df = processed_df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ['country_code', 'country_name']
        for col in required_columns:
            if col not in processed_df.columns:
                if col == 'country_code' and 'country_name' in processed_df.columns:
                    # Try to map country names to codes
                    processed_df[col] = processed_df['country_name'].map(self.country_code_mapping)
                else:
                    raise ValueError(f"Required column '{col}' not found in countries data")

        # Standardize country codes
        if 'country_name' in processed_df.columns:
            processed_df['country_code'] = processed_df.apply(
                lambda row: self.country_code_mapping.get(row['country_name'], row.get('country_code')),
                axis=1
            )

        # Clean numeric columns
        numeric_columns = ['population', 'gdp_per_capita', 'electricity_access_percent', 'rural_population_percent']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        # Remove duplicates based on country_code
        processed_df = processed_df.drop_duplicates(subset=['country_code'], keep='first')

        # Select only the columns we need
        schema_columns = list(get_table_schema('countries').keys())
        available_columns = [col for col in schema_columns if col in processed_df.columns]
        processed_df = processed_df[available_columns]

        self.logger.info(f"Processed {len(processed_df)} countries")
        return processed_df

    def _preprocess_covid_cases_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess COVID-19 cases data for loading.

        Args:
            df: Raw COVID cases DataFrame

        Returns:
            Processed DataFrame ready for loading
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Handle time series data (melt if needed)
        date_columns = [col for col in processed_df.columns if self._is_date_column(col)]

        if date_columns:
            # This is time series format, melt it
            id_vars = [col for col in processed_df.columns if col not in date_columns]
            processed_df = processed_df.melt(
                id_vars=id_vars,
                value_vars=date_columns,
                var_name='date',
                value_name='value'
            )

            # Convert date column
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

            # Pivot to get separate columns for confirmed, deaths, recoveries
            if 'variable' in processed_df.columns:
                # Try to identify the type from the source column
                processed_df['data_type'] = processed_df.get('variable', '').str.lower()

                # Create separate dataframes for each type
                confirmed_df = processed_df[processed_df['data_type'].str.contains('confirmed')].copy()
                deaths_df = processed_df[processed_df['data_type'].str.contains('death')].copy()
                recovered_df = processed_df[processed_df['data_type'].str.contains('recover')].copy()

                # Rename value columns
                confirmed_df = confirmed_df.rename(columns={'value': 'confirmed_cases'})
                deaths_df = deaths_df.rename(columns={'value': 'deaths'})
                recovered_df = recovered_df.rename(columns={'value': 'recoveries'})

                # Merge the dataframes
                merge_keys = ['country_code', 'country_name', 'date']
                available_keys = [k for k in merge_keys if k in confirmed_df.columns]

                final_df = confirmed_df[available_keys + ['confirmed_cases']].copy()

                if not deaths_df.empty:
                    final_df = final_df.merge(
                        deaths_df[available_keys + ['deaths']],
                        on=available_keys,
                        how='outer'
                    )

                if not recovered_df.empty:
                    final_df = final_df.merge(
                        recovered_df[available_keys + ['recoveries']],
                        on=available_keys,
                        how='outer'
                    )

                processed_df = final_df

        # Standardize column names
        column_mapping = {
            'Country/Region': 'country_name',
            'Country': 'country_name',
            'Country_Code': 'country_code',
            'Date': 'date',
            'Confirmed': 'confirmed_cases',
            'Deaths': 'deaths',
            'Recovered': 'recoveries',
            'Active': 'active_cases'
        }

        processed_df = processed_df.rename(columns=column_mapping)

        # Ensure country_code exists
        if 'country_code' not in processed_df.columns and 'country_name' in processed_df.columns:
            processed_df['country_code'] = processed_df['country_name'].map(self.country_code_mapping)

        # Convert date column
        if 'date' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

        # Clean numeric columns
        numeric_columns = ['confirmed_cases', 'deaths', 'recoveries', 'active_cases',
                          'cases_per_million', 'deaths_per_million', 'mortality_rate',
                          'recovery_rate', 'doubling_time']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        # Calculate derived metrics if not present
        if 'confirmed_cases' in processed_df.columns and 'deaths' in processed_df.columns:
            if 'mortality_rate' not in processed_df.columns:
                processed_df['mortality_rate'] = (
                    processed_df['deaths'] / processed_df['confirmed_cases']
                ).where(processed_df['confirmed_cases'] > 0)

            if 'recovery_rate' not in processed_df.columns and 'recoveries' in processed_df.columns:
                processed_df['recovery_rate'] = (
                    processed_df['recoveries'] / processed_df['confirmed_cases']
                ).where(processed_df['confirmed_cases'] > 0)

        # Remove rows with missing essential data
        processed_df = processed_df.dropna(subset=['country_code', 'date'])

        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['country_code', 'date'], keep='first')

        # Select only the columns we need
        schema_columns = list(get_table_schema('covid_cases').keys())
        available_columns = [col for col in schema_columns if col in processed_df.columns]
        processed_df = processed_df[available_columns]

        self.logger.info(f"Processed {len(processed_df)} COVID-19 case records")
        return processed_df

    def _preprocess_vaccinations_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess vaccinations data for loading.

        Args:
            df: Raw vaccinations DataFrame

        Returns:
            Processed DataFrame ready for loading
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Standardize column names
        column_mapping = {
            'Country/Region': 'country_name',
            'Country': 'country_name',
            'Country_Code': 'country_code',
            'Date': 'date',
            'Total_Vaccinations': 'total_vaccinations',
            'People_Vaccinated': 'people_vaccinated',
            'People_Fully_Vaccinated': 'people_fully_vaccinated',
            'Daily_Vaccinations': 'daily_vaccinations',
            'Total_Vaccinations_Per_Hundred': 'total_vaccinations_per_hundred',
            'People_Vaccinated_Per_Hundred': 'people_vaccinated_per_hundred',
            'People_Fully_Vaccinated_Per_Hundred': 'people_fully_vaccinated_per_hundred'
        }

        processed_df = processed_df.rename(columns=column_mapping)

        # Ensure country_code exists
        if 'country_code' not in processed_df.columns and 'country_name' in processed_df.columns:
            processed_df['country_code'] = processed_df['country_name'].map(self.country_code_mapping)

        # Convert date column
        if 'date' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

        # Clean numeric columns
        numeric_columns = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                          'daily_vaccinations', 'total_vaccinations_per_hundred',
                          'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        # Remove rows with missing essential data
        processed_df = processed_df.dropna(subset=['country_code', 'date'])

        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['country_code', 'date'], keep='first')

        # Select only the columns we need
        schema_columns = list(get_table_schema('vaccinations').keys())
        available_columns = [col for col in schema_columns if col in processed_df.columns]
        processed_df = processed_df[available_columns]

        self.logger.info(f"Processed {len(processed_df)} vaccination records")
        return processed_df

    def _preprocess_testing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess testing data for loading.

        Args:
            df: Raw testing DataFrame

        Returns:
            Processed DataFrame ready for loading
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Standardize column names
        column_mapping = {
            'Country/Region': 'country_name',
            'Country': 'country_name',
            'Country_Code': 'country_code',
            'Date': 'date',
            'Total_Tests': 'total_tests',
            'New_Tests': 'new_tests',
            'Total_Tests_Per_Thousand': 'total_tests_per_thousand',
            'New_Tests_Per_Thousand': 'new_tests_per_thousand',
            'Tests_Per_Case': 'tests_per_case',
            'Positive_Rate': 'positive_rate'
        }

        processed_df = processed_df.rename(columns=column_mapping)

        # Ensure country_code exists
        if 'country_code' not in processed_df.columns and 'country_name' in processed_df.columns:
            processed_df['country_code'] = processed_df['country_name'].map(self.country_code_mapping)

        # Convert date column
        if 'date' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')

        # Clean numeric columns
        numeric_columns = ['total_tests', 'new_tests', 'total_tests_per_thousand',
                          'new_tests_per_thousand', 'tests_per_case', 'positive_rate']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        # Remove rows with missing essential data
        processed_df = processed_df.dropna(subset=['country_code', 'date'])

        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['country_code', 'date'], keep='first')

        # Select only the columns we need
        schema_columns = list(get_table_schema('testing').keys())
        available_columns = [col for col in schema_columns if col in processed_df.columns]
        processed_df = processed_df[available_columns]

        self.logger.info(f"Processed {len(processed_df)} testing records")
        return processed_df

    def _is_date_column(self, column_name: str) -> bool:
        """Check if a column name represents a date.

        Args:
            column_name: Column name to check

        Returns:
            True if column represents a date, False otherwise
        """
        import re

        # Common date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or M/D/YYYY
            r'\d{4}-\d{2}-\d{2}',        # YYYY-MM-DD
            r'\d{2}-\d{2}-\d{4}',        # DD-MM-YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # D-M-YYYY
        ]

        # Check if column name matches date patterns
        for pattern in date_patterns:
            if re.match(pattern, str(column_name)):
                return True

        # Check for month abbreviations
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in months:
            if month in str(column_name):
                return True

        return False
