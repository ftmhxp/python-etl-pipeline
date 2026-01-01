"""
Database schema definitions for COVID-19 ETL pipeline.
"""

from typing import Dict, Any, List


# COVID-19 Database Schema Definitions
COVID_SCHEMA = {
    "countries": {
        "country_code": {
            "type": "VARCHAR(3)",
            "nullable": False,
            "primary_key": True,
            "description": "ISO 3-letter country code"
        },
        "country_name": {
            "type": "VARCHAR(255)",
            "nullable": False,
            "description": "Full country name"
        },
        "continent": {
            "type": "VARCHAR(50)",
            "nullable": True,
            "description": "Continent name"
        },
        "population": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Total population"
        },
        "gdp_per_capita": {
            "type": "DECIMAL(10,2)",
            "nullable": True,
            "description": "GDP per capita in USD"
        },
        "electricity_access_percent": {
            "type": "DECIMAL(5,2)",
            "nullable": True,
            "description": "Percentage of population with electricity access"
        },
        "rural_population_percent": {
            "type": "DECIMAL(5,2)",
            "nullable": True,
            "description": "Percentage of rural population"
        }
    },

    "covid_cases": {
        "id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
            "description": "Auto-incrementing primary key"
        },
        "country_code": {
            "type": "VARCHAR(3)",
            "nullable": False,
            "description": "ISO 3-letter country code (foreign key to countries)"
        },
        "date": {
            "type": "DATE",
            "nullable": False,
            "description": "Date of the data point"
        },
        "confirmed_cases": {
            "type": "INTEGER",
            "nullable": True,
            "description": "Cumulative confirmed COVID-19 cases"
        },
        "deaths": {
            "type": "INTEGER",
            "nullable": True,
            "description": "Cumulative COVID-19 deaths"
        },
        "recoveries": {
            "type": "INTEGER",
            "nullable": True,
            "description": "Cumulative COVID-19 recoveries"
        },
        "active_cases": {
            "type": "INTEGER",
            "nullable": True,
            "description": "Active COVID-19 cases"
        },
        "cases_per_million": {
            "type": "DECIMAL(8,2)",
            "nullable": True,
            "description": "Confirmed cases per million population"
        },
        "deaths_per_million": {
            "type": "DECIMAL(8,2)",
            "nullable": True,
            "description": "Deaths per million population"
        },
        "mortality_rate": {
            "type": "DECIMAL(5,4)",
            "nullable": True,
            "description": "Case fatality rate (deaths/confirmed)"
        },
        "recovery_rate": {
            "type": "DECIMAL(5,4)",
            "nullable": True,
            "description": "Recovery rate (recoveries/confirmed)"
        },
        "doubling_time": {
            "type": "DECIMAL(6,2)",
            "nullable": True,
            "description": "Case doubling time in days"
        }
    },

    "vaccinations": {
        "id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
            "description": "Auto-incrementing primary key"
        },
        "country_code": {
            "type": "VARCHAR(3)",
            "nullable": False,
            "description": "ISO 3-letter country code (foreign key to countries)"
        },
        "date": {
            "type": "DATE",
            "nullable": False,
            "description": "Date of the vaccination data"
        },
        "total_vaccinations": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Total number of vaccination doses administered"
        },
        "people_vaccinated": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Number of people who received at least one vaccine dose"
        },
        "people_fully_vaccinated": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Number of people who received all required vaccine doses"
        },
        "daily_vaccinations": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Number of vaccination doses administered on this date"
        },
        "total_vaccinations_per_hundred": {
            "type": "DECIMAL(6,2)",
            "nullable": True,
            "description": "Total vaccinations per 100 people"
        },
        "people_vaccinated_per_hundred": {
            "type": "DECIMAL(6,2)",
            "nullable": True,
            "description": "People vaccinated per 100 people"
        },
        "people_fully_vaccinated_per_hundred": {
            "type": "DECIMAL(6,2)",
            "nullable": True,
            "description": "People fully vaccinated per 100 people"
        }
    },

    "testing": {
        "id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
            "description": "Auto-incrementing primary key"
        },
        "country_code": {
            "type": "VARCHAR(3)",
            "nullable": False,
            "description": "ISO 3-letter country code (foreign key to countries)"
        },
        "date": {
            "type": "DATE",
            "nullable": False,
            "description": "Date of the testing data"
        },
        "total_tests": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Total number of tests performed"
        },
        "new_tests": {
            "type": "BIGINT",
            "nullable": True,
            "description": "Number of new tests performed on this date"
        },
        "total_tests_per_thousand": {
            "type": "DECIMAL(8,3)",
            "nullable": True,
            "description": "Total tests per thousand people"
        },
        "new_tests_per_thousand": {
            "type": "DECIMAL(8,3)",
            "nullable": True,
            "description": "New tests per thousand people"
        },
        "tests_per_case": {
            "type": "DECIMAL(8,2)",
            "nullable": True,
            "description": "Tests conducted per new confirmed case"
        },
        "positive_rate": {
            "type": "DECIMAL(5,4)",
            "nullable": True,
            "description": "Share of COVID-19 tests that are positive"
        }
    }
}


# Indexes to create for performance
COVID_INDEXES = {
    "countries": [
        "CREATE INDEX IF NOT EXISTS idx_countries_continent ON countries(continent)",
        "CREATE INDEX IF NOT EXISTS idx_countries_population ON countries(population)"
    ],

    "covid_cases": [
        "CREATE INDEX IF NOT EXISTS idx_covid_cases_country_date ON covid_cases(country_code, date)",
        "CREATE INDEX IF NOT EXISTS idx_covid_cases_date ON covid_cases(date)",
        "CREATE INDEX IF NOT EXISTS idx_covid_cases_country ON covid_cases(country_code)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_covid_cases_unique ON covid_cases(country_code, date)"
    ],

    "vaccinations": [
        "CREATE INDEX IF NOT EXISTS idx_vaccinations_country_date ON vaccinations(country_code, date)",
        "CREATE INDEX IF NOT EXISTS idx_vaccinations_date ON vaccinations(date)",
        "CREATE INDEX IF NOT EXISTS idx_vaccinations_country ON vaccinations(country_code)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_vaccinations_unique ON vaccinations(country_code, date)"
    ],

    "testing": [
        "CREATE INDEX IF NOT EXISTS idx_testing_country_date ON testing(country_code, date)",
        "CREATE INDEX IF NOT EXISTS idx_testing_date ON testing(date)",
        "CREATE INDEX IF NOT EXISTS idx_testing_country ON testing(country_code)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_testing_unique ON testing(country_code, date)"
    ]
}


# Foreign key constraints
COVID_CONSTRAINTS = [
    """
    ALTER TABLE covid_cases
    ADD CONSTRAINT fk_covid_cases_country
    FOREIGN KEY (country_code) REFERENCES countries(country_code)
    ON DELETE CASCADE ON UPDATE CASCADE
    """,

    """
    ALTER TABLE vaccinations
    ADD CONSTRAINT fk_vaccinations_country
    FOREIGN KEY (country_code) REFERENCES countries(country_code)
    ON DELETE CASCADE ON UPDATE CASCADE
    """,

    """
    ALTER TABLE testing
    ADD CONSTRAINT fk_testing_country
    FOREIGN KEY (country_code) REFERENCES countries(country_code)
    ON DELETE CASCADE ON UPDATE CASCADE
    """
]


def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get schema definition for a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Dictionary defining table schema

    Raises:
        ValueError: If table name is not found
    """
    if table_name not in COVID_SCHEMA:
        available_tables = list(COVID_SCHEMA.keys())
        raise ValueError(f"Table '{table_name}' not found. Available tables: {available_tables}")

    return COVID_SCHEMA[table_name]


def get_all_table_names() -> List[str]:
    """Get list of all available table names.

    Returns:
        List of table names
    """
    return list(COVID_SCHEMA.keys())


def get_table_indexes(table_name: str) -> List[str]:
    """Get index creation statements for a specific table.

    Args:
        table_name: Name of the table

    Returns:
        List of SQL index creation statements

    Raises:
        ValueError: If table name is not found
    """
    if table_name not in COVID_INDEXES:
        available_tables = list(COVID_INDEXES.keys())
        raise ValueError(f"Indexes for table '{table_name}' not found. Available tables: {available_tables}")

    return COVID_INDEXES[table_name]


def get_all_indexes() -> Dict[str, List[str]]:
    """Get all index creation statements.

    Returns:
        Dictionary mapping table names to lists of index creation statements
    """
    return COVID_INDEXES.copy()


def get_foreign_key_constraints() -> List[str]:
    """Get all foreign key constraint creation statements.

    Returns:
        List of SQL constraint creation statements
    """
    return COVID_CONSTRAINTS.copy()
