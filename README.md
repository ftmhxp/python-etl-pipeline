# COVID-19 Global Data ETL Pipeline

A comprehensive ETL (Extract, Transform, Load) pipeline for COVID-19 data analysis and prediction. This project demonstrates advanced data engineering skills by building a robust pipeline that handles multiple data sources, complex transformations, and database loading.

## Project Overview

This ETL pipeline extracts COVID-19 data from multiple sources, transforms it through comprehensive cleaning and feature engineering, and loads it into a PostgreSQL database for analysis. The project showcases:

- **Multi-source data extraction** (APIs, CSV, JSON)
- **Advanced data cleaning** (missing values, outliers, duplicates)
- **Feature engineering** (rates, rolling statistics, geographic features)
- **Data validation** and quality checks
- **PostgreSQL integration** with Supabase
- **CLI interface** for pipeline operations
- **Comprehensive testing** and logging

## Data Sources

The pipeline integrates data from:

1. **Johns Hopkins University** - Daily COVID-19 case/death/recovery data
2. **Our World in Data** - Comprehensive dataset with vaccinations and testing
3. **World Health Organization** - Global situation reports via API
4. **Country metadata** - Geographic and demographic information

## Architecture

```
etl-covid19-portfolio/
├── data/
│   ├── raw/                    # Downloaded data files
│   └── processed/              # Cleaned/transformed data
├── src/
│   ├── extract/               # Data extraction modules
│   ├── transform/             # Data transformation modules
│   ├── load/                  # Data loading modules
│   └── config.py              # Configuration management
├── tests/                     # Unit and integration tests
├── notebooks/                 # Analysis and visualization
├── config.yaml               # Pipeline configuration
├── main.py                   # CLI entry point
└── requirements.txt          # Python dependencies
```


### Prerequisites

- Python 3.8+
- PostgreSQL database (Supabase)
- Git

### Data Files

1. **Johns Hopkins Data** (from GitHub):
   - `time_series_covid19_confirmed_global.csv`
   - `time_series_covid19_deaths_global.csv`
   - `time_series_covid19_recovered_global.csv`

2. **Our World in Data**:
   - `owid-covid-data.csv`

3. **WHO Data**:
   - `WHO-COVID-19-global-data.csv`


## Pipeline Stages

### Extract Phase
- Download data from APIs and local files
- Handle different file formats (CSV, JSON, XML)
- Implement retry logic and error handling
- Validate data integrity

### Transform Phase
- **Data Cleaning**: Handle missing values, duplicates, outliers
- **Feature Engineering**: Create derived metrics (infection rates, doubling times)
- **Data Validation**: Ensure data quality thresholds
- **Normalization**: Standardize country names and date formats

### Load Phase
- Create PostgreSQL tables with proper schemas
- Handle incremental loading and updates
- Implement data partitioning for performance
- Create indexes and constraints

## Database Schema

```sql
-- Countries dimension table
CREATE TABLE countries (
    country_code VARCHAR(3) PRIMARY KEY,
    country_name VARCHAR(255),
    continent VARCHAR(50),
    population BIGINT,
    gdp_per_capita DECIMAL(10,2)
);

-- COVID-19 facts table
CREATE TABLE covid_cases (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) REFERENCES countries(country_code),
    date DATE,
    confirmed_cases INTEGER,
    deaths INTEGER,
    recoveries INTEGER,
    active_cases INTEGER,
    cases_per_million DECIMAL(8,2),
    deaths_per_million DECIMAL(8,2),
    UNIQUE(country_code, date)
);

-- Vaccinations table
CREATE TABLE vaccinations (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) REFERENCES countries(country_code),
    date DATE,
    total_vaccinations BIGINT,
    people_vaccinated BIGINT,
    people_fully_vaccinated BIGINT,
    daily_vaccinations BIGINT,
    UNIQUE(country_code, date)
);
```


## Key Features Demonstrated

- **Data Pipeline Orchestration**: Complete ETL workflow
- **Data Quality**: Comprehensive validation and cleaning
- **Scalability**: Handles large datasets efficiently
- **Modularity**: Clean, maintainable code architecture
- **Testing**: Unit and integration test coverage
