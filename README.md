# COVID-19 Global Data ETL Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

## Quick Start

### 1. Prerequisites

- Python 3.8+
- PostgreSQL database (Supabase)
- Git

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ftmhxp/python-etl-pipeline.git
cd python-etl-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data Files

Download the following files and place them in the `data/raw/` directory:

1. **Johns Hopkins Data** (from GitHub):
   - `time_series_covid19_confirmed_global.csv`
   - `time_series_covid19_deaths_global.csv`
   - `time_series_covid19_recovered_global.csv`

2. **Our World in Data**:
   - `owid-covid-data.csv`

3. **WHO Data**:
   - `WHO-COVID-19-global-data.csv`

### 4. Database Setup

#### Option A: Supabase (Recommended)

1. Go to [supabase.com](https://supabase.com) and create a free account
2. Create a new project
3. Go to Settings → Database → Connection string
4. Copy your connection details

#### Option B: Local PostgreSQL

```bash
# Using Docker
docker run --name postgres-etl -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=covid_etl -p 5432:5432 -d postgres:13
```

### 5. Configuration

Create a `.env` file in the project root:

```bash
# Database credentials
DB_USER=your_supabase_user
DB_PASSWORD=your_supabase_password
DB_HOST=db.your-project-ref.supabase.co
DB_PORT=5432
DB_NAME=postgres

# Optional: API keys for additional data sources
# WHO_API_KEY=your_api_key_here
```

### 6. Run the Pipeline

```bash
# Run complete ETL pipeline
python main.py run

# Or run individual phases
python main.py extract
python main.py transform
python main.py load

# Validate data quality
python main.py validate

# Generate analysis reports
python main.py analyze
```

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

## Configuration

The pipeline is highly configurable via `config.yaml`:

```yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  database: "covid_etl"

pipeline:
  extract:
    retry_attempts: 3
    timeout: 30
  transform:
    missing_data_threshold: 0.5
    outlier_method: "tukey"
```

## Analysis & Visualization

Explore the data using the provided Jupyter notebooks:

```bash
# Install analysis dependencies
pip install -r requirements.txt --extra analysis

# Launch Jupyter
jupyter notebook notebooks/
```

Available notebooks:
- `01_data_exploration.ipynb` - Initial data analysis
- `02_etl_pipeline_demo.ipynb` - Pipeline walkthrough
- `03_predictive_modeling.ipynb` - ML model development

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Key Features Demonstrated

- **Data Pipeline Orchestration**: Complete ETL workflow
- **Error Handling**: Robust error handling and logging
- **Data Quality**: Comprehensive validation and cleaning
- **Scalability**: Handles large datasets efficiently
- **Modularity**: Clean, maintainable code architecture
- **Documentation**: Comprehensive documentation and examples
- **Testing**: Unit and integration test coverage