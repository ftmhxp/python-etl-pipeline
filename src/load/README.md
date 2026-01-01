# COVID-19 ETL Load Module

This module handles the loading phase of the COVID-19 ETL pipeline, responsible for loading transformed data into a PostgreSQL database.

## Overview

The load module provides:

- **BaseLoader**: Abstract base class for all loaders
- **SQLLoader**: PostgreSQL database loader with connection pooling
- **COVIDDataLoader**: Specialized loader for COVID-19 data with preprocessing
- **LoaderOrchestrator**: Coordinates the entire loading pipeline
- **Database Schema**: Complete table definitions and constraints

## Database Schema

The module creates the following tables:

### countries
Demographic and geographic data for countries.

### covid_cases
Daily COVID-19 case data with calculated metrics.

### vaccinations
Vaccination data with various dose counts and rates.

### testing
COVID-19 testing data and positivity rates.

## Usage

### Command Line (via main.py)

```bash
# Load all processed data
python main.py load

# Load specific data files
python main.py load --data-files countries=data/processed/countries.csv --data-files covid_cases=data/processed/covid_cases.csv

# Skip table/index creation
python main.py load --no-create-tables --no-create-indexes
```

### Programmatic Usage

```python
from src.load import LoaderOrchestrator

# Create orchestrator
orchestrator = LoaderOrchestrator()

# Run complete loading pipeline
results = orchestrator.run_loading_pipeline()

# Load specific data
from src.load import COVIDDataLoader, SQLLoader
import pandas as pd

# Load countries data
countries_df = pd.read_csv('data/processed/countries.csv')
sql_loader = SQLLoader()
data_loader = COVIDDataLoader(sql_loader)
result = data_loader.load_countries_data(countries_df)
```

## Configuration

The load process uses configuration from `config.yaml`:

```yaml
database:
  type: "postgresql"
  host: "your-host"
  port: 5432
  database: "your-database"
  username: "your-username"
  password: "your-password"

pipeline:
  load:
    batch_size: 1000
    if_exists: "replace"  # replace, append, or fail
```

## Features

### Data Preprocessing
- Country code standardization
- Date format normalization
- Numeric data cleaning
- Duplicate removal
- Derived metric calculation

### Database Operations
- Automatic table creation
- Index creation for performance
- Foreign key constraints
- Connection pooling
- Error handling and retries

### Loading Strategies
- Full refresh (replace)
- Incremental append
- Batch processing for performance

## Error Handling

The load module includes comprehensive error handling:

- Database connection failures
- Invalid data formats
- Constraint violations
- Batch processing with rollback

## Performance

- Batch loading (configurable batch size)
- Connection pooling
- Database indexes for query performance
- Parallel processing support

## Testing

Run the test script to verify functionality:

```bash
python test_load.py
```

This tests imports, schema definitions, and orchestrator creation.
