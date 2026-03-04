# COVID-19 Global Data ETL Pipeline

This ETL pipeline extracts COVID-19 data from multiple sources, transforms it through comprehensive cleaning and feature engineering, and loads it into a PostgreSQL database for analysis. 

<img width="2008" height="1091" alt="architecture_diagram" src="https://github.com/user-attachments/assets/a2c1037d-7c08-4b4d-bb9c-0fe31b4add1c" />

The project showcases:

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

### Stack

- Python
- PostgreSQL database (Supabase)
- Git


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
