# Python ETL Pipeline Framework

A modular, extensible ETL framework built in Python. The architecture separates extraction, transformation, and loading into independent, reusable layers — each new data pipeline is a set of concrete implementations that plug into the same base classes and CLI.

<img width="2008" height="1091" alt="architecture_diagram" src="https://github.com/user-attachments/assets/a2c1037d-7c08-4b4d-bb9c-0fe31b4add1c" />

---

## Built-in pipelines

| Pipeline | Sources | Tables |
|---|---|---|
| **COVID-19** | Johns Hopkins University, Our World in Data, WHO | `countries`, `covid_cases`, `vaccinations`, `testing` |
| **Music Charts** | Billboard Hot 100 (2000–2025), Last.fm API | `artists`, `tracks`, `chart_entries`, `track_tags` |

---

## Architecture

Every pipeline follows the same three-layer pattern:

```
Extract  →  Transform  →  Load
```

Each layer is built on an abstract base class. Adding a new data source means subclassing the right base — the orchestrators, retry logic, logging, and CLI wiring are inherited automatically.

```
src/
├── extract/
│   ├── base_extractor.py          # HTTP retry, file download, validation
│   ├── csv_extractor.py           # COVID — CSV over HTTP
│   ├── api_extractor.py           # COVID — REST API
│   ├── billboard_extractor.py     # Music — web scraping
│   └── lastfm_extractor.py        # Music — REST API with rate limiting
│
├── transform/
│   ├── base_transformer.py        # Load, save, quality stats, logging
│   ├── data_cleaner.py            # COVID — nulls, outliers, duplicates
│   ├── feature_engineer.py        # COVID — rates, moving averages
│   ├── data_validator.py          # COVID — schema validation
│   ├── music_cleaner.py           # Music — artist name normalisation
│   └── music_feature_engineer.py  # Music — chart velocity, popularity score
│
└── load/
    ├── base_loader.py             # Batch processing, progress logging
    ├── sql_loader.py              # PostgreSQL via SQLAlchemy
    ├── database_schema.py         # COVID schema + indexes + constraints
    ├── data_loader.py             # COVID table-level loading
    ├── music_schema.py            # Music schema + indexes
    └── music_loader.py            # Music table-level loading + orchestrator
```

---

## Stack

- **Python 3.10+**
- **pandas / numpy / scipy** — data processing
- **requests / beautifulsoup4** — HTTP extraction and scraping
- **billboard.py / pylast** — Billboard and Last.fm clients
- **SQLAlchemy / psycopg2** — PostgreSQL
- **Supabase** — hosted PostgreSQL
- **Click** — CLI
- **Great Expectations / pytest** — validation and testing

---

## Setup

```bash
git clone https://github.com/ftmhxp/python-etl-pipeline.git
cd python-etl-pipeline
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env` and fill in credentials:

```bash
# Database
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=5432
DB_NAME=postgres

# Last.fm (register free at https://www.last.fm/api/account/create)
LASTFM_API_KEY=your_key
LASTFM_API_SECRET=your_secret
```

---

## Usage

### COVID-19 pipeline

```bash
python main.py extract       # download from JHU, OWID, WHO
python main.py transform     # clean, engineer features, validate
python main.py load          # create tables, load to Supabase
```

### Music pipeline

```bash
python main.py music extract --source billboard   # ~1,300 weekly charts
python main.py music extract --source lastfm      # enrich unique tracks
python main.py music transform                    # clean + feature engineer
python main.py music load                         # create tables, load to Supabase
```

Run a full pipeline end-to-end:

```bash
python main.py run           # COVID full pipeline
python main.py music run     # Music full pipeline
```

---

## Adding a new pipeline

1. **Extract** — subclass `BaseExtractor`, implement `extract() -> dict`
2. **Transform** — subclass `BaseTransformer`, implement `transform(df) -> df`
3. **Load** — define a schema dict, subclass pattern from `music_loader.py`
4. **Config** — add a source block to `config.yaml`
5. **CLI** — add a `@cli.group()` in `main.py`

The base classes handle retries, logging, file I/O, batch loading, and progress tracking — your implementation only needs the domain logic.

---

## Configuration

All pipeline behaviour is controlled via `config.yaml`:

- Source URLs, file names, API rate limits
- Data quality thresholds (missing data %, outlier method)
- Database connection and schema
- Logging level and output path

Secrets (passwords, API keys) are read from `.env` and never committed.
