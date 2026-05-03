"""
Music data loader and orchestrator for the music ETL pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from ..config import config
from .sql_loader import SQLLoader
from .music_schema import (
    get_music_table_schema, get_music_table_names, get_music_indexes,
    MUSIC_TABLE_ORDER,
)


class MusicDataLoader:
    """Handles per-table loading of music data."""

    def __init__(self, sql_loader: SQLLoader):
        self.logger = logging.getLogger(f"{__name__}.MusicDataLoader")
        self.sql_loader = sql_loader

    # ------------------------------------------------------------------
    # public load methods
    # ------------------------------------------------------------------

    def load_artists(self, df: pd.DataFrame, if_exists: str = 'replace') -> Dict[str, Any]:
        self.logger.info("Loading artists")
        processed = self._prepare_artists(df)
        return self.sql_loader.load_data(processed, 'artists', if_exists=if_exists, index=False)

    def load_tracks(self, df: pd.DataFrame, if_exists: str = 'replace') -> Dict[str, Any]:
        self.logger.info("Loading tracks")
        processed = self._prepare_tracks(df)
        return self.sql_loader.load_data(processed, 'tracks', if_exists=if_exists, index=False)

    def load_chart_entries(self, df: pd.DataFrame, if_exists: str = 'replace') -> Dict[str, Any]:
        self.logger.info("Loading chart_entries")
        processed = self._prepare_chart_entries(df)
        return self.sql_loader.load_data(processed, 'chart_entries', if_exists=if_exists, index=False)

    def load_track_tags(self, df: pd.DataFrame, if_exists: str = 'replace') -> Dict[str, Any]:
        self.logger.info("Loading track_tags")
        processed = self._prepare_track_tags(df)
        if processed.empty:
            self.logger.warning("No tag data found — skipping track_tags load")
            return {"status": "skipped", "rows_loaded": 0}
        return self.sql_loader.load_data(processed, 'track_tags', if_exists=if_exists, index=False)

    # ------------------------------------------------------------------
    # preparation helpers
    # ------------------------------------------------------------------

    def _prepare_artists(self, billboard_df: pd.DataFrame) -> pd.DataFrame:
        """Derive unique artist records from cleaned billboard data."""
        artist_col = 'artist_main' if 'artist_main' in billboard_df.columns else 'artist'
        artists = (
            billboard_df[[artist_col]]
            .drop_duplicates()
            .dropna()
            .rename(columns={artist_col: 'name'})
            .copy()
        )
        # name_clean: lowercase, stripped (for case-insensitive joins later)
        artists['name_clean'] = artists['name'].str.lower().str.strip()
        artists['lastfm_mbid'] = None
        return self._select_schema_columns('artists', artists)

    def _prepare_tracks(self, lastfm_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare track records from cleaned Last.fm data."""
        df = lastfm_df.copy()
        # Rename 'artist' → 'artist_name' to match schema
        if 'artist' in df.columns and 'artist_name' not in df.columns:
            df = df.rename(columns={'artist': 'artist_name'})
        df = df.drop_duplicates(subset=['title', 'artist_name'])
        return self._select_schema_columns('tracks', df)

    def _prepare_chart_entries(self, billboard_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare chart entry records from cleaned + feature-engineered billboard data."""
        df = billboard_df.copy()
        # Normalise artist column name
        if 'artist_main' in df.columns and 'artist_name' not in df.columns:
            df = df.rename(columns={'artist_main': 'artist_name'})
        elif 'artist' in df.columns and 'artist_name' not in df.columns:
            df = df.rename(columns={'artist': 'artist_name'})

        df['chart_date'] = pd.to_datetime(df['chart_date'], errors='coerce')
        df = df.dropna(subset=['title', 'artist_name', 'chart_date', 'rank'])

        # Convert pandas NA integers (Int64) to plain int for DB compatibility
        for col in ['chart_velocity', 'decade', 'rank', 'peak_position',
                    'weeks_on_chart', 'last_week_rank']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return self._select_schema_columns('chart_entries', df)

    def _prepare_track_tags(self, lastfm_df: pd.DataFrame) -> pd.DataFrame:
        """Expand comma-separated tags into one row per (track, tag)."""
        df = lastfm_df.copy()
        if 'tags' not in df.columns or df['tags'].isna().all():
            return pd.DataFrame()

        if 'artist' in df.columns and 'artist_name' not in df.columns:
            df = df.rename(columns={'artist': 'artist_name'})

        tag_rows = []
        for _, row in df[df['tags'].notna()].iterrows():
            for tag in str(row['tags']).split(','):
                tag = tag.strip()
                if tag:
                    tag_rows.append({
                        'title': row['title'],
                        'artist_name': row['artist_name'],
                        'tag_name': tag,
                    })

        if not tag_rows:
            return pd.DataFrame()

        tags_df = pd.DataFrame(tag_rows).drop_duplicates(
            subset=['title', 'artist_name', 'tag_name']
        )
        return self._select_schema_columns('track_tags', tags_df)

    def _select_schema_columns(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        schema_cols = list(get_music_table_schema(table_name).keys())
        # Exclude SERIAL primary key columns (auto-populated by the DB)
        pk_cols = {
            col for col, defn in get_music_table_schema(table_name).items()
            if defn.get('primary_key')
        }
        load_cols = [c for c in schema_cols if c not in pk_cols and c in df.columns]
        return df[load_cols]


class MusicLoaderOrchestrator:
    """Orchestrates schema creation, indexing, and data loading for the music pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MusicLoaderOrchestrator")
        self.load_config = config.get('pipeline.load', {})
        self.sql_loader = SQLLoader(config.get('database', {}))
        self.data_loader = MusicDataLoader(self.sql_loader)
        self.processed_data_path = config.processed_data_path

    def run_loading_pipeline(
        self,
        create_tables: bool = True,
        create_indexes: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info("Starting music loading pipeline")

        try:
            if create_tables:
                table_results = self._create_all_tables()
            else:
                table_results = {"status": "skipped"}

            if create_indexes:
                index_results = self._create_all_indexes()
            else:
                index_results = {"status": "skipped"}

            data_files = self._discover_data_files()
            loading_results, total_rows = self._load_all_tables(data_files)

            duration = time.time() - start_time
            statuses = [r['result'].get('status', 'unknown') for r in loading_results]
            overall = (
                'success' if all(s in ('success', 'skipped') for s in statuses)
                else 'partial_success' if any(s == 'success' for s in statuses)
                else 'failed'
            )

            return {
                "status": overall,
                "total_duration_seconds": duration,
                "total_rows_loaded": total_rows,
                "table_creation": table_results,
                "index_creation": index_results,
                "data_loading": loading_results,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Music loading pipeline failed: {e}")
            return {"status": "failed", "error": str(e), "timestamp": time.time()}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _discover_data_files(self) -> Dict[str, Optional[Path]]:
        """Map table names to processed CSV file paths."""
        files: Dict[str, Optional[Path]] = {t: None for t in MUSIC_TABLE_ORDER}

        if not self.processed_data_path.exists():
            return files

        for fp in self.processed_data_path.glob("*.csv"):
            stem = fp.stem.lower()
            if 'billboard' in stem:
                files['chart_entries'] = fp
                # Artists are derived from the same billboard file
                if files['artists'] is None:
                    files['artists'] = fp
            elif 'lastfm' in stem:
                files['tracks'] = fp
                files['track_tags'] = fp

        self.logger.info(f"Discovered music data files: { {k: str(v) for k, v in files.items()} }")
        return files

    def _create_all_tables(self) -> Dict[str, Any]:
        results, errors = [], []
        for table in get_music_table_names():
            try:
                schema = get_music_table_schema(table)
                created = self.sql_loader.create_table(table, schema, if_exists='skip')
                results.append(f"{'Created' if created else 'Exists'}: {table}")
            except Exception as e:
                errors.append(f"Failed to create {table}: {e}")
                self.logger.error(errors[-1])
        return {"status": "success" if not errors else "partial_success",
                "tables_processed": len(results), "results": results, "errors": errors}

    def _create_all_indexes(self) -> Dict[str, Any]:
        results, errors = [], []
        for table, index_sqls in get_music_indexes().items():
            for sql in index_sqls:
                try:
                    self.sql_loader.execute_query(sql)
                    results.append(f"Index on {table}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        results.append(f"Index on {table} already exists")
                    else:
                        errors.append(f"Index on {table}: {e}")
                        self.logger.error(errors[-1])
        return {"status": "success" if not errors else "partial_success",
                "indexes_processed": len(results), "results": results, "errors": errors}

    def _load_all_tables(
        self, data_files: Dict[str, Optional[Path]]
    ) -> tuple:
        results = []
        total_rows = 0

        # Cache loaded DataFrames to avoid re-reading the same file twice
        _cache: Dict[str, pd.DataFrame] = {}

        def _read(path: Optional[Path]) -> Optional[pd.DataFrame]:
            if path is None:
                return None
            key = str(path)
            if key not in _cache:
                _cache[key] = pd.read_csv(path)
            return _cache[key]

        for table in MUSIC_TABLE_ORDER:
            path = data_files.get(table)
            df = _read(path)

            if df is None:
                results.append({"table": table, "result": {"status": "skipped", "rows_loaded": 0}})
                continue

            try:
                if table == 'artists':
                    result = self.data_loader.load_artists(df)
                elif table == 'tracks':
                    result = self.data_loader.load_tracks(df)
                elif table == 'chart_entries':
                    result = self.data_loader.load_chart_entries(df)
                elif table == 'track_tags':
                    result = self.data_loader.load_track_tags(df)
                else:
                    result = {"status": "skipped", "rows_loaded": 0}
            except Exception as e:
                self.logger.error(f"Failed loading {table}: {e}")
                result = {"status": "failed", "error": str(e), "rows_loaded": 0}

            total_rows += result.get("rows_loaded", 0)
            results.append({"table": table, "result": result})

        return results, total_rows
