"""
Database schema definitions for the music ETL pipeline.
"""

from typing import Dict, Any, List


MUSIC_SCHEMA = {
    "artists": {
        "artist_id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
        },
        "name": {
            "type": "TEXT",
            "nullable": False,
        },
        "name_clean": {
            "type": "TEXT",
            "nullable": True,
        },
        "lastfm_mbid": {
            "type": "TEXT",
            "nullable": True,
        },
    },

    "tracks": {
        "track_id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
        },
        "title": {
            "type": "TEXT",
            "nullable": False,
        },
        "artist_name": {
            "type": "TEXT",
            "nullable": False,
        },
        "duration_ms": {
            "type": "INTEGER",
            "nullable": True,
        },
        "lastfm_play_count": {
            "type": "BIGINT",
            "nullable": True,
        },
        "lastfm_listeners": {
            "type": "INTEGER",
            "nullable": True,
        },
        "plays_per_listener": {
            "type": "DECIMAL(8,2)",
            "nullable": True,
        },
        "popularity_score": {
            "type": "DECIMAL(6,2)",
            "nullable": True,
        },
    },

    "chart_entries": {
        "entry_id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
        },
        "title": {
            "type": "TEXT",
            "nullable": False,
        },
        "artist_name": {
            "type": "TEXT",
            "nullable": False,
        },
        "chart_date": {
            "type": "DATE",
            "nullable": False,
        },
        "rank": {
            "type": "SMALLINT",
            "nullable": False,
        },
        "peak_position": {
            "type": "SMALLINT",
            "nullable": True,
        },
        "weeks_on_chart": {
            "type": "SMALLINT",
            "nullable": True,
        },
        "last_week_rank": {
            "type": "SMALLINT",
            "nullable": True,
        },
        "is_new": {
            "type": "BOOLEAN",
            "nullable": True,
        },
        "chart_velocity": {
            "type": "SMALLINT",
            "nullable": True,
        },
        "longevity_score": {
            "type": "DECIMAL(8,4)",
            "nullable": True,
        },
        "peak_ratio": {
            "type": "DECIMAL(8,4)",
            "nullable": True,
        },
        "decade": {
            "type": "SMALLINT",
            "nullable": True,
        },
    },

    "track_tags": {
        "tag_id": {
            "type": "SERIAL",
            "nullable": False,
            "primary_key": True,
        },
        "title": {
            "type": "TEXT",
            "nullable": False,
        },
        "artist_name": {
            "type": "TEXT",
            "nullable": False,
        },
        "tag_name": {
            "type": "TEXT",
            "nullable": False,
        },
    },
}


MUSIC_INDEXES = {
    "artists": [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_artists_name ON artists(name)",
    ],
    "tracks": [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_title_artist ON tracks(title, artist_name)",
        "CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name)",
    ],
    "chart_entries": [
        "CREATE INDEX IF NOT EXISTS idx_chart_entries_date ON chart_entries(chart_date)",
        "CREATE INDEX IF NOT EXISTS idx_chart_entries_title_artist ON chart_entries(title, artist_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chart_entries_unique ON chart_entries(title, artist_name, chart_date)",
    ],
    "track_tags": [
        "CREATE INDEX IF NOT EXISTS idx_track_tags_title_artist ON track_tags(title, artist_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_track_tags_unique ON track_tags(title, artist_name, tag_name)",
    ],
}

# Load order matters: artists before tracks before chart_entries/track_tags
MUSIC_TABLE_ORDER = ['artists', 'tracks', 'chart_entries', 'track_tags']


def get_music_table_schema(table_name: str) -> Dict[str, Any]:
    if table_name not in MUSIC_SCHEMA:
        raise ValueError(f"Table '{table_name}' not found. Available: {list(MUSIC_SCHEMA.keys())}")
    return MUSIC_SCHEMA[table_name]


def get_music_table_names() -> List[str]:
    return MUSIC_TABLE_ORDER


def get_music_indexes() -> Dict[str, List[str]]:
    return MUSIC_INDEXES.copy()
