"""
Tests for music database schema definitions.
"""

import pytest
from src.load.music_schema import (
    MUSIC_SCHEMA,
    MUSIC_TABLE_ORDER,
    get_music_table_schema,
    get_music_table_names,
    get_music_indexes,
)
from src.transform.audio_enricher import AUDIO_COLS


AUDIO_FEATURE_COLS = AUDIO_COLS + ['explicit', 'audio_match_type', 'audio_match_score']


class TestSchemaCompleteness:
    def test_all_expected_tables_exist(self):
        for table in ('artists', 'tracks', 'chart_entries', 'track_tags'):
            assert table in MUSIC_SCHEMA

    def test_tracks_has_all_audio_feature_cols(self):
        tracks = MUSIC_SCHEMA['tracks']
        missing = [c for c in AUDIO_FEATURE_COLS if c not in tracks]
        assert missing == [], f"Missing audio cols in tracks schema: {missing}"

    def test_tracks_has_lastfm_cols(self):
        tracks = MUSIC_SCHEMA['tracks']
        for col in ('lastfm_play_count', 'lastfm_listeners', 'plays_per_listener', 'popularity_score'):
            assert col in tracks

    def test_chart_entries_has_engineered_features(self):
        ce = MUSIC_SCHEMA['chart_entries']
        for col in ('chart_velocity', 'longevity_score', 'peak_ratio', 'decade'):
            assert col in ce

    def test_every_table_has_primary_key(self):
        for table, columns in MUSIC_SCHEMA.items():
            pk_cols = [c for c, defn in columns.items() if defn.get('primary_key')]
            assert len(pk_cols) == 1, f"{table} should have exactly one primary key"


class TestSchemaHelpers:
    def test_get_table_schema_returns_dict(self):
        schema = get_music_table_schema('tracks')
        assert isinstance(schema, dict)
        assert len(schema) > 0

    def test_get_table_schema_raises_on_unknown(self):
        with pytest.raises(ValueError):
            get_music_table_schema('nonexistent_table')

    def test_table_order_matches_schema_keys(self):
        assert set(MUSIC_TABLE_ORDER) == set(MUSIC_SCHEMA.keys())

    def test_get_music_table_names_returns_ordered_list(self):
        names = get_music_table_names()
        assert names[0] == 'artists', "artists must load before tracks (FK dependency)"
        assert names[1] == 'tracks'

    def test_indexes_exist_for_all_tables(self):
        indexes = get_music_indexes()
        for table in MUSIC_SCHEMA:
            assert table in indexes, f"No indexes defined for {table}"
