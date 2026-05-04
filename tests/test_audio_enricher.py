"""
Tests for audio_enricher: normalisation, artist parsing, and fuzzy matching logic.
"""

import logging
import pytest
import pandas as pd
from unittest.mock import patch

from src.transform.audio_enricher import (
    _normalise,
    _parse_artists,
    AudioEnricher,
    AUDIO_COLS,
    FINAL_COL_ORDER,
)


# ---------------------------------------------------------------------------
# Pure function tests — no fixtures needed
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalise("it's a test!") == "it s a test"

    def test_collapses_whitespace(self):
        assert _normalise("  hello   world  ") == "hello world"

    def test_handles_non_string(self):
        assert isinstance(_normalise(123), str)

    def test_hyphen_becomes_space(self):
        assert _normalise("Jay-Z") == "jay z"


class TestParseArtists:
    def test_list_format_returns_first(self):
        assert _parse_artists("['Drake', 'Future']") == "drake"

    def test_plain_string(self):
        assert _parse_artists("Beyonce") == "beyonce"

    def test_list_with_punctuation(self):
        assert _parse_artists("['Jay-Z']") == "jay z"

    def test_malformed_list_falls_back(self):
        result = _parse_artists("not a list")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Matching tests — use an in-memory Spotify fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def spotify_rows():
    return pd.DataFrame({
        'name':    ['Blinding Lights', 'Shape of You', 'Old Town Road'],
        'artists': ["['The Weeknd']",  "['Ed Sheeran']", "['Lil Nas X', 'Billy Ray Cyrus']"],
        'valence':           [0.334, 0.931, 0.592],
        'energy':            [0.730, 0.736, 0.661],
        'danceability':      [0.514, 0.825, 0.678],
        'acousticness':      [0.001, 0.082, 0.123],
        'instrumentalness':  [0.0,   0.0,   0.0  ],
        'liveness':          [0.094, 0.083, 0.053],
        'speechiness':       [0.060, 0.088, 0.256],
        'loudness':          [-5.9,  -3.2,  -6.7 ],
        'tempo':             [171.0, 96.0,  136.0],
        'key':               [1,     1,     8    ],
        'mode':              [0,     1,     1    ],
        'time_signature':    [4,     4,     4    ],
        'duration_ms':       [200040, 233713, 113163],
        'explicit':          [False, False, False],
    })


@pytest.fixture
def enricher(spotify_rows):
    """AudioEnricher with in-memory Spotify data — skips all file I/O."""
    with patch.object(AudioEnricher, '__init__', return_value=None):
        e = AudioEnricher()

    e.logger = logging.getLogger('test_enricher')
    e.match_threshold = 85

    # Build the prepared dataframe directly (mirrors _prepare_spotify logic)
    df = spotify_rows.copy()
    df['_key_title']  = df['name'].apply(_normalise)
    df['_key_artist'] = df['artists'].apply(_parse_artists)
    df['_match_key']  = df['_key_title'] + ' ||| ' + df['_key_artist']
    keep = ['_match_key', '_key_title', '_key_artist', 'name', 'artists'] + \
           [c for c in AUDIO_COLS + ['duration_ms', 'explicit'] if c in df.columns]
    e._spotify = df[keep].drop_duplicates(subset=['_match_key'])
    return e


class TestMatchTrack:
    def test_exact_match(self, enricher):
        row, match_type, score = enricher._match_track('Blinding Lights', 'The Weeknd')
        assert match_type == 'exact'
        assert score == 100
        assert row is not None

    def test_exact_match_case_insensitive(self, enricher):
        row, match_type, score = enricher._match_track('blinding lights', 'the weeknd')
        assert match_type == 'exact'

    def test_fuzzy_artist_match(self, enricher):
        # Slightly misspelled artist
        row, match_type, score = enricher._match_track('Shape of You', 'Ed Sheeran Jr')
        assert match_type in ('fuzzy_artist', 'fuzzy', 'exact')
        assert row is not None

    def test_no_match_below_threshold(self, enricher):
        row, match_type, score = enricher._match_track('zzz unknown song xyz', 'nobody artist')
        assert match_type == 'no_match'
        assert score == 0
        assert row is None

    def test_matched_row_has_audio_cols(self, enricher):
        row, _, _ = enricher._match_track('Blinding Lights', 'The Weeknd')
        for col in ['valence', 'energy', 'danceability']:
            assert col in row.index


class TestFinalColOrder:
    def test_no_duplicates(self):
        assert len(FINAL_COL_ORDER) == len(set(FINAL_COL_ORDER))

    def test_audio_cols_present(self):
        for col in AUDIO_COLS:
            assert col in FINAL_COL_ORDER
