"""
Tests for MusicCleaner transformation logic.
"""

import pytest
import pandas as pd
from src.transform.music_cleaner import MusicCleaner


@pytest.fixture(scope='module')
def cleaner():
    return MusicCleaner()


def _billboard_df(**overrides):
    """Minimal valid billboard DataFrame."""
    data = {
        'chart_date': ['2020-01-04', '2020-01-04', '2020-01-04'],
        'rank':       [1, 2, 3],
        'title':      ['Blinding Lights', 'Rockstar', 'Old Town Road'],
        'artist':     ['The Weeknd', 'Post Malone feat. 21 Savage', 'Lil Nas X ft. Billy Ray Cyrus'],
        'weeks_on_chart': [10, 5, 19],
        'peak_position':  [1, 1, 1],
        'last_week_rank': [1, 2, 3],
        'is_new':         [False, False, False],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _lastfm_df():
    return pd.DataFrame({
        'title':              ['Blinding Lights', 'Shape of You'],
        'artist':             ['The Weeknd', 'Ed Sheeran'],
        'lastfm_play_count':  [500_000_000, 800_000_000],
        'lastfm_listeners':   [5_000_000, 7_000_000],
        'duration_ms':        [200040, 233713],
    })


class TestFeatSplitting:
    def test_feat_dot(self, cleaner):
        main, feat = cleaner._split_featuring('Post Malone feat. 21 Savage')
        assert main == 'Post Malone'
        assert feat == '21 Savage'

    def test_ft_dot(self, cleaner):
        main, feat = cleaner._split_featuring('Lil Nas X ft. Billy Ray Cyrus')
        assert main == 'Lil Nas X'
        assert feat == 'Billy Ray Cyrus'

    def test_featuring_full(self, cleaner):
        main, feat = cleaner._split_featuring('Drake featuring Rihanna')
        assert main == 'Drake'
        assert feat == 'Rihanna'

    def test_no_feature(self, cleaner):
        main, feat = cleaner._split_featuring('The Weeknd')
        assert main == 'The Weeknd'
        assert feat is None

    def test_nan_input(self, cleaner):
        main, feat = cleaner._split_featuring(float('nan'))
        assert feat is None


class TestBillboardCleaning:
    def test_creates_artist_main_col(self, cleaner):
        result = cleaner.transform(_billboard_df())
        assert 'artist_main' in result.columns

    def test_creates_artist_feat_col(self, cleaner):
        result = cleaner.transform(_billboard_df())
        assert 'artist_feat' in result.columns

    def test_feat_correctly_split(self, cleaner):
        result = cleaner.transform(_billboard_df())
        rockstar = result[result['title'] == 'Rockstar'].iloc[0]
        assert rockstar['artist_main'] == 'Post Malone'
        assert rockstar['artist_feat'] == '21 Savage'

    def test_drops_rows_missing_rank(self, cleaner):
        df = _billboard_df()
        df.loc[0, 'rank'] = None
        result = cleaner.transform(df)
        assert len(result) == 2

    def test_chart_date_parsed(self, cleaner):
        result = cleaner.transform(_billboard_df())
        assert pd.api.types.is_datetime64_any_dtype(result['chart_date'])


class TestLastFmCleaning:
    def test_drops_rows_with_no_lastfm_data(self, cleaner):
        df = _lastfm_df().copy()
        df.loc[0, 'lastfm_play_count'] = None
        df.loc[0, 'lastfm_listeners'] = None
        result = cleaner.transform(df)
        assert len(result) == 1

    def test_numeric_cols_coerced(self, cleaner):
        result = cleaner.transform(_lastfm_df())
        assert pd.api.types.is_numeric_dtype(result['lastfm_play_count'])
