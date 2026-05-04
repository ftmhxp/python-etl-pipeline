"""
Tests for MusicFeatureEngineer derived feature logic.
"""

import pytest
import pandas as pd
import numpy as np
from src.transform.music_feature_engineer import MusicFeatureEngineer


@pytest.fixture(scope='module')
def engineer():
    return MusicFeatureEngineer()


def _billboard_df():
    return pd.DataFrame({
        'chart_date':     ['2020-01-04', '2015-06-13', '2009-03-07'],
        'rank':           [5,   1,   50],
        'last_week_rank': [10,  3,   0],   # 0 = new entry
        'weeks_on_chart': [8,   20,  1],
        'peak_position':  [2,   1,   50],
        'title':          ['Song A', 'Song B', 'Song C'],
        'artist_main':    ['Artist A', 'Artist B', 'Artist C'],
    })


def _lastfm_df():
    return pd.DataFrame({
        'title':             ['Song A', 'Song B'],
        'artist':            ['Artist A', 'Artist B'],
        'lastfm_play_count': [1_000_000, 500_000_000],
        'lastfm_listeners':  [100_000,   5_000_000],
    })


class TestChartVelocity:
    def test_positive_velocity_means_drop(self, engineer):
        # rank 5, last_week 10 -> velocity = 5 - 10 = -5 (climbed)
        result = engineer.transform(_billboard_df())
        assert result.loc[0, 'chart_velocity'] == -5

    def test_new_entry_gets_nan_velocity(self, engineer):
        # last_week_rank == 0 should yield NaN
        result = engineer.transform(_billboard_df())
        assert pd.isna(result.loc[2, 'chart_velocity'])


class TestLongevityScore:
    def test_higher_weeks_higher_score(self, engineer):
        result = engineer.transform(_billboard_df())
        # Song B: 20/1 = 20.0, Song A: 8/2 = 4.0
        assert result.loc[1, 'longevity_score'] > result.loc[0, 'longevity_score']

    def test_score_is_numeric(self, engineer):
        result = engineer.transform(_billboard_df())
        assert pd.api.types.is_numeric_dtype(result['longevity_score'])


class TestDecade:
    def test_correct_decades(self, engineer):
        result = engineer.transform(_billboard_df())
        assert result.loc[0, 'decade'] == 2020
        assert result.loc[1, 'decade'] == 2010
        assert result.loc[2, 'decade'] == 2000


class TestLastFmFeatures:
    def test_plays_per_listener_calculated(self, engineer):
        result = engineer.transform(_lastfm_df())
        assert 'plays_per_listener' in result.columns
        expected = round(1_000_000 / 100_000, 2)
        assert result.loc[0, 'plays_per_listener'] == expected

    def test_popularity_score_in_range(self, engineer):
        result = engineer.transform(_lastfm_df())
        assert result['popularity_score'].between(0, 100).all()

    def test_higher_plays_higher_popularity(self, engineer):
        result = engineer.transform(_lastfm_df())
        assert result.loc[1, 'popularity_score'] > result.loc[0, 'popularity_score']
