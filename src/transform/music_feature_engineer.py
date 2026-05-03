"""
Music-specific feature engineering transformer.
"""

import numpy as np
import pandas as pd

from .base_transformer import BaseTransformer


class MusicFeatureEngineer(BaseTransformer):
    """Creates derived features for Billboard and Last.fm data."""

    def __init__(self):
        super().__init__('music_feature_engineer', 'music_feature_engineer')

    # ------------------------------------------------------------------
    # per-source engineering
    # ------------------------------------------------------------------

    def _engineer_billboard(self, df: pd.DataFrame) -> pd.DataFrame:
        rank = pd.to_numeric(df['rank'], errors='coerce')
        last_week = pd.to_numeric(df.get('last_week_rank'), errors='coerce')
        weeks = pd.to_numeric(df.get('weeks_on_chart'), errors='coerce')
        peak = pd.to_numeric(df.get('peak_position'), errors='coerce')

        # chart_velocity: negative = climbed (lower rank = better), positive = dropped
        # new entries (last_week == 0 or NaN) get NaN velocity
        df['chart_velocity'] = (rank - last_week.replace(0, np.nan)).astype('Int64')

        # longevity_score: weeks on chart / peak position; higher = sustained success
        df['longevity_score'] = (weeks / peak.replace(0, np.nan)).round(4)

        # peak_ratio: peak / current rank; 1.0 means currently at all-time peak
        df['peak_ratio'] = (peak / rank.replace(0, np.nan)).round(4)

        # decade for era-level analysis
        if 'chart_date' in df.columns:
            df['chart_date'] = pd.to_datetime(df['chart_date'], errors='coerce')
            df['decade'] = (df['chart_date'].dt.year // 10 * 10).astype('Int64')

        return df

    def _engineer_lastfm(self, df: pd.DataFrame) -> pd.DataFrame:
        play_count = pd.to_numeric(df['lastfm_play_count'], errors='coerce')
        listeners = pd.to_numeric(df['lastfm_listeners'], errors='coerce')

        # plays_per_listener: average replays per unique listener
        df['plays_per_listener'] = (
            play_count / listeners.replace(0, np.nan)
        ).round(2)

        # popularity_score: log-scaled composite (0–100)
        # log1p avoids dominance by mega-hit outliers
        log_plays = np.log1p(play_count.fillna(0))
        log_listeners = np.log1p(listeners.fillna(0))

        max_plays = log_plays.max() or 1
        max_listeners = log_listeners.max() or 1

        df['popularity_score'] = (
            (0.6 * log_plays / max_plays + 0.4 * log_listeners / max_listeners) * 100
        ).round(2)

        return df

    # ------------------------------------------------------------------
    # BaseTransformer interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'rank' in df.columns:
            return self._engineer_billboard(df)
        if 'lastfm_play_count' in df.columns:
            return self._engineer_lastfm(df)

        self.logger.warning("MusicFeatureEngineer: unrecognised DataFrame shape — returning as-is")
        return df
