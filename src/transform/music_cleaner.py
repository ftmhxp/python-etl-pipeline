"""
Music-specific data cleaning transformer.
"""

import re
import pandas as pd
from typing import Optional, Tuple

from .base_transformer import BaseTransformer


class MusicCleaner(BaseTransformer):
    """Cleans Billboard and Last.fm raw data."""

    # Matches 'feat.', 'ft.', 'featuring', 'with' and everything after
    _FEAT_RE = re.compile(
        r'\s*(?:feat(?:uring)?\.?|ft\.?|with)\s+.+$',
        flags=re.IGNORECASE,
    )

    def __init__(self):
        super().__init__('music_cleaner', 'music_cleaner')

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_featuring(self, artist: str) -> Tuple[str, Optional[str]]:
        """Return (main_artist, featured_artist_or_None)."""
        if pd.isna(artist):
            return artist, None
        m = self._FEAT_RE.search(str(artist))
        if m:
            feat_raw = m.group().strip()
            feat = re.sub(
                r'^(?:feat(?:uring)?\.?|ft\.?|with)\s+', '', feat_raw,
                flags=re.IGNORECASE,
            ).strip()
            return artist[:m.start()].strip(), feat or None
        return str(artist).strip(), None

    # ------------------------------------------------------------------
    # per-source cleaning
    # ------------------------------------------------------------------

    def _clean_billboard(self, df: pd.DataFrame) -> pd.DataFrame:
        df['chart_date'] = pd.to_datetime(df['chart_date'], errors='coerce')

        splits = df['artist'].apply(self._split_featuring)
        df['artist_main'] = splits.apply(lambda x: x[0])
        df['artist_feat'] = splits.apply(lambda x: x[1])

        df['title'] = df['title'].str.strip()
        df['artist_main'] = df['artist_main'].str.strip()

        for col in ['rank', 'weeks_on_chart', 'peak_position', 'last_week_rank']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'is_new' in df.columns:
            df['is_new'] = df['is_new'].fillna(False).astype(bool)

        # Drop rows missing core fields
        df = df.dropna(subset=['chart_date', 'rank', 'title', 'artist_main'])
        return df

    def _clean_lastfm(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['lastfm_play_count', 'lastfm_listeners', 'duration_ms']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['title'] = df['title'].str.strip()
        df['artist'] = df['artist'].str.strip()

        # Keep only tracks found on Last.fm (have at least one metric)
        has_data = df[['lastfm_play_count', 'lastfm_listeners']].notna().any(axis=1)
        df = df[has_data].copy()

        return df

    # ------------------------------------------------------------------
    # BaseTransformer interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().drop_duplicates()

        if 'chart_date' in df.columns:
            return self._clean_billboard(df)
        if 'lastfm_play_count' in df.columns:
            return self._clean_lastfm(df)

        self.logger.warning("MusicCleaner: unrecognised DataFrame shape — returning as-is")
        return df
