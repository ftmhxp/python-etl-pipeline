"""
Audio feature enrichment transformer.

Fuzzy-matches Billboard+Last.fm tracks against the Spotify Kaggle dataset,
then builds the final wide dataframe ready for Kaggle upload and Supabase load.
"""

import re
import ast
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from rapidfuzz import process, fuzz

from .base_transformer import BaseTransformer
from ..config import config


AUDIO_COLS = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'liveness', 'speechiness',
    'loudness', 'tempo', 'key', 'mode', 'time_signature',
]

FINAL_COL_ORDER = [
    # Identity
    'title', 'artist_main', 'artist_feat',
    # Chart
    'chart_date', 'decade', 'rank', 'peak_position',
    'weeks_on_chart', 'last_week_rank', 'is_new',
    # Engineered chart features
    'chart_velocity', 'longevity_score', 'peak_ratio',
    # Last.fm
    'lastfm_play_count', 'lastfm_listeners',
    'plays_per_listener', 'popularity_score', 'tags',
    # Audio features (Spotify)
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'liveness', 'speechiness',
    'loudness', 'tempo', 'key', 'mode', 'time_signature',
    'duration_ms', 'explicit',
    # Match metadata
    'audio_match_type', 'audio_match_score',
]


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_artists(raw: str) -> str:
    """Convert Spotify's "['Artist A', 'Artist B']" to 'artist a'."""
    try:
        names = ast.literal_eval(raw)
        if isinstance(names, list):
            return _normalise(names[0])
    except Exception:
        pass
    return _normalise(raw)


class AudioEnricher(BaseTransformer):
    """Enriches processed music data with Spotify audio features."""

    def __init__(self):
        super().__init__('audio_enricher', 'audio_enricher')

        enricher_cfg = config.get('data_sources.audio_enricher', {})
        self.match_threshold = enricher_cfg.get('match_threshold', 85)
        spotify_filename = enricher_cfg.get('spotify_file', 'spotify_audio_features.csv')
        self.output_file = enricher_cfg.get('output_file', 'music_enriched.csv')
        self.final_export = enricher_cfg.get('final_export_file', 'music_dataset_final.csv')

        spotify_path = self.raw_data_path / spotify_filename
        if not spotify_path.exists():
            raise FileNotFoundError(
                f"Spotify features not found at {spotify_path}. "
                "Run 'python main.py music extract --source kaggle' first."
            )

        self.logger.info(f"Loading Spotify features from {spotify_path} ...")
        self._spotify = self._prepare_spotify(spotify_path)
        self.logger.info(f"Spotify index ready: {len(self._spotify):,} tracks")

    # ------------------------------------------------------------------
    # Spotify preparation
    # ------------------------------------------------------------------

    def _prepare_spotify(self, path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)

        # Normalise column names (different dataset versions use different names)
        df.columns = df.columns.str.lower().str.strip()
        rename = {'track_name': 'name', 'track_artist': 'artists',
                  'song_name': 'name', 'artist_name': 'artists'}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if 'name' not in df.columns or 'artists' not in df.columns:
            raise ValueError(
                "Spotify CSV must have 'name' (track) and 'artists' columns. "
                f"Found: {list(df.columns)}"
            )

        df['_key_title'] = df['name'].apply(_normalise)
        df['_key_artist'] = df['artists'].apply(_parse_artists)
        df['_match_key'] = df['_key_title'] + ' ||| ' + df['_key_artist']

        # Keep only relevant columns + the match keys
        keep = ['_match_key', '_key_title', '_key_artist', 'name', 'artists'] + \
               [c for c in AUDIO_COLS + ['duration_ms', 'explicit'] if c in df.columns]
        return df[keep].drop_duplicates(subset=['_match_key'])

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_track(self, title: str, artist: str) -> Tuple[Optional[pd.Series], str, int]:
        """Return (spotify_row, match_type, score)."""
        norm_title = _normalise(title)
        norm_artist = _normalise(artist)
        query_key = f"{norm_title} ||| {norm_artist}"

        # 1 — exact key match
        exact = self._spotify[self._spotify['_match_key'] == query_key]
        if not exact.empty:
            return exact.iloc[0], 'exact', 100

        # 2 — exact title, fuzzy artist
        title_matches = self._spotify[self._spotify['_key_title'] == norm_title]
        if not title_matches.empty:
            choices = title_matches['_key_artist'].tolist()
            result = process.extractOne(norm_artist, choices, scorer=fuzz.token_sort_ratio)
            if result and result[1] >= self.match_threshold:
                row = title_matches[title_matches['_key_artist'] == result[0]].iloc[0]
                return row, 'fuzzy_artist', result[1]

        # 3 — full key fuzzy match
        choices = self._spotify['_match_key'].tolist()
        result = process.extractOne(query_key, choices, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= self.match_threshold:
            row = self._spotify[self._spotify['_match_key'] == result[0]].iloc[0]
            return row, 'fuzzy', result[1]

        return None, 'no_match', 0

    # ------------------------------------------------------------------
    # Core enrichment
    # ------------------------------------------------------------------

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add audio feature columns to a track-level dataframe."""
        artist_col = 'artist_main' if 'artist_main' in df.columns else 'artist'

        audio_rows = []
        for _, row in df.iterrows():
            spotify_row, match_type, score = self._match_track(
                row['title'], row[artist_col]
            )
            entry = {'audio_match_type': match_type, 'audio_match_score': score}
            if spotify_row is not None:
                for col in AUDIO_COLS + ['duration_ms', 'explicit']:
                    if col in spotify_row.index:
                        entry[col] = spotify_row[col]
            audio_rows.append(entry)

        audio_df = pd.DataFrame(audio_rows, index=df.index)
        return pd.concat([df, audio_df], axis=1)

    # ------------------------------------------------------------------
    # Final dataset assembly
    # ------------------------------------------------------------------

    def build_final_dataset(self) -> pd.DataFrame:
        """
        Join enriched Billboard data with Last.fm data and produce the
        final wide dataset ready for Kaggle upload.
        """
        processed = self.processed_data_path

        # Load enriched billboard (produced by transform())
        enriched_path = processed / self.output_file
        if not enriched_path.exists():
            raise FileNotFoundError(
                f"Enriched billboard file not found at {enriched_path}. "
                "Run 'python main.py music enrich' first."
            )

        # Load processed lastfm (produced by MusicFeatureEngineer)
        lastfm_candidates = list(processed.glob('*lastfm*'))
        if not lastfm_candidates:
            raise FileNotFoundError("No processed Last.fm file found in data/processed/.")
        lastfm_path = lastfm_candidates[0]

        billboard_df = pd.read_csv(enriched_path)
        lastfm_df = pd.read_csv(lastfm_path)

        # Normalise join keys
        for df in (billboard_df, lastfm_df):
            df['_join_title'] = df['title'].apply(_normalise)

        artist_col = 'artist_main' if 'artist_main' in billboard_df.columns else 'artist'
        lastfm_artist_col = 'artist'

        billboard_df['_join_artist'] = billboard_df[artist_col].apply(_normalise)
        lastfm_df['_join_artist'] = lastfm_df[lastfm_artist_col].apply(_normalise)

        lastfm_cols = ['_join_title', '_join_artist',
                       'lastfm_play_count', 'lastfm_listeners',
                       'plays_per_listener', 'popularity_score', 'tags']
        lastfm_cols = [c for c in lastfm_cols if c in lastfm_df.columns]

        final = billboard_df.merge(
            lastfm_df[lastfm_cols].drop_duplicates(subset=['_join_title', '_join_artist']),
            on=['_join_title', '_join_artist'],
            how='left',
        ).drop(columns=['_join_title', '_join_artist'], errors='ignore')

        # Reorder columns — keep only those that exist
        ordered = [c for c in FINAL_COL_ORDER if c in final.columns]
        remaining = [c for c in final.columns if c not in ordered]
        final = final[ordered + remaining]

        export_path = self.processed_data_path.parent / 'output' / self.final_export
        export_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(export_path, index=False)
        self.logger.info(
            f"Final dataset saved: {len(final):,} rows, {len(final.columns)} columns -> {export_path}"
        )
        return final

    # ------------------------------------------------------------------
    # BaseTransformer interface
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich a billboard dataframe with audio features."""
        if 'rank' not in df.columns:
            self.logger.warning("AudioEnricher expects billboard data — skipping.")
            return df

        self.logger.info(f"Matching {len(df):,} chart entries against Spotify ...")
        enriched = self._enrich(df)

        matched = (enriched['audio_match_type'] != 'no_match').sum()
        pct = matched / len(enriched) * 100
        self.logger.info(
            f"Match results: {matched:,}/{len(enriched):,} matched ({pct:.1f}%) — "
            f"exact: {(enriched['audio_match_type']=='exact').sum():,}, "
            f"fuzzy: {(enriched['audio_match_type']=='fuzzy').sum():,}, "
            f"no_match: {(enriched['audio_match_type']=='no_match').sum():,}"
        )
        return enriched

    def run(self, input_files):
        """Override to use the enriched output filename."""
        result = super().run(input_files)
        # Rename output to the configured name (not the default transformed_ prefix)
        if result.get('status') == 'success':
            for f in result.get('files', []):
                src = self.processed_data_path / f['output_file'].split('/')[-1].split('\\')[-1]
                dst = self.processed_data_path / self.output_file
                if src.exists() and src != dst:
                    src.rename(dst)
        return result
