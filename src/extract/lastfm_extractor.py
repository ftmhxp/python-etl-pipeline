"""
Last.fm API extractor for track metadata enrichment.
"""

import os
import time
import pandas as pd
from typing import Dict, Any, Optional

import pylast
from dotenv import load_dotenv

from .base_extractor import BaseExtractor


class LastFmExtractor(BaseExtractor):
    """Enriches Billboard tracks with Last.fm metadata."""

    def __init__(self):
        super().__init__('lastfm', 'lastfm')
        load_dotenv()

        api_key_env = self.source_config.get('api_key_env', 'LASTFM_API_KEY')
        api_secret_env = self.source_config.get('api_secret_env', 'LASTFM_API_SECRET')
        self.api_key = os.environ.get(api_key_env)
        self.api_secret = os.environ.get(api_secret_env)

        if not self.api_key or self.api_key == 'your_api_key_here':
            raise ValueError(
                f"Last.fm API key not set. Add {api_key_env} to your .env file. "
                "Register at: https://www.last.fm/api/account/create"
            )

        self.rate_limit_sleep = self.source_config.get('rate_limit_sleep', 0.2)
        self.billboard_input = self.source_config.get('billboard_input', 'billboard_hot100.csv')
        self.output_file = self.source_config.get('output_file', 'lastfm_track_data.csv')

        self.network = pylast.LastFMNetwork(
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

    def _unique_tracks(self) -> pd.DataFrame:
        input_path = self.raw_data_path / self.billboard_input
        if not input_path.exists():
            raise FileNotFoundError(
                f"Billboard data not found at {input_path}. "
                "Run 'python main.py music extract --source billboard' first."
            )
        df = pd.read_csv(input_path, usecols=['title', 'artist'])
        unique = df.drop_duplicates().reset_index(drop=True)
        self.logger.info(f"Found {len(unique):,} unique tracks to look up on Last.fm")
        return unique

    def _fetch_track(self, title: str, artist: str) -> Dict[str, Any]:
        base = {'title': title, 'artist': artist}
        try:
            track = self.network.get_track(artist, title)

            def safe(fn):
                try:
                    return fn()
                except Exception:
                    return None

            top_tags = safe(lambda: track.get_top_tags(limit=5)) or []
            tag_names = ','.join(t.item.get_name() for t in top_tags) if top_tags else None

            return {
                **base,
                'lastfm_play_count': safe(track.get_playcount),
                'lastfm_listeners': safe(track.get_listener_count),
                'duration_ms': safe(track.get_duration),
                'tags': tag_names,
                'lastfm_artist_mbid': safe(lambda: self.network.get_artist(artist).get_mbid()),
                'lastfm_found': True,
            }

        except pylast.WSError:
            return {**base, 'lastfm_play_count': None, 'lastfm_listeners': None,
                    'duration_ms': None, 'tags': None, 'lastfm_artist_mbid': None,
                    'lastfm_found': False}

    def extract(self) -> Dict[str, Any]:
        unique_tracks = self._unique_tracks()

        results = []
        found = 0

        for i, row in unique_tracks.iterrows():
            info = self._fetch_track(row['title'], row['artist'])
            results.append(info)
            if info['lastfm_found']:
                found += 1

            if (i + 1) % 200 == 0:
                self.logger.info(
                    f"Progress: {i + 1}/{len(unique_tracks)} tracks fetched "
                    f"({found} found on Last.fm)"
                )

            time.sleep(self.rate_limit_sleep)

        df = pd.DataFrame(results)
        output_path = self.raw_data_path / self.output_file
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(df):,} track records to {output_path}")

        return {
            'total_tracks': len(unique_tracks),
            'found_on_lastfm': found,
            'not_found_on_lastfm': len(unique_tracks) - found,
            'output_file': str(output_path),
            'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2),
        }
