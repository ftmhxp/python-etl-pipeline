"""
Kaggle dataset extractor — downloads Spotify audio features.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from .base_extractor import BaseExtractor


class KaggleExtractor(BaseExtractor):
    """Downloads a Kaggle dataset and stores it in the raw data directory."""

    def __init__(self):
        super().__init__('kaggle_spotify', 'kaggle_spotify')
        load_dotenv()

        self.dataset = self.source_config.get(
            'dataset', 'yamaerenay/spotify-dataset-19212020-600k-tracks'
        )
        self.source_filename = self.source_config.get('source_filename', 'tracks.csv')
        self.output_file = self.source_config.get('output_file', 'spotify_audio_features.csv')

        api_key_env = self.source_config.get('api_key_env', 'KAGGLE_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key or api_key == 'your_kaggle_key_here':
            raise ValueError(
                f"Kaggle API key not set. Add {api_key_env} to your .env file. "
                "Get your key at kaggle.com -> Settings -> API -> Create Token."
            )
        # Kaggle library v2 requires both KAGGLE_USERNAME and KAGGLE_KEY in env
        os.environ['KAGGLE_KEY'] = api_key
        username = os.environ.get('KAGGLE_USERNAME', '')
        if username:
            os.environ['KAGGLE_USERNAME'] = username

    def extract(self) -> Dict[str, Any]:
        # Import here so missing package gives a clear error at runtime
        try:
            from kaggle import KaggleApi
        except ImportError:
            raise ImportError("Run: pip install kaggle>=1.6.0")

        output_path = self.raw_data_path / self.output_file
        if output_path.exists():
            self.logger.info(f"Spotify features already downloaded at {output_path}, skipping.")
            return {
                'dataset': self.dataset,
                'output_file': str(output_path),
                'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2),
                'skipped': True,
            }

        self.logger.info(f"Downloading Kaggle dataset: {self.dataset}")

        api = KaggleApi()
        api.authenticate()

        tmp_dir = self.raw_data_path / '_kaggle_tmp'
        tmp_dir.mkdir(exist_ok=True)

        try:
            api.dataset_download_files(
                self.dataset,
                path=str(tmp_dir),
                unzip=True,
                quiet=False,
            )

            # Locate the target file — fall back to the first CSV if name differs
            candidate = tmp_dir / self.source_filename
            if not candidate.exists():
                csvs = sorted(tmp_dir.glob('*.csv'))
                if not csvs:
                    raise FileNotFoundError(
                        f"No CSV files found after downloading {self.dataset}. "
                        f"Check dataset slug in config.yaml."
                    )
                candidate = csvs[0]
                self.logger.warning(
                    f"Expected '{self.source_filename}', using '{candidate.name}' instead."
                )

            shutil.move(str(candidate), str(output_path))
            self.logger.info(f"Saved to {output_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            'dataset': self.dataset,
            'output_file': str(output_path),
            'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2),
            'skipped': False,
        }
