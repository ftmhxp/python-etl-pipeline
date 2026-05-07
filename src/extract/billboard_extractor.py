"""
Billboard Hot 100 extractor for the music ETL pipeline.
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

import billboard

from .base_extractor import BaseExtractor


class BillboardExtractor(BaseExtractor):
    """Scrapes weekly Billboard Hot 100 chart history."""

    def __init__(self):
        super().__init__('billboard', 'billboard')
        self.chart_name = self.source_config.get('chart_name', 'hot-100')
        self.start_date = self.source_config.get('start_date', '2000-01-01')
        self.end_date = self.source_config.get('end_date', '2025-01-01')
        self.sleep_seconds = self.source_config.get('sleep_seconds', 1.5)
        self.output_file = self.source_config.get('output_file', 'billboard_hot100.csv')
        self.resume = self.source_config.get('resume', False)

    def _get_chart_dates(self) -> List[str]:
        """Generate one date per week (Saturdays) from start to end."""
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        # Billboard publishes on Saturdays; find first Saturday >= start
        days_ahead = (5 - start.weekday()) % 7
        current = start + timedelta(days=days_ahead)

        dates = []
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(weeks=1)
        return dates

    def extract(self) -> Dict[str, Any]:
        dates = self._get_chart_dates()
        self.logger.info(
            f"Fetching {len(dates)} weeks of Billboard {self.chart_name} "
            f"({self.start_date} to {self.end_date})"
        )

        rows = []
        failed_weeks = []
        output_path = self.raw_data_path / self.output_file
        total_saved = 0

        if not self.resume and output_path.exists():
            output_path.unlink()
            self.logger.info("Fresh run: cleared existing output file")

        for i, date in enumerate(dates):
            try:
                chart = billboard.ChartData(self.chart_name, date=date)
                for entry in chart:
                    rows.append({
                        'chart_date': chart.date,
                        'rank': entry.rank,
                        'title': entry.title,
                        'artist': entry.artist,
                        'weeks_on_chart': entry.weeks,
                        'peak_position': entry.peakPos,
                        'last_week_rank': entry.lastPos if entry.lastPos else None,
                        'is_new': entry.isNew,
                    })

                if (i + 1) % 50 == 0:
                    chunk = pd.DataFrame(rows)
                    chunk.to_csv(output_path, mode='a', header=not output_path.exists() or total_saved == 0, index=False)
                    total_saved += len(rows)
                    rows = []
                    self.logger.info(f"Progress: {i + 1}/{len(dates)} weeks fetched, {total_saved:,} entries saved")

                time.sleep(self.sleep_seconds)

            except Exception as e:
                self.logger.warning(f"Failed to fetch chart for {date}: {e}")
                failed_weeks.append(date)
                time.sleep(self.retry_delay)

        # flush any remaining rows
        if rows:
            chunk = pd.DataFrame(rows)
            chunk.to_csv(output_path, mode='a', header=not output_path.exists() or total_saved == 0, index=False)
            total_saved += len(rows)

        if total_saved == 0:
            raise RuntimeError("No Billboard data extracted — all weeks failed.")

        self.logger.info(f"Saved {total_saved:,} chart entries to {output_path}")
        df = pd.read_csv(output_path)

        return {
            'total_weeks_attempted': len(dates),
            'total_weeks_fetched': len(dates) - len(failed_weeks),
            'failed_weeks': len(failed_weeks),
            'total_entries': len(df),
            'output_file': str(output_path),
            'file_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2),
        }
