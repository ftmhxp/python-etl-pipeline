"""
Extractor orchestrator for coordinating data extraction from multiple sources.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from tqdm import tqdm

from .csv_extractor import CSVExtractor
from .api_extractor import APIExtractor
from ..config import config


class ExtractorOrchestrator:
    """Orchestrates data extraction from multiple sources."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger(__name__)
        self.extractors = []

        # Create extractors based on configuration
        self._create_extractors()

    def _create_extractors(self):
        """Create extractor instances based on configuration."""
        data_sources = config.get("data_sources", {})

        for source_key, source_config in data_sources.items():
            if source_key == "jhu":
                # Johns Hopkins University data (CSV files)
                self.extractors.append(CSVExtractor("JHU", source_key))
            elif source_key == "owid":
                # Our World in Data (CSV file)
                self.extractors.append(CSVExtractor("OWID", source_key))
            elif source_key == "who":
                # World Health Organization (API)
                self.extractors.append(APIExtractor("WHO", source_key))
            else:
                self.logger.warning(f"Unknown data source: {source_key}")

        self.logger.info(f"Created {len(self.extractors)} extractors")

    def run_extraction(self, parallel: bool = True, max_workers: int = 3) -> Dict[str, Any]:
        """Run data extraction from all sources.

        Args:
            parallel: Whether to run extractors in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary containing extraction results
        """
        self.logger.info("Starting data extraction process")
        self.logger.info(f"Running {'parallel' if parallel else 'sequential'} extraction")

        results = []
        total_start_time = __import__('time').time()

        if parallel and len(self.extractors) > 1:
            # Run extractors in parallel
            results = self._run_parallel_extraction(max_workers)
        else:
            # Run extractors sequentially
            results = self._run_sequential_extraction()

        total_end_time = __import__('time').time()
        total_duration = total_end_time - total_start_time

        # Summarize results
        summary = self._create_summary(results, total_duration)

        self.logger.info(".2f")
        return summary

    def _run_sequential_extraction(self) -> List[Dict[str, Any]]:
        """Run extractors sequentially.

        Returns:
            List of extraction results
        """
        results = []

        for extractor in tqdm(self.extractors, desc="Extracting data", unit="source"):
            result = extractor.run()
            results.append(result)

        return results

    def _run_parallel_extraction(self, max_workers: int) -> List[Dict[str, Any]]:
        """Run extractors in parallel.

        Args:
            max_workers: Maximum number of parallel workers

        Returns:
            List of extraction results
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            future_to_extractor = {
                executor.submit(extractor.run): extractor
                for extractor in self.extractors
            }

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_extractor),
                             total=len(self.extractors),
                             desc="Extracting data",
                             unit="source"):
                extractor = future_to_extractor[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Extractor {extractor.source_name} failed: {e}")
                    results.append({
                        "source": extractor.source_name,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": __import__('time').time()
                    })

        return results

    def _create_summary(self, results: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
        """Create summary of extraction results.

        Args:
            results: List of individual extraction results
            total_duration: Total extraction duration

        Returns:
            Summary dictionary
        """
        successful_extractions = [r for r in results if r.get("status") == "success"]
        failed_extractions = [r for r in results if r.get("status") == "failed"]

        total_files = sum(r.get("total_files", 0) for r in successful_extractions)
        valid_files = sum(r.get("valid_files", 0) for r in successful_extractions)

        summary = {
            "total_duration_seconds": total_duration,
            "total_sources": len(results),
            "successful_sources": len(successful_extractions),
            "failed_sources": len(failed_extractions),
            "total_files_downloaded": total_files,
            "valid_files": valid_files,
            "results": results,
            "overall_status": "success" if len(failed_extractions) == 0 else "partial_success" if successful_extractions else "failed"
        }

        return summary
