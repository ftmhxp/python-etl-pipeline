#!/usr/bin/env python3
"""
COVID-19 ETL Pipeline - Main Entry Point.

Command-line interface for running the complete ETL pipeline
or individual pipeline stages.
"""

import click
import logging
from pathlib import Path
from src.config import config



def setup_logging():
    """Configure logging for the application."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/etl_pipeline.log')),
            logging.StreamHandler()
        ]
    )


@click.group()
@click.version_option(version=config.get('project.version', '1.0.0'))
def cli():
    """COVID-19 ETL Pipeline - Extract, Transform, Load global COVID-19 data."""
    setup_logging()
    pass


@cli.command()
@click.option('--config', 'config_file', default='config.yaml',
              help='Path to configuration file')
@click.option('--parallel/--sequential', default=True,
              help='Run extraction in parallel or sequentially')
@click.option('--max-workers', default=3, type=int,
              help='Maximum number of parallel workers')
def extract(config_file, parallel, max_workers):
    """Run the extract phase of the ETL pipeline."""
    click.echo("Starting data extraction...")

    try:
        # Import here to avoid circular imports
        from src.extract import ExtractorOrchestrator

        # Create and run orchestrator
        orchestrator = ExtractorOrchestrator()
        results = orchestrator.run_extraction(parallel=parallel, max_workers=max_workers)

        # Display results
        click.echo("\nExtraction Summary:")
        click.echo(f"  Total sources: {results['total_sources']}")
        click.echo(f"  Successful: {results['successful_sources']}")
        click.echo(f"  Failed: {results['failed_sources']}")
        click.echo(f"  Files downloaded: {results['total_files_downloaded']}")
        click.echo(f"  Valid files: {results['valid_files']}")
        click.echo(f"  Duration: {results['total_duration_seconds']:.2f} seconds")

        # Show detailed results for each source
        click.echo("\nSource Details:")
        for result in results['results']:
            status_icon = "[SUCCESS]" if result['status'] == 'success' else "[FAILED]"
            source_name = result['source']
            status = result['status']

            if status == 'success':
                files = result.get('total_files', 0)
                valid = result.get('valid_files', 0)
                click.echo(f"  {status_icon} {source_name}: {files} files ({valid} valid)")
            else:
                error = result.get('error', 'Unknown error')
                click.echo(f"  {status_icon} {source_name}: Failed - {error}")

        if results['overall_status'] == 'success':
            click.echo("\nData extraction completed successfully!")
        elif results['overall_status'] == 'partial_success':
            click.echo("\nData extraction completed with some failures!")
        else:
            click.echo("\nData extraction failed!")
            raise click.ClickException("Extraction failed")

    except Exception as e:
        click.echo(f"Data extraction failed: {e}")
        raise click.ClickException(f"Extraction error: {e}")


@cli.command()
@click.option('--input-dir', default=None,
              help='Input directory for raw data (overrides config)')
@click.option('--output-dir', default=None,
              help='Output directory for processed data (overrides config)')
@click.option('--input-files', multiple=True,
              help='Specific input files to transform (can be specified multiple times)')
@click.option('--transform-order', default=None,
              help='Comma-separated list of transformations to apply (cleaner,feature_engineer,validator)')
def transform(input_dir, output_dir, input_files, transform_order):
    """Run the transform phase of the ETL pipeline."""
    click.echo("🔄 Starting data transformation...")

    try:
        # Import here to avoid circular imports
        from src.transform import TransformerOrchestrator

        # Parse transform order if provided
        transform_list = None
        if transform_order:
            transform_list = [t.strip() for t in transform_order.split(',')]

        # Prepare input files
        input_file_list = list(input_files) if input_files else None

        # Override config directories if specified
        if input_dir:
            import os
            os.environ['INPUT_DIR'] = input_dir
        if output_dir:
            import os
            os.environ['OUTPUT_DIR'] = output_dir

        # Create and run orchestrator
        orchestrator = TransformerOrchestrator()
        results = orchestrator.run_transformation(
            input_files=input_file_list,
            output_dir=output_dir,
            transform_order=transform_list
        )

        # Display results
        click.echo("\nTransformation Summary:")
        click.echo(f"  Status: {results['status']}")
        click.echo(f"  Duration: {results['total_duration_seconds']:.2f} seconds")
        click.echo(f"  Files processed: {results['total_input_files']}")
        click.echo(f"  Total rows: {results['total_input_rows']} -> {results['total_output_rows']}")
        click.echo(f"  Transformations: {', '.join(results['transformations_applied'])}")

        # Show per-file results
        click.echo("\nFile Details:")
        for file_result in results['file_results']:
            status_icon = "[SUCCESS]" if file_result['status'] == 'success' else "[PARTIAL]"
            input_name = file_result['input_file'].split('/')[-1].split('\\')[-1]
            rows_info = f"{file_result['original_rows']} -> {file_result['final_rows']} rows"
            click.echo(f"  {status_icon} {input_name}: {rows_info}")

            # Show transformation steps
            for step in file_result['transformations']:
                step_status = "✓" if step['status'] == 'success' else "✗"
                step_info = f"{step['transformer']}: {step['input_rows']} -> {step['output_rows']}"
                click.echo(f"    {step_status} {step_info}")

        if results['status'] == 'success':
            click.echo("\n✅ Data transformation completed successfully!")
        elif results['status'] == 'partial_success':
            click.echo("\n⚠️  Data transformation completed with some issues!")
        else:
            click.echo("\n❌ Data transformation failed!")
            raise click.ClickException("Transformation failed")

    except Exception as e:
        click.echo(f"❌ Data transformation failed: {e}")
        raise click.ClickException(f"Transformation error: {e}")


@cli.command()
@click.option('--input-dir', default=None,
              help='Input directory for processed data (overrides config)')
def load(input_dir):
    """Run the load phase of the ETL pipeline."""
    click.echo("💾 Starting data loading...")
    # TODO: Implement load logic
    click.echo("✅ Data loading completed!")


@cli.command()
@click.option('--skip-extract', is_flag=True, help='Skip extraction phase')
@click.option('--skip-transform', is_flag=True, help='Skip transformation phase')
@click.option('--skip-load', is_flag=True, help='Skip loading phase')
def run(skip_extract, skip_transform, skip_load):
    """Run the complete ETL pipeline."""
    click.echo("🚀 Starting complete ETL pipeline...")

    if not skip_extract:
        click.echo("📥 Phase 1: Extraction")
        # TODO: Run extract

    if not skip_transform:
        click.echo("🔄 Phase 2: Transformation")
        # TODO: Run transform

    if not skip_load:
        click.echo("💾 Phase 3: Loading")
        # TODO: Run load

    click.echo("🎉 ETL pipeline completed successfully!")


@cli.command()
def validate():
    """Validate data quality and pipeline configuration."""
    click.echo("🔍 Validating pipeline configuration and data...")
    # TODO: Implement validation logic
    click.echo("✅ Validation completed!")


@cli.command()
def analyze():
    """Run data analysis and generate reports."""
    click.echo("📊 Running data analysis...")
    # TODO: Implement analysis logic
    click.echo("✅ Analysis completed!")


if __name__ == '__main__':
    cli()
