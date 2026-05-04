.PHONY: install test test-cov music-extract music-transform music-enrich music-export music-load music-run covid-extract covid-transform covid-load covid-run

install:
	pip install -r requirements.txt

# Testing
test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing

# Music pipeline
music-extract:
	python main.py music extract --source all

music-transform:
	python main.py music transform

music-enrich:
	python main.py music enrich

music-export:
	python main.py music export

music-load:
	python main.py music load

music-run:
	python main.py music run

# COVID pipeline
covid-extract:
	python main.py extract

covid-transform:
	python main.py transform

covid-load:
	python main.py load

covid-run:
	python main.py run
