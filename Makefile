# Makefile
.PHONY: help install dev test clean format lint docs run

help:
	@echo "Commands disponibili:"
	@echo "  install     : Installa dipendenze"
	@echo "  dev         : Installa in modalità sviluppo"
	@echo "  test        : Esegui test"
	@echo "  clean       : Pulisci file temporanei"
	@echo "  format      : Formatta codice con black"
	@echo "  lint        : Controlla stile con flake8"
	@echo "  run-example : Esegui esempio 01"

install:
	pip install -r requirements.txt

dev:
	pip install -e .

test:
	pytest tests/ -v --cov=src

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +

format:
	black src/ tests/ examples/

lint:
	flake8 src/ tests/ examples/

docs:
	@echo "Genera documentazione..."
	# Aggiungi comandi per Sphinx qui

run-example:
	python examples/01_ecommerce_customer.py

docker-build:
	docker build -t ecommerce-cleaner .

docker-run:
	docker run -it --rm ecommerce-cleaner
