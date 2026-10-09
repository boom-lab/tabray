#!/bin/bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=tabray --cov-report=html --cov-report=term-missing
