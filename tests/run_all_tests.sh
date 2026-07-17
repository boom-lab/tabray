#!/bin/bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=data_sparsity --cov-report=html --cov-report=term-missing
