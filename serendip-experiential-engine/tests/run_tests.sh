#!/bin/bash
# Unit testing for Serendip Experiential Engine Backend
cd "$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pytest -v tests/