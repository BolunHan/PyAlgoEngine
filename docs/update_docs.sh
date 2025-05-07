#!/bin/bash
sphinx-apidoc -o docs/source/ algo_engine/ -f
cd docs
make clean
make html