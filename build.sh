#!/bin/bash
MODE=$1

python setup.py build_ext
cp build/*.so needle

if [[ "$MODE" == "edit" ]]; then
    pip install -e .
else
    python setup.py bdist_wheel
    pip install dist/*.whl
fi