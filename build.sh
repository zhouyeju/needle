#!/bin/bash

python setup.py build_ext
cp build/*.so needle
python setup.py bdist_wheel
pip install dist/*.whl