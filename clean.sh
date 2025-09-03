#!/bin/bash
pip uninstall needle -y
rm -rf build/
rm -rf dist/
rm -f *.so
rm -f needle/*.so
rm -rf *.egg-info