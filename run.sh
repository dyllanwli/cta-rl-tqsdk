#!/bin/bash

# run main.py and redirect output to logs with date
cd src; python main.py > ../logs/$(date +%Y%m%d_%H%M%S).txt