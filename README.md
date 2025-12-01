# data-science-course-klu-2025
# a simple Data-Science project for the course Data-Science @ KLU
#
# Commands:

test_script: python3 test_script --chunk-size 250000 --top 10 --min-airport-flights 1500

Chunk Size can be manipulated as well as the airport flights search restriction parameter.

# Pipelin-Model:

Schneller Test auf Sample:
baseline_model.py: python3 baseline_model.py --sample --nrows 50000 --model logreg

Random Forest auf Teilmenge:
python3 baseline_model.py --nrows 200000 --model rf --top 10

Vollständiger Lauf: python3 baseline_model.py --model logreg --test-size 0.2