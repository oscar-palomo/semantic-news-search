"""Shared constants. Defaults mirror the original notebook so results are reproducible."""

RANDOM_STATE = 42

DATASET_NAME = "fancyzhx/ag_news"
MODEL_NAME = "all-MiniLM-L6-v2"

# AG News integer label -> human-readable category
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

N_SAMPLES = 3000
N_CLUSTERS = len(LABEL_NAMES)
BATCH_SIZE = 64
