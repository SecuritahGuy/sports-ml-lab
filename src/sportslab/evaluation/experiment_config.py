"""Shared experiment configuration constants.

All experiments should import HOLDOUT_SEASON and ROLLING_FOLDS from here
to maintain consistency when expanding or modifying the season range.
"""

HOLDOUT_SEASON = 2025

# All available seasons (2021–2024 for training/validation)
ALL_SEASONS = [2021, 2022, 2023, 2024]

# Rolling-origin folds: each fold trains on all seasons before the val season.
# Validation seasons are 2022, 2023, 2024 (3 folds).
ROLLING_FOLDS = [
    ([s for s in ALL_SEASONS if s < 2022], 2022),
    ([s for s in ALL_SEASONS if s < 2023], 2023),
    ([s for s in ALL_SEASONS if s < 2024], 2024),
]
