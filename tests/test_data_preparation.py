from src.initialize import initialize
from src.data_preparation import DataPreparation

initialize()  # Set up Constants
dp = DataPreparation(None)
dp.prepare_patches()  # Run data preparation