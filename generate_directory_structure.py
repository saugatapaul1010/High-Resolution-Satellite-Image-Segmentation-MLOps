import os

# Define the folder structure as a nested dictionary.
# A value of None indicates an empty file; a nested dict indicates a directory.
structure = {
    "data": {
        "raw": {},
        "processed": {},
        "train": {
            "train_images": {
                "train": {}
            },
            "train_masks": {
                "train": {}
            }
        },
        "val": {
            "val_images": {
                "val": {}
            },
            "val_masks": {
                "val": {}
            }
        },
    },
    "models": {},
    "src": {
        "data_preparation.py": None,
        "dataset_generator.py": None,
        "metrics.py": None,
        "model_trainer.py": None,
        "config.py": None,
        "constants.py": None,
        "main.py": None,
        "train.py": None,
    },
    "config": {
        "config.yaml": None,
        "hyperparameters.yaml": None,
    },
    "tests": {
        "test_metrics.py": None,
        "test_data_preparation.py": None,
        "test_model_trainer.py": None,
    },
    ".dvc": {},
    ".github": {
        "workflows": {
            "ci.yaml": None,
        },
    },
    "sonar-project.properties": None,
    "dvc.yaml": None,
    "requirements.txt": None,
    "Dockerfile": None,
    "README.md": None,
}

def create_structure(base_path: str, tree: dict):
    """
    Recursively creates directories and files from the nested dictionary.
    
    Args:
        base_path (str): The base directory where creation begins.
        tree (dict): The nested dictionary representing the folder structure.
    """
    for name, subtree in tree.items():
        current_path = os.path.join(base_path, name)
        if subtree is None:
            # Create an empty file.
            with open(current_path, 'w') as f:
                f.write("")  # Empty file
            print(f"Created file: {current_path}")
        else:
            # Create the directory (if it doesn't exist), then recursively create its contents.
            os.makedirs(current_path, exist_ok=True)
            print(f"Created directory: {current_path}")
            create_structure(current_path, subtree)

if __name__ == "__main__":
    # Get the current directory where the script is run
    current_directory = os.getcwd()
    create_structure(current_directory, structure)
