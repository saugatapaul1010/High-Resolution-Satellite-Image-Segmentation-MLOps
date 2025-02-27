import os

# Define the folder structure
folder_structure = {
    "config": [],  # Configuration files (YAML)
    "data": [],  # Raw data (tracked by DVC)
    "src": [
        "__init__.py",
        "data_preparation.py",
        "dataset_generator.py",
        "metrics.py",
        "model_trainer.py",
        "model.py",
        "utils.py"
    ],
    "src/api": [
        "__init__.py",
        "main.py",
        "schemas.py"
    ],
    "tests": [
        "__init__.py",
        "test_data_preparation.py",
        "test_model_trainer.py"
    ],
    "mlflow": [],  # MLflow artifacts (not directly tracked)
    "kubernetes": [
        "deployment.yaml",
        "service.yaml"
    ],
    ".github/workflows": [
        "ci.yaml"
    ]
}

# Define files at the root level
root_files = [
    "dvc.yaml",
    "requirements.txt",
    "Dockerfile"
]

# Function to create directories and files
def create_structure():
    base_path = os.getcwd()  # Get the current working directory

    for folder, files in folder_structure.items():
        folder_path = os.path.join(base_path, folder)

        # ✅ Check if it's a file instead of a folder and remove it
        if os.path.exists(folder_path) and not os.path.isdir(folder_path):
            print(f"❌ Error: {folder_path} exists as a file. Deleting it...")
            os.remove(folder_path)
        
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):  # Avoid overwriting existing files
                with open(file_path, "w") as f:
                    f.write("")  # Create an empty file

    # Create root-level files
    for file in root_files:
        file_path = os.path.join(base_path, file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")  # Create an empty file

    print("✅ Project structure created successfully!")

# Run the function
if __name__ == "__main__":
    create_structure()
