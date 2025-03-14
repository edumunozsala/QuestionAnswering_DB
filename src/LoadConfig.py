import yaml

def read_file_descriptions(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        return data.get('files', [])

# Imprimir la lista de nombres de archivos y descripciones
#for file in file_descriptions:
#    print(f"Filename: {file['filename']}, Description: {file['description']}")

class LoadConfig:
    def __init__(self, yaml_file: str) -> None:
        self.yaml_file = yaml_file
        self.file_descriptions = read_file_descriptions(yaml_file)
    
    def get_file_descriptions(self) -> list:
        return self.file_descriptions