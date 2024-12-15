import sys
import os
sys.path.append('..')
import zipfile
import json
from typing import Dict, Any


class ResponseQAloader:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def _extract_zip(self, zip_path: str) -> Dict[str, Any]:
    
        extracted_data = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_name in z.namelist():
                if file_name.endswith('.json'):
                    with z.open(file_name) as f:
                        try:
                            json_data = json.load(f)
                            extracted_data.append(json_data)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON in {file_name}")
        return extracted_data

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        
        data_dict = {}

        # Process Training and Validation directories
        for data_type in ['Training', 'Validation']:
            path_to_labeling_data = os.path.join(self.base_path, data_type, '02.라벨링데이터')

            if not os.path.exists(path_to_labeling_data):
                print(f"Warning: Path not found - {path_to_labeling_data}")
                continue

            # Iterate through .zip files in the directory
            for zip_file in os.listdir(path_to_labeling_data):
                if zip_file.endswith('.zip'):
                    # Parse TL prefix and topic key
                    parts = zip_file.split('_', 2)
                    if len(parts) < 3:
                        print(f"Warning: Invalid zip file name format - {zip_file}")
                        continue

                    tl_prefix = parts[0]  # e.g., TL_1
                    topic_key = parts[2].split('.')[0]  # e.g., 인문사회
                    
                    zip_path = os.path.join(path_to_labeling_data, zip_file)
                    print(f"Processing {zip_path}...")

                    # Extract and process JSON files
                    json_data = self._extract_zip(zip_path)

                    # Store in nested dictionary
                    if tl_prefix not in data_dict:
                        data_dict[tl_prefix] = {}
                    data_dict[tl_prefix][topic_key] = json_data

        return data_dict





if __name__ == "__main__":
    # Example usage
    loader = ResponseQAloader(base_path="projectdataset/responsedata")
    dataset = loader.load_data()
    
    # Print summary of loaded data
    for tl_key, topics in dataset.items():
        print(f"Category: {tl_key}")
        summed = 0
        for topic, files in topics.items():
            print(f"  - Topic: {topic}, Files Loaded: {len(files)}")
            summed += len(files)
        print(f'\n category size : {summed}')
    # print(dataset['VL'].keys())
