
from datasets import load_dataset

def get_dataset(name: str, split: str,  list_ids: dict):
    ds = load_dataset(name, split)

    selected_features = ds['train'].filter(lambda x: x['id'] in list_ids)

    return selected_features