from torch.utils.data import DataLoader
import   numpy as np
import torch
from ..preprocessing.preprocess import CustomDataset, DatasetImporterCustom


def custom_collate_fn(batch):
    """
    Custom collate function to properly batch the data from the dataset.
    
    Args:
        batch: List of tuples (ts, sc) from the dataset
        
    Returns:
        A tuple of batched tensors (ts_batch, sc_batch)
    """
    try:
        # Separate the time series and static conditions
        ts, sc = zip(*batch)
        
        # Convert to numpy arrays if they aren't already
        ts = [t.astype('float32') if hasattr(t, 'astype') else np.array(t, dtype='float32') for t in ts]
        sc = [s.astype('float32') if hasattr(s, 'astype') else np.array(s, dtype='float32') for s in sc]
        
        # Convert to tensors
        ts_batch = torch.tensor(np.array(ts), dtype=torch.float32)
        sc_batch = torch.tensor(np.array(sc), dtype=torch.float32)
        
        return ts_batch, sc_batch
    except Exception as e:
        print(f"Error in custom_collate_fn: {str(e)}")
        print(f"Batch contents: {batch}")
        raise e


def build_custom_data_pipeline(batch_size, dataset_importer:DatasetImporterCustom, config: dict, kind: str) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    # DataLoader
    if kind == 'train':
        custom_dataset = CustomDataset('train', dataset_importer)
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, 
                         shuffle=True, drop_last=False, pin_memory=True, 
                         collate_fn=custom_collate_fn)  # Use custom collate function
    elif kind == 'test':
        custom_dataset = CustomDataset('test', dataset_importer)
        return DataLoader(custom_dataset, batch_size, num_workers=num_workers, 
                         shuffle=False, drop_last=False, pin_memory=True,
                         collate_fn=custom_collate_fn)  # Use custom collate function
    else:
        raise ValueError
