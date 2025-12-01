"""
Instance Retrieval Datasets: Revisited Oxford (ROxford) and Revisited Paris (RParis)
"""
import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RevisitedOxfordParisDataset(Dataset):
    """
    Revisited Oxford5k and Paris6k datasets for instance retrieval.
    
    References:
    - Radenović et al., "Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking", CVPR 2018
    """
    
    def __init__(self, root, dataset_name='roxford5k', input_size=224, transform=None):
        """
        Args:
            root: Path to dataset directory (containing images/ and gnd_*.pkl)
            dataset_name: 'roxford5k' or 'rparis6k'
            input_size: Image size for resizing
            transform: Optional additional transforms
        """
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.input_size = input_size
        
        # Load ground truth
        gnd_file = os.path.join(root, f'gnd_{self.dataset_name}.pkl')
        if not os.path.exists(gnd_file):
            raise FileNotFoundError(f"Ground truth file not found: {gnd_file}")
        
        with open(gnd_file, 'rb') as f:
            self.gnd = pickle.load(f)
        
        # Get image list
        self.images = self.gnd['imlist']
        self.queries = self.gnd['qimlist']
        self.query_gnd = self.gnd['gnd']
        
        # Image directory
        self.image_dir = os.path.join(root, 'images')
        if not os.path.exists(self.image_dir):
            # Try without 'images' subdirectory (some datasets have images directly in root)
            self.image_dir = root
        
        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print(f"✓ {dataset_name} dataset loaded:")
        print(f"  - Database images: {len(self.images)}")
        print(f"  - Query images: {len(self.queries)}")
    
    def __len__(self):
        return len(self.images)
    
    def get_image_path(self, img_name):
        """Get full path to image"""
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            path = os.path.join(self.image_dir, img_name + ext)
            if os.path.exists(path):
                return path
        
        # For Paris, images might be in subdirectories
        if 'paris' in self.dataset_name:
            for subdir in os.listdir(self.image_dir):
                subdir_path = os.path.join(self.image_dir, subdir)
                if os.path.isdir(subdir_path):
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        path = os.path.join(subdir_path, img_name + ext)
                        if os.path.exists(path):
                            return path
        
        return None
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.get_image_path(img_name)
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, idx
    
    def get_query(self, query_idx):
        """Get query image with bounding box"""
        query_name = self.queries[query_idx]
        img_path = self.get_image_path(query_name)
        
        if img_path is None:
            raise FileNotFoundError(f"Query image not found: {query_name}")
        
        image = Image.open(img_path).convert('RGB')
        
        # Get bounding box (if available)
        bbox = self.query_gnd[query_idx].get('bbx', None)
        
        if bbox is not None:
            # Crop to bounding box
            x1, y1, x2, y2 = [int(b) for b in bbox]
            image = image.crop((x1, y1, x2, y2))
        
        image = self.transform(image)
        
        return image, query_idx
    
    def get_query_ground_truth(self, query_idx):
        """
        Get ground truth for a query.
        
        Returns:
            easy: List of easy positive indices
            hard: List of hard positive indices
            junk: List of junk indices (to be ignored)
        """
        gnd = self.query_gnd[query_idx]
        easy = gnd.get('easy', [])
        hard = gnd.get('hard', [])
        junk = gnd.get('junk', [])
        
        return easy, hard, junk
    
    def get_num_queries(self):
        return len(self.queries)


class QueryDataset(Dataset):
    """Dataset for query images only"""
    
    def __init__(self, retrieval_dataset):
        self.retrieval_dataset = retrieval_dataset
        self.num_queries = retrieval_dataset.get_num_queries()
    
    def __len__(self):
        return self.num_queries
    
    def __getitem__(self, idx):
        return self.retrieval_dataset.get_query(idx)


def compute_ap(ranks, num_positives):
    """
    Compute Average Precision for a single query.
    
    Args:
        ranks: Sorted list of ranks where positives appear (0-indexed)
        num_positives: Total number of positive images
    
    Returns:
        AP score
    """
    if num_positives == 0:
        return 0.0
    
    ap = 0.0
    for i, rank in enumerate(ranks):
        precision_at_i = (i + 1) / (rank + 1)
        ap += precision_at_i
    
    ap /= num_positives
    return ap


def evaluate_retrieval(query_features, db_features, dataset, protocol='medium'):
    """
    Evaluate retrieval performance.
    
    Args:
        query_features: (num_queries, dim) tensor of query features
        db_features: (num_db, dim) tensor of database features
        dataset: RevisitedOxfordParisDataset instance
        protocol: 'easy', 'medium', or 'hard'
    
    Returns:
        mAP: Mean Average Precision
        recalls: Dict of recall@k values
    """
    num_queries = query_features.shape[0]
    
    # Normalize features
    query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
    db_features = torch.nn.functional.normalize(db_features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.mm(query_features, db_features.t())  # (num_queries, num_db)
    
    # Get rankings
    _, rankings = torch.sort(similarity, dim=1, descending=True)
    rankings = rankings.cpu().numpy()
    
    aps = []
    recalls_at_k = {1: [], 5: [], 10: []}
    
    for q_idx in range(num_queries):
        easy, hard, junk = dataset.get_query_ground_truth(q_idx)
        
        # Define positives based on protocol
        if protocol == 'easy':
            positives = set(easy)
        elif protocol == 'medium':
            positives = set(easy + hard)
        else:  # hard
            positives = set(hard)
        
        junk_set = set(junk)
        
        if len(positives) == 0:
            continue
        
        # Filter out junk from rankings
        filtered_ranks = []
        pos_ranks = []
        
        for rank, db_idx in enumerate(rankings[q_idx]):
            if db_idx in junk_set:
                continue
            
            if db_idx in positives:
                pos_ranks.append(len(filtered_ranks))
            
            filtered_ranks.append(db_idx)
        
        # Compute AP
        ap = compute_ap(pos_ranks, len(positives))
        aps.append(ap)
        
        # Compute recall@k
        for k in recalls_at_k.keys():
            top_k = set(filtered_ranks[:k])
            recall = len(top_k & positives) / len(positives)
            recalls_at_k[k].append(recall)
    
    mAP = np.mean(aps) * 100 if aps else 0.0
    mean_recalls = {k: np.mean(v) * 100 for k, v in recalls_at_k.items()}
    
    return mAP, mean_recalls


def get_retrieval_dataset(dataset_name, data_path, input_size=224):
    """
    Get retrieval dataset by name.
    
    Args:
        dataset_name: 'roxford5k' or 'rparis6k'
        data_path: Root path to dataset
        input_size: Image size
    
    Returns:
        Database dataset, Query dataset
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['roxford5k', 'oxford5k', 'oxford']:
        db_dataset = RevisitedOxfordParisDataset(data_path, 'roxford5k', input_size)
    elif dataset_name in ['rparis6k', 'paris6k', 'paris']:
        db_dataset = RevisitedOxfordParisDataset(data_path, 'rparis6k', input_size)
    else:
        raise ValueError(f"Unknown retrieval dataset: {dataset_name}")
    
    query_dataset = QueryDataset(db_dataset)
    
    return db_dataset, query_dataset


if __name__ == '__main__':
    # Test Oxford5k
    print("\n=== Testing ROxford5k ===")
    oxford_path = "/home/user/zhoutianjian/DataSets/Oxford5k"
    try:
        db_dataset, query_dataset = get_retrieval_dataset('roxford5k', oxford_path)
        print(f"Database: {len(db_dataset)} images")
        print(f"Queries: {len(query_dataset)} images")
        
        # Test loading
        img, idx = db_dataset[0]
        print(f"DB image shape: {img.shape}")
        
        q_img, q_idx = query_dataset[0]
        print(f"Query image shape: {q_img.shape}")
        
        easy, hard, junk = db_dataset.get_query_ground_truth(0)
        print(f"Query 0 GT: easy={len(easy)}, hard={len(hard)}, junk={len(junk)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test Paris6k
    print("\n=== Testing RParis6k ===")
    paris_path = "/home/user/zhoutianjian/DataSets/Paris6k"
    try:
        db_dataset, query_dataset = get_retrieval_dataset('rparis6k', paris_path)
        print(f"Database: {len(db_dataset)} images")
        print(f"Queries: {len(query_dataset)} images")
    except Exception as e:
        print(f"Error: {e}")

