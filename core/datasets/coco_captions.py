"""
COCO Captions Dataset for DINOtxt Training
Loads image-caption pairs from COCO 2017 dataset
"""
import torch
from torch.utils.data import Dataset
import json
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import gzip
import urllib.request


def get_clip_tokenizer(max_seq_len=77):
    """
    Get CLIP BPE tokenizer - downloads vocab if needed
    Returns a tokenizer function that converts text to tensor
    """
    # Try to use the CLIP tokenizer from dinov3
    try:
        import sys
        dinov3_path = Path(__file__).parent.parent.parent / "dinov3"
        if str(dinov3_path) not in sys.path:
            sys.path.insert(0, str(dinov3_path))
        
        from dinov3.thirdparty.CLIP.clip.simple_tokenizer import SimpleTokenizer as CLIPTokenizer
        
        # Check if BPE vocab file exists, download if not
        clip_dir = dinov3_path / "dinov3" / "thirdparty" / "CLIP" / "clip"
        bpe_path = clip_dir / "bpe_simple_vocab_16e6.txt.gz"
        
        if not bpe_path.exists():
            print("Downloading CLIP BPE vocabulary...")
            bpe_url = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
            
            # Setup proxy if available (common proxy settings)
            proxy_handler = urllib.request.ProxyHandler({
                'http': os.environ.get('http_proxy', os.environ.get('HTTP_PROXY', '')),
                'https': os.environ.get('https_proxy', os.environ.get('HTTPS_PROXY', '')),
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
            
            try:
                urllib.request.urlretrieve(bpe_url, str(bpe_path))
                print(f"✓ Downloaded BPE vocab to {bpe_path}")
            except Exception as download_err:
                print(f"⚠ Download failed: {download_err}")
                print("  Trying alternative URL...")
                # Try OpenAI's original CLIP vocab
                alt_url = "https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz"
                urllib.request.urlretrieve(alt_url, str(bpe_path))
                print(f"✓ Downloaded BPE vocab from alternative URL")
        
        tokenizer = CLIPTokenizer(bpe_path=str(bpe_path))
        
        def tokenize(text):
            """Tokenize text using CLIP BPE tokenizer"""
            sot_token = tokenizer.encoder["<|startoftext|>"]
            eot_token = tokenizer.encoder["<|endoftext|>"]
            
            tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
            
            # Truncate if needed
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                tokens[-1] = eot_token
            
            # Pad to max_seq_len
            result = torch.zeros(max_seq_len, dtype=torch.long)
            result[:len(tokens)] = torch.tensor(tokens)
            
            return result
        
        print("✓ Using CLIP BPE tokenizer (vocab_size=49408)")
        return tokenize, 49408
        
    except Exception as e:
        print(f"⚠ Failed to load CLIP tokenizer: {e}")
        print("  Falling back to simple character tokenizer")
        return None, None


class SimpleTokenizer:
    """
    Fallback simple character-level tokenizer
    Only used if CLIP tokenizer fails to load
    """
    def __init__(self, vocab_size=30522, max_seq_len=77):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Special tokens
        self.pad_token = 0
        self.cls_token = 1
        self.sep_token = 2
        self.unk_token = 3
        
        self.char_to_idx = {}
        self.build_vocab()
    
    def build_vocab(self):
        """Build a simple character vocabulary"""
        idx = 4
        for c in 'abcdefghijklmnopqrstuvwxyz':
            self.char_to_idx[c] = idx
            idx += 1
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.char_to_idx[c] = idx
            idx += 1
        for c in '0123456789':
            self.char_to_idx[c] = idx
            idx += 1
        for c in ' .,!?;:\'"()-':
            self.char_to_idx[c] = idx
            idx += 1
    
    def encode(self, text):
        tokens = [self.cls_token]
        for char in text.lower()[:self.max_seq_len - 2]:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.unk_token)
        tokens.append(self.sep_token)
        
        while len(tokens) < self.max_seq_len:
            tokens.append(self.pad_token)
        tokens = tokens[:self.max_seq_len]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def __call__(self, text):
        return self.encode(text)


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions Dataset
    
    Expected directory structure:
    data_path/
    ├── annotations/
    │   ├── captions_train2017.json
    │   └── captions_val2017.json
    ├── train2017/
    │   └── *.jpg
    └── val2017/
        └── *.jpg
    """
    
    def __init__(
        self,
                 data_path,
                 split='train',
                 input_size=224,
                 max_seq_len=77,
        vocab_size=30522,
        sample_ratio=None,
    ):
        """
        Args:
            data_path: Path to COCO dataset root
            split: 'train' or 'val'
            input_size: Target image size
            max_seq_len: Maximum sequence length for captions
            vocab_size: Vocabulary size for tokenizer
            sample_ratio: Use only a fraction of the dataset (0.0 to 1.0)
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.input_size = input_size
        self.max_seq_len = max_seq_len
        
        # Initialize tokenizer - try CLIP BPE first, fallback to simple
        clip_tokenize, clip_vocab_size = get_clip_tokenizer(max_seq_len)
        if clip_tokenize is not None:
            self.tokenizer = clip_tokenize
            self.vocab_size = clip_vocab_size
        else:
            self.tokenizer = SimpleTokenizer(vocab_size, max_seq_len)
            self.vocab_size = vocab_size
        
        # Load annotations
        if split == 'train':
            ann_file = self.data_path / 'annotations' / 'captions_train2017.json'
            self.image_dir = self.data_path / 'train2017'
        else:
            ann_file = self.data_path / 'annotations' / 'captions_val2017.json'
            self.image_dir = self.data_path / 'val2017'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        print(f"Loading COCO Captions {split} set from {ann_file}...")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Build image_id -> captions mapping
        self.image_id_to_captions = {}
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_captions:
                self.image_id_to_captions[img_id] = []
            self.image_id_to_captions[img_id].append(ann['caption'])
        
        # Build image_id -> filename mapping
        self.image_id_to_filename = {}
        for img in annotations['images']:
            self.image_id_to_filename[img['id']] = img['file_name']
        
        # Create list of (image_id, filename) pairs
        self.samples = []
        for img_id, filename in self.image_id_to_filename.items():
            if img_id in self.image_id_to_captions:
                self.samples.append((img_id, filename))
        
        # Apply sample ratio if specified
        if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            num_samples = max(1, num_samples)
            random.seed(42)  # For reproducibility
            self.samples = random.sample(self.samples, num_samples)
            print(f"Using {sample_ratio*100:.1f}% of data: {num_samples} samples")
        
        print(f"Loaded {len(self.samples)} image-caption pairs")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, filename = self.samples[idx]
        
        # Load image
        img_path = self.image_dir / filename
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.input_size, self.input_size)
        
        # Get random caption for this image
        captions = self.image_id_to_captions[img_id]
        caption = random.choice(captions)
        
        # Tokenize caption
        text_tokens = self.tokenizer(caption)
        
        return image, text_tokens
    
    def get_caption(self, idx):
        """Get the raw caption text for visualization"""
        img_id, _ = self.samples[idx]
        captions = self.image_id_to_captions[img_id]
        return captions[0]  # Return first caption


def test_dataset():
    """Test the dataset loader"""
    print("Testing COCO Captions Dataset...")
    
    # Test with actual COCO path
    data_path = "/home/user/zhoutianjian/DataSets/COCO 2017"
    
    try:
        # Test train set
        train_dataset = COCOCaptionsDataset(
            data_path=data_path,
            split='train',
            input_size=224,
            sample_ratio=0.001  # Use 0.1% for testing
        )
        
        print(f"\nTrain dataset size: {len(train_dataset)}")
        
        # Test loading a sample
        image, text_tokens = train_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Text tokens shape: {text_tokens.shape}")
        print(f"Text tokens range: [{text_tokens.min()}, {text_tokens.max()}]")
        
        # Test val set
        val_dataset = COCOCaptionsDataset(
            data_path=data_path,
            split='val',
            input_size=224,
        )
        
        print(f"\nVal dataset size: {len(val_dataset)}")
        
        print("\n✅ Dataset test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        return False


if __name__ == '__main__':
    test_dataset()
