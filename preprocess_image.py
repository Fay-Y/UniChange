import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import re

def natural_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]

def load_image_pil(image_path):
    image_pil = Image.open(image_path).convert('RGB')
    original_image = np.array(image_pil)  # (H, W, C)
    return original_image

def normalize_image(image_np):

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()  # (C, H, W)
    normalized_tensor = image_tensor / 255.0
    normalized_tensor = normalized_tensor * 2.0 - 1.0
    
    return normalized_tensor

def clip_feature_single(image_tensor, clip_model, clip_preprocess, device):

    image_denorm = (image_tensor + 1.0) / 2.0 * 255.0
    image_np = image_denorm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    image_pil = Image.fromarray(image_np)
    if image_pil.mode == 'L' or image_pil.mode == 'LA':
        image_pil = image_pil.convert('RGB')
    with torch.no_grad():
        clip_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
        clip_feature = clip_model.encode_image(clip_input)
    
    return clip_feature

def get_image_name_from_filename(filename):
    return os.path.splitext(filename)[0]

def process_dataset(data_root, output_root, device='cuda'):

    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.to(torch.float32).to(device)
    clip_model.eval()
    
    os.makedirs(output_root, exist_ok=True)
    
    splits = ['train', 'test', 'val']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping...")
            continue
        
        a_path = os.path.join(split_path, 'A')
        b_path = os.path.join(split_path, 'B')
        label_path = os.path.join(split_path, 'label')
        
        if not all(os.path.exists(p) for p in [a_path, b_path, label_path]):
            print(f"Warning: Missing A, B, or label folder in {split_path}, skipping...")
            continue
        
        filenames = sorted(os.listdir(a_path), key=natural_sort_key)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(filenames)} images in {split}")
        
        data_items = {}
        label_items = {}
        
        for filename in tqdm(filenames, desc=f"Processing {split}"):
            try:
                a_file = os.path.join(a_path, filename)
                b_file = os.path.join(b_path, filename)
                label_file = os.path.join(label_path, filename)
                
                if not all(os.path.exists(f) for f in [a_file, b_file, label_file]):
                    print(f"Warning: Missing files for {filename}, skipping...")
                    continue
                
                image_name = get_image_name_from_filename(filename)
                
                image_a = load_image_pil(a_file)  # (H, W, C)
                image_b = load_image_pil(b_file)  # (H, W, C)
                image_label = load_image_pil(label_file)  # (H, W, C)
                
                normalized_a = normalize_image(image_a)  # (C, H, W), [-1, 1]
                normalized_b = normalize_image(image_b)  # (C, H, W), [-1, 1]
                normalized_label = normalize_image(image_label)  # (C, H, W), [-1, 1]
                
                feat_before = clip_feature_single(normalized_a, clip_model, clip_preprocess, device)
                feat_after = clip_feature_single(normalized_b, clip_model, clip_preprocess, device)
                feat_label = clip_feature_single(normalized_label, clip_model, clip_preprocess, device)
                
                data_items[image_name] = {
                    'image_before': image_a.astype(np.float32),
                    'image_after': image_b.astype(np.float32),
                    'feat_before': feat_before.cpu().numpy().astype(np.float32),
                    'feat_after': feat_after.cpu().numpy().astype(np.float32),
                }
                
                label_items[image_name] = {
                    'label': image_label.astype(np.float32),
                    'feat_label': feat_label.cpu().numpy().astype(np.float32),
                }
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if len(data_items) > 0:
            image_output_path = os.path.join(output_root, f'{split}_images.pkl')
            label_output_path = os.path.join(output_root, f'{split}_labels.pkl')
            
            with open(image_output_path, 'wb') as f:
                pickle.dump(data_items, f)
            
            with open(label_output_path, 'wb') as f:
                pickle.dump(label_items, f)
            
            print(f"Saved {split} data:")
            print(f"Images: {image_output_path} ({len(data_items)} items)")
            print(f"Labels: {label_output_path} ({len(label_items)} items)")

           
        else:
            print(f"No valid data found for {split}")

def load_preprocessed_data(output_root, split):

    image_path = os.path.join(output_root, f'{split}_images.pkl')
    label_path = os.path.join(output_root, f'{split}_labels.pkl')
    
    with open(image_path, 'rb') as f:
        image_data = pickle.load(f)
    
    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
    
    return image_data, label_data


if __name__ == "__main__":

    data_root = "LEVIR-MCI-dataset/images"   
    output_root = "preprocessed_data"   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    process_dataset(data_root, output_root, device)
    
    print("\nPreprocessing completed!")
    