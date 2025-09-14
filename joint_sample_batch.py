import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops
import libs.clip
from torchvision.utils import save_image, make_grid

import numpy as np
from PIL import Image
import os
import time
import json
from tqdm import tqdm
from libs.uvit_multi_post_ln_v3 import UViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models():
    nnet = UViT(
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        text_dim=64,
        num_text_tokens=77,
        clip_img_dim=512,
        use_checkpoint=True
    )
    nnet.to(device)
    checkpoint = torch.load('checkpoints/model_step_124000.pth', map_location='cpu')

    nnet.load_state_dict(checkpoint['nnet'])
    nnet.eval()

    from libs import caption_decoder, autoencoder
    caption_decoder = caption_decoder.CaptionDecoder(
        device=device, 
        pretrained_path="caption_checkpoints/caption_model_step_10000.pth", 
        hidden_dim=64
    )

    autoencoder = autoencoder.get_model(pretrained_path='models/autoencoder_kl.pth')
    autoencoder.to(device)
    
    return nnet, caption_decoder, autoencoder

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

def combine_joint(z, clip_img, text):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    text = einops.rearrange(text, 'B L D -> B (L D)')
    return torch.concat([z, clip_img, text], dim=-1)

def split_joint(x):
    C, H, W = 4, 32, 32
    z_dim = C * H * W
    z, clip_img, text = x.split([z_dim, 512, 77 * 64], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
    text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=64)
    return z, clip_img, text

def combine_diff(z, clip_img,diff, text):
    """Combines latent image (z), CLIP image embedding, and CLIP text embedding into a single flat tensor."""
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    diff= einops.rearrange(diff, 'B L D -> B (L D)')
    text = einops.rearrange(text, 'B L D -> B (L D)')
    return torch.concat([z, clip_img, diff, text], dim=-1)

def split_diff(x):
    """Splits a combined tensor back into latent image (z), CLIP image embedding, and CLIP text embedding."""
    C, H, W = 4, 32, 32  
    z_dim = C * H * W
    clip_img_dim = 512
    text_dim = 64
    num_text_tokens = 77
    
    z, clip_img,diff, text = x.split([z_dim, clip_img_dim,clip_img_dim*3, num_text_tokens * text_dim], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=clip_img_dim)
    diff = einops.rearrange(diff, 'B (L D) -> B L D', L=3, D=clip_img_dim)
    text = einops.rearrange(text, 'B (L D) -> B L D', L=num_text_tokens, D=text_dim)
    return z, clip_img, diff, text

def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watermarking(save_path):
    img_pre = Image.open(save_path)
    img_pos = utils.add_water(img_pre)
    img_pos.save(save_path)

def joint_nnet(x, timesteps,  data_type=1):
    z, clip_img, diff, text = split_diff(x)
    z_out, clip_img_out,_, text_out = nnet(img=z, clip_img=clip_img, text=text, t_img=timesteps, t_text=timesteps,diff=diff,
                    data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)
    x_out = combine_diff(z_out, clip_img_out, diff, text_out)

    return x_out


def sample_joint_on_dataset(test_loader, steps=50,  output_dir='out_joint_test'):
    os.makedirs(output_dir, exist_ok=True)
    inference_dir = os.path.join(output_dir, 'test_inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    generation_data = {}
    json_path = os.path.join(output_dir, 'generation.json')

    clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()

    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())
    
    batch_idx = 0  
    
    for batch_data in tqdm(test_loader):
        img_before, img_after, feat_before, feat_after, label, feat_label, text, key = batch_data
        
        label = label.to(torch.float32).to(device) / 255. * 2. - 1. 
        img_before = img_before.to(torch.float32).to(device) / 255. * 2. - 1.
        img_after = img_after.to(torch.float32).to(device) / 255. * 2. - 1.
        feat_before = feat_before.to(device)
        feat_after = feat_after.to(device)
        
        batch_size = label.size(0)
        
        with torch.no_grad():
            z = autoencoder.encode(label)  # shape: (B, 4, 64, 64)
            _diff = feat_after - feat_before  # (B, 1, 512)
            diff = torch.cat([feat_before, _diff, feat_after], dim=1)

            _z = torch.randn(z.size(0), *(4, 32, 32), device=device)
            _clip_img = torch.randn(z.size(0), 1, 512, device=device)
            _text = torch.randn(z.size(0), 77, 64, device=device)
            x_init = combine_diff(_z, _clip_img, diff, _text)
            
            def model_fn(x, t_continuous):
                t = (t_continuous * N).long()
                return joint_nnet(x, t,  data_type=1)

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
           
            with torch.autocast(device_type=device):
                start_time = time.time()
                x_gen = dpm_solver.sample(x_init, steps=steps, eps=1./N, T=1.)
                end_time = time.time()
                print(f'Generated {x_init.size(0)} samples with {steps} steps in {end_time - start_time:.2f}s')
                
            z_gen, _, _, text_gen = split_diff(x_gen)
            img_gen = unpreprocess(autoencoder.decode(z_gen))
            captions = caption_decoder.generate_captions(text_gen)
            
        for i in range(batch_size):
            raw_bef = unpreprocess(img_before[i])
            raw_aft = unpreprocess(img_after[i])
            single_label = unpreprocess(label[i])
            single_img_gen = img_gen[i]
            
            combined = torch.cat([raw_bef, raw_aft, single_label, single_img_gen], dim=2)
            
            if isinstance(key, (list, tuple)):
                filename = key[i]
            else:
                filename = key if batch_size == 1 else f"{key}_{i}"
            
            save_image(combined, os.path.join(output_dir, f"{filename}_gen.png"))
            
            inference_filename = f"{filename}.png"
            save_image(single_img_gen, os.path.join(inference_dir, inference_filename))
            
            if i < len(captions):
                generation_data[inference_filename] = captions[i]
            else:
                generation_data[inference_filename] = "No caption generated"
            
            caption_text = captions[i] if i < len(captions) else "No caption generated"
            gt_text = text[i] if i < len(text) else "No GT text"
            
            with open(os.path.join(output_dir, f'{filename}_caption.txt'), 'w') as f:
                f.write(f'{caption_text}\n')
                f.write(f'GT:{gt_text}')
        
        if batch_idx % 10 == 0:
            with open(json_path, 'w') as f:
                json.dump(generation_data, f, indent=2)
        
        batch_idx += 1
    
    with open(json_path, 'w') as f:
        json.dump(generation_data, f, indent=2)
    
    print(f" Generation completed. Results saved to:")
    print(f" Combined images: {output_dir}")
    print(f" Inference images: {inference_dir}")
    print(f" Captions JSON: {json_path}")
    print(f" Total images processed: {len(generation_data)}")

if __name__ == "__main__":
    nnet, caption_decoder, autoencoder = load_models()
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)
    
    steps = 50    # diffusion sampling steps

    seed = 1234    
    set_seed(seed)   
    output_path = 'out_joint_sample_step_'+str(steps)  
    os.makedirs(output_path, exist_ok=True)

    from dataloader_txt  import CCDataset
    from torch.utils.data import DataLoader
    
    batch_size = 88 
    test_dataset = CCDataset('datasets', split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Starting batch generation with batch_size={batch_size}")
    sample_joint_on_dataset(test_loader, steps=steps,  output_dir=output_path)