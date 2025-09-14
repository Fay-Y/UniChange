
import torch
import random
from absl import logging
import einops
import libs.clip
from torchvision.utils import save_image 
import numpy as np
import clip
import os
from tqdm import tqdm  

from dataloader_txt import CCDataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from libs.uvit_multi_post_ln_v3 import UViT

from torch.optim.lr_scheduler import LambdaLR
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse
import yaml

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas

def stp(s, ts: torch.Tensor):   
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1): 
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object): 
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def loads_for_training():
  
    logging.info("Loading models...")
    
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

    from libs import caption_decoder
    caption_decoder = caption_decoder.CaptionDecoder(
        device=device,
        pretrained_path="caption_checkpoints/caption_model_step_10000.pth", #the finetuned caption model
        hidden_dim=64 
    )

    from libs import autoencoder
    autoencoder = autoencoder.get_model(pretrained_path='models/autoencoder_kl.pth') 
    autoencoder.to(device)
    autoencoder.eval() 

    clip_img, clip_img_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_img.to(torch.float32).to(device)
    clip_img.eval() 

    clip_text = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text.to(device)
    clip_text.eval()

    logging.info("All models loaded successfully.")
    return nnet, caption_decoder, autoencoder, clip_img, clip_img_preprocess, clip_text

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

def unpreprocess(v):  
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

def customized_lr_scheduler(optimizer, warmup_steps=-1):
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)

def joint_nnet(nnet,x, timesteps, data_type=1):
    z, clip_img, diff, text = split_diff(x)
    z_out, clip_img_out,_, text_out = nnet(img=z, clip_img=clip_img, text=text, t_img=timesteps, t_text=timesteps,diff=diff,
                    data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)
    x_out = combine_diff(z_out, clip_img_out, diff, text_out)
    return x_out

def save_demo_samples(model, autoencoder, caption_decoder, batch_data, global_step, output_dir="demo_sample", sample_num=2):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    imgs_before,imgs_after,feat_before, feat_after, labels, feat_label, texts, keys = batch_data  # adjust if needed

    labels = labels.to(torch.float32).to(device) / 255. * 2. - 1.
    imgs_before = imgs_before.to(torch.float32).to(device) / 255. * 2. - 1.
    imgs_after = imgs_after.to(torch.float32).to(device) / 255. * 2. - 1.
    feat_before = feat_before.to(device)
    feat_after = feat_after.to(device)

    with torch.no_grad():
        z = autoencoder.encode(labels)
        _diff = feat_after - feat_before
        diff = torch.cat([feat_before, _diff, feat_after],dim=1)
        _z = torch.randn(z.size(0), *(4, 32, 32), device=device)
        _clip_img = torch.randn(z.size(0), 1, 512, device=device)
        _text = torch.randn(z.size(0), 77, 64, device=device)
        x_init = combine_diff(_z, _clip_img, diff, _text)

        def model_fn(x, t_continuous):
            t = (t_continuous * N).long()
            return joint_nnet(model, x, t, data_type=1)

        dpm_solver = DPM_Solver(model_fn, NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float()), predict_x0=True, thresholding=False)
        x_gen = dpm_solver.sample(x_init, steps=100, eps=1./N, T=1.)

        z_gen, _, _, text_gen = split_diff(x_gen)
        img_gen = unpreprocess(autoencoder.decode(z_gen))
        captions = caption_decoder.generate_captions(text_gen)

    indices = random.sample(range(labels.size(0)), min(sample_num, labels.size(0)))
    for i in indices:
        gen_img = img_gen[i]
        gt_img = unpreprocess(labels[i])
        raw_bef = unpreprocess((imgs_before[i]))
        raw_aft = unpreprocess(imgs_after[i])

        combined = torch.cat([raw_bef, raw_aft, gt_img, gen_img], dim=2)  
        save_image(combined, os.path.join(output_dir, f"step{global_step}_{keys[i]}.png"))

        with open(os.path.join(output_dir, f"step{global_step}_{keys[i]}.txt"), 'w') as f:
            f.write(f"Generated: {captions[i]}\n")
            f.write(f"GroundTruth: {texts[i]}")

def train():
    """Main function to perform the training of the diffusion model."""
    logging.set_verbosity(logging.INFO)
    logging.info(f"Using device: {device}")

    config = parse_config()
    resume = config.get("resume", False)
    checkpoint_folder = config.get("checkpoint_folder", "checkpoints")
    tensorboard_path = config.get("tensorboard_path", "logs/latest")
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-5)
    weight_decay = config.get("weight_decay", 1e-4)
    dataset_path = config.get("dataset_path", "datasets")

    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_path)
    nnet, caption_decoder, autoencoder, clip_img, clip_img_preprocess, clip_text = loads_for_training()
    nnet.train() 

    train_dataset = CCDataset(dataset_path, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 

    optimizer = torch.optim.AdamW(nnet.parameters(), lr=float(lr), weight_decay=float(weight_decay),betas=(0.9, 0.9))

    global_step = 0
    start_epoch = 0
    if resume:
        checkpoint_path = config.get("checkpoint_path", "checkpoints_v3/latest.pth")
        if os.path.exists(checkpoint_path):
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            nnet.load_state_dict(checkpoint["nnet"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            global_step = checkpoint.get("global_step", 0)
            start_epoch = checkpoint.get("epoch", 0)
    else:
        logging.info("Starting from scratch.")

    epochs = 1000000
    save_interval = 1000  
    joint_data_type = 1
    loss_fn = torch.nn.MSELoss()
    logging.info("Starting training...")

    for epoch in range(start_epoch, epochs):
        nnet.train()  
        total_epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")  
        for batch_idx, batch in enumerate(pbar):
            image_bef, image_aft, feats_before, feats_after, labels, feat_labels, texts, keys = batch
            batch_size = feats_before.size(0)  
            optimizer.zero_grad()  
            labels = labels.to(torch.float32).to(device) / 255.0
            labels = labels * 2.0 - 1.0
            feat_labels = feat_labels.to(device)
            feats_before = feats_before.to(device)
            feats_after = feats_after.to(device)
            diff = feats_after - feats_before  
            diff_true = torch.cat([feats_before,diff,feats_after],dim=1)
            with torch.no_grad():  
                z_true = autoencoder.encode(labels) #  (B, C, H, W)
                text_feat = clip_text.encode(texts) 
                text_true = caption_decoder.encode_prefix(text_feat)

            clip_img_true = feat_labels # (B, 1, D_clip_img_dim)

            x_init = combine_joint(z_true, clip_img_true, text_true) # (B, C*H*W + 512 + 77*64)
           
            n, eps, xt = _schedule.sample(x_init)  # n in {1, ..., 1000}
            
            z_t, clip_img_t, text_t = split_joint(xt)  
            predicted_x0, predicted_clip_img, _,predicted_text = nnet(
                img=z_t,   text=text_t,
                t_img=n, # Timestep for image 
                t_text=n, # Timestep for text 
                clip_img = clip_img_t,
                diff=diff_true,
                data_type=torch.full((batch_size,), joint_data_type, device=device, dtype=torch.int)  
            )
            eps_pred = combine_joint(predicted_x0, predicted_clip_img,  predicted_text) 
            loss = loss_fn(eps_pred, eps)  
            
            eps_x,eps_clip, eps_text = split_joint(eps)
            loss_label = loss_fn(predicted_x0,eps_x)
            loss_clip = loss_fn(predicted_clip_img,eps_clip)
            loss_text = loss_fn(predicted_text,eps_text)

            loss.backward()  
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/loss_label', loss_label.item(), global_step)
            writer.add_scalar('train/loss_clip', loss_clip.item(), global_step)
            writer.add_scalar('train/loss_text', loss_text.item(), global_step)
            writer.add_scalars('train/all_losses', {
                'total_loss': loss.item(),
                'label_loss': loss_label.item(),
                'clip_loss': loss_clip.item(),
                'text_loss': loss_text.item(),
            }, global_step)
            
            optimizer.step() 
            total_epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item()) 
            pbar.set_postfix(step=global_step)

            global_step += 1
            if global_step % save_interval == 0:
                step_checkpoint = os.path.join(checkpoint_folder,f"model_step_{global_step}.pth")
                torch.save({
                    "nnet": nnet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step
                }, step_checkpoint)
                logging.info(f"Checkpoint saved: {step_checkpoint}")

                latest_checkpoint_path = os.path.join(checkpoint_folder, f"latest.pth")
                torch.save({
                    "nnet": nnet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step
                }, latest_checkpoint_path)
                logging.info(f"Latest checkpoint saved to: {latest_checkpoint_path}")
                save_demo_samples(nnet, autoencoder, caption_decoder, batch, global_step)
            
        avg_epoch_loss = total_epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} finished, Average Loss: {avg_epoch_loss:.4f}")

    writer.close()
    logging.info("Training complete.")

if __name__ == "__main__":

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    N = len(_betas) 

    train()