import torch
import torch.nn as nn
from tqdm import tqdm
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from latch_model import LatCH

def sample_euler_latch_guided(
    model_fn,       
    latch_model,    
    target_feature, 
    shape,          
    sigmas,         
    alphas,         # Requires alpha schedule associated with sigmas
    extra_args=None,
    device='cuda',
    rho=0.03,       
    mu=0.03,
    gamma=0.3,
    guidance_steps_pct=0.20,
    n_iter=4
):
    """
    Implements Selective TFG using LatCH-F.
    Incorporates:
    - Mean guidance (iterations on z_{0|t})
    - Variance guidance
    - Gamma noise augmentation
    - Time domain weighting s(t)
    - LatCH evaluations matching SAO v-parameterization
    """
    b, c, t = shape
    latents = torch.randn(shape, device=device) * sigmas[0]
    
    num_steps = len(sigmas) - 1
    guidance_stop_step = int(num_steps * guidance_steps_pct)
    
    # Example MSE, but BCE should be passed or chosen based on feature
    criterion = nn.MSELoss()
    
    # Calculate s(t) normalization factor: sum of alphas
    sum_alphas = alphas.sum()

    pbar = tqdm(range(num_steps), desc="Sampling (LatCH Guided)")
    
    for i in pbar:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        alpha = alphas[i]
        
        # Time weighting s(t)
        s_t = alpha / sum_alphas
        rho_t = rho * s_t
        mu_t = mu * s_t
        
        requires_guidance = i < guidance_stop_step
        if requires_guidance:
            # We need gradients for variance guidance wrt latents (z_t)
            latents = latents.detach().requires_grad_(True)
            
        # 1. Forward pass SAO Model
        # SAO is a v-prediction model, returns v.
        v_pred = model_fn(latents, sigma * torch.ones(b, device=device), **(extra_args or {}))
        
        # Calculate z_{0|t} (denoised x0 estimate)
        # For VP schedule: z_{0|t} = alpha * z_t - sigma * v_pred
        z_0_t = alpha * latents - sigma * v_pred
        
        grad_variance = 0.0
        
        if requires_guidance:
            # --- Variance Guidance ---
            # t_norm in [0, 1] mapped from step i
            t_norm = (num_steps - i) / num_steps
            t_tensor = torch.full((b,), t_norm, device=device)
            
            # Predict from noisy z_t
            pred_feat_var = latch_model(latents, t_tensor)
            loss_var = criterion(pred_feat_var, target_feature)
            grad_variance = torch.autograd.grad(loss_var, latents, retain_graph=False)[0]
            
            # --- Mean Guidance Iteration on z_{0|t} ---
            for _ in range(n_iter):
                z_0_t = z_0_t.detach().requires_grad_(True)
                
                # gamma noise augmentation: N(LatCH(z_{0|t}, 0), gamma)
                # Since we evaluate at clean latents, t=0
                t_zero = torch.zeros((b,), device=device)
                
                # Evaluate clean feature
                clean_feat = latch_model(z_0_t, t_zero)
                
                # Add Gaussian noise with stdev gamma_t = gamma * s(t)
                e_0_t = clean_feat + torch.randn_like(clean_feat) * (gamma * s_t)
                
                loss_mean = criterion(e_0_t, target_feature)
                grad_mean = torch.autograd.grad(loss_mean, z_0_t)[0]
                
                # Gradient step on z_{0|t}
                z_0_t = z_0_t - mu_t * grad_mean
                
        # Standard Euler update step using derived 'd'
        d = (latents - z_0_t) / sigma
        
        latents_next = latents + d * (sigma_next - sigma)
        
        if requires_guidance:
            # Apply variance guidance by subtracting rho_t * grad_variance from z_{t-1}
            latents_next = latents_next - rho_t * grad_variance
            
        latents = latents_next.detach()
            
    return latents

def generate_wrapped(sao_pipeline, latch_path, prompt, target_curve, sigmas, alphas, device='cuda'):
    latch = LatCH(in_channels=64, out_channels=1, dim=256, depth=6, num_heads=8).to(device)
    latch.load_state_dict(torch.load(latch_path, map_location=device))
    latch.eval()
    print("LatCH guided generation setup complete.")
    pass
