import torch
import torch.nn as nn
import torch.nn.functional as F

def rep_distill_loss(h_mem, h_gen):
    """
    Representation distillation loss.
    Pulls h_gen towards detached h_mem.
    Args:
        h_mem (torch.Tensor): Hidden states from P_mem. Shape: (B, D)
        h_gen (torch.Tensor): Hidden states from P_gen. Shape: (B, D)
    """
    # Normalize
    h_mem_norm = F.normalize(h_mem.detach(), p=2, dim=-1)
    h_gen_norm = F.normalize(h_gen, p=2, dim=-1)
    
    # Cosine similarity
    cos_sim = (h_mem_norm * h_gen_norm).sum(dim=-1)
    return 1.0 - cos_sim.mean()

def contrastive_loss(h_mem, h_gen, temperature=0.1):
    """
    InfoNCE Contrastive loss with in-batch negatives.
    Args:
        h_mem (torch.Tensor): (B, D)
        h_gen (torch.Tensor): (B, D)
    """
    h_mem_norm = F.normalize(h_mem.detach(), p=2, dim=-1)
    h_gen_norm = F.normalize(h_gen, p=2, dim=-1)
    
    # Similarity matrix (B, B)
    sim_matrix = torch.matmul(h_gen_norm, h_mem_norm.t()) / temperature
    
    labels = torch.arange(h_gen.size(0), device=h_gen.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
