import torch
import torch.nn as nn


class TemporalModulation(nn.Module):
    def __init__(self, input_dim, output_dim, transform = True):
        super().__init__()
        if input_dim:
            self.fc_gamma = nn.Linear(input_dim, output_dim)  # γ: scale
            self.fc_beta = nn.Linear(input_dim, output_dim)   # β: bias
            
            nn.init.constant_(self.fc_gamma.weight, 0)
            nn.init.constant_(self.fc_gamma.bias, 1)
            nn.init.constant_(self.fc_beta.weight, 0)
            nn.init.constant_(self.fc_beta.bias, 0)
            
            if transform:
                self.fc_lamda = nn.Linear(input_dim, output_dim)  # λ: transform
                
                nn.init.constant_(self.fc_lamda.weight, 0)
                nn.init.constant_(self.fc_lamda.bias, 1)
        
        self.input_dim = input_dim
        self.transform = transform
    
    
    def forward(self, x, idx):
        if not self.input_dim:
            return x
        gamma = self.fc_gamma(idx)
        beta = self.fc_beta(idx)
        if self.transform:
            lamda = self.fc_lamda(idx)
            x = self.yeo_johnson(x, lamda)
        return gamma * x + beta
    
    
    def yeo_johnson(self, x, lamda):
        result = torch.zeros_like(x)
        epsilon = 1e-8
        
        mask_pos = x >= 0
        if mask_pos.any():
            x_pos = x[mask_pos]
            lam_pos = lamda[mask_pos]
            
            term_pos_nonzero = (torch.pow(x_pos + 1, lam_pos) - 1) / lam_pos
            term_pos_zero = torch.log(x_pos + 1)
            
            use_nonzero = torch.abs(lam_pos) >= epsilon
            term_pos = torch.where(use_nonzero, term_pos_nonzero, term_pos_zero)
            result[mask_pos] = term_pos
        
        mask_neg = x < 0
        if mask_neg.any():
            x_neg = x[mask_neg]
            lam_neg = lamda[mask_neg]
            denominator = 2 - lam_neg
            
            term_neg_nonzero = - (torch.pow(-x_neg + 1, denominator) - 1) / denominator
            term_neg_zero = -torch.log(-x_neg + 1)
            
            use_neg_nonzero = torch.abs(denominator) >= epsilon
            term_neg = torch.where(use_neg_nonzero, term_neg_nonzero, term_neg_zero)
            result[mask_neg] = term_neg
        
        return result