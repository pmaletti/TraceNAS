import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k

        # self.topkroute_linear = nn.Linear(n_embed, num_experts)
        
        self.topkroute_linear = nn.Sequential(
            nn.Linear(n_embed, 16),
            nn.Linear(16, num_experts)
        )
        
        self.noise_linear = nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        mh_output = mh_output.to(next(self.topkroute_linear.parameters()))
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # print(top_k_logits.requires_grad)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # print(sparse_logits, sparse_logits.requires_grad)

        router_output = F.softmax(sparse_logits, dim=-1)        
        return router_output, indices


class SparseMoE(nn.Module):
    ### following the implementation in https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
    
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        gating_output, indices = self.router(x)  # [batch_size, num_experts]
        
        return gating_output.squeeze(0), indices.squeeze(0)

        # final_output = torch.zeros_like(x)
        
        # # Reshape inputs for batch processing
        # flat_x = x.view(-1, x.size(-1))
        # flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # # Process each expert in parallel
        # for i, expert in enumerate(self.experts):
        #     # Create a mask for the inputs where the current expert is in top-k
        #     expert_mask = (indices == i).any(dim=-1)
        #     flat_mask = expert_mask.view(-1)

        #     if flat_mask.any():
        #         expert_input = flat_x[flat_mask]
        #         expert_output = expert(expert_input)

        #         # Extract and apply gating scores
        #         gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
        #         weighted_output = expert_output * gating_scores

        #         # Update final output additively by indexing and adding
        #         final_output[expert_mask] += weighted_output.squeeze(1)

        # return final_output

class SoftMoE(nn.Module):
    """
    Implements a Soft Mixture of Experts, where all expert outputs are
    weighted by the router's scores and then summed.
    """
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        # self.experts = MoEExperts(n_embed, num_experts) # ModuleList of experts
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        gating_output, _ = self.router(x)
        
        # Step 2: Get the output of ALL experts
        # We need to run the input through every single expert to get their outputs
        expert_outputs = torch.stack([self.experts.experts[i](x) for i in range(self.num_experts)], dim=1) # [batch_size, num_experts, n_embed]
        
        # Step 3: Perform a weighted sum of the expert outputs
        # The gating_output scores are used as weights.
        # We need to reshape gating_output to [batch_size, num_experts, 1] to broadcast correctly
        weighted_expert_outputs = gating_output.unsqueeze(2) * expert_outputs
        
        # Step 4: Sum the weighted outputs to get the final output
        final_output = torch.sum(weighted_expert_outputs, dim=1) # [batch_size, n_embed]

        return final_output.squeeze(0) if final_output.shape[0] == 1 else final_output