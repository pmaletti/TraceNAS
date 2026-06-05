import torch
from torch.optim.lr_scheduler import LRScheduler
import math

class CustomWSDScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer, 
        num_warmup_steps: int,
        num_stable_steps_1: int,
        num_warmup_steps_2: int,
        num_stable_steps_2: int,
        num_decay_steps: int,
        lr_1: float = 1e-5,
        lr_2: float = 1e-4,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = max(1, num_warmup_steps)
        self.num_stable_steps_1 = num_stable_steps_1
        self.num_warmup_steps_2 = max(1, num_warmup_steps_2)
        self.num_stable_steps_2 = num_stable_steps_2
        self.num_decay_steps = max(1, num_decay_steps)
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        
        # 1. First Warmup to lr_1 (1e-5)
        if step < self.num_warmup_steps:
            alpha = step / self.num_warmup_steps
            curr_lr = alpha * self.lr_1
            
        # 2. First Stable Period at lr_1
        elif step < (self.num_warmup_steps + self.num_stable_steps_1):
            curr_lr = self.lr_1
            
        # 3. Second Warmup from lr_1 to lr_2 (1e-4)
        elif step < (self.num_warmup_steps + self.num_stable_steps_1 + self.num_warmup_steps_2):
            local_step = step - (self.num_warmup_steps + self.num_stable_steps_1)
            alpha = local_step / self.num_warmup_steps_2
            curr_lr = self.lr_1 + alpha * (self.lr_2 - self.lr_1)
            
        # 4. Second Stable Period at lr_2
        elif step < (self.num_warmup_steps + self.num_stable_steps_1 + self.num_warmup_steps_2 + self.num_stable_steps_2):
            curr_lr = self.lr_2
            
        # 5. Final Decay (Cosine)
        else:
            total_before_decay = (self.num_warmup_steps + self.num_stable_steps_1 + 
                                  self.num_warmup_steps_2 + self.num_stable_steps_2)
            local_step = step - total_before_decay
            
            # Decay factor calculation (Cosine)
            progress = min(1.0, local_step / self.num_decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            curr_lr = self.min_lr + (self.lr_2 - self.min_lr) * cosine_decay
            
        return [curr_lr for _ in self.base_lrs]