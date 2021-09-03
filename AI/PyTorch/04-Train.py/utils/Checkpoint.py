import torch
import os
from .log import log

class Checkpoint():
    def __init__(self, model, checkpoints_max, checkpoint_batches):
        self.model = model
        self.checkpoints_max = checkpoints_max
        self.checkpoint_batches = checkpoint_batches
        self.current_batch = 0
        self.root_dir = './checkpoints'

        if checkpoints_max > 0 and os.path.exists(self.root_dir) == False:
            os.mkdir(self.root_dir)

    def should_save(self, current_batch):
        self.current_batch = current_batch
        if self.checkpoints_max > 0 and current_batch % self.checkpoint_batches == 0:
            return True
        else:
            return False

    def save(self):
        expired_ckp_name = 'ckp_%d.pth' % (self.current_batch - self.checkpoints_max * self.checkpoint_batches)
        expired_ckp_path = os.path.join(self.root_dir, expired_ckp_name)

        if os.path.exists(expired_ckp_path) == True:
            os.remove(expired_ckp_path)

        weights = self.model.state_dict()
        ckp_path = os.path.join(self.root_dir, 'ckp_%d.pth' % self.current_batch)
        log('Saving checkpoint %d' % (self.current_batch))
        torch.save(weights, ckp_path)