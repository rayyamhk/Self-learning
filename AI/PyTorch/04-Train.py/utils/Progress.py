import time
import os
from .log import log

class Progress():
    def __init__(self, train_size, train_batch_count, val_size, val_batch_count, log_batch):
        self.train_size = train_size
        self.train_batch_count = train_batch_count
        self.val_size = val_size
        self.val_batch_count = val_batch_count
        self.log_batch = log_batch

        self.start_time = None
        self.end_time = None

        self.current_epoch = 0
        self.current_batch = 0
        self.current_epoch_batch = 0
        self.current_epoch_progress = 0
        self.current_batch_size = 0

        self.running_count = 0
        self.running_correct_count = 0
        self.running_error = 0.0

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        log('The whole training takes %d seconds' % (self.end_time - self.start_time))

    def update_train_batch(self, current_batch_size, current_correct_count, current_loss):
        self.current_batch_size = current_batch_size
        self.current_batch += 1
        self.current_epoch_batch += 1
        self.current_epoch_progress += current_batch_size

        self.running_count += current_batch_size
        self.running_correct_count += current_correct_count
        self.running_error += current_loss * current_batch_size

    def update_val_batch(self, current_batch_size, current_correct_count, current_loss):
        self.running_count += current_batch_size
        self.running_correct_count += current_correct_count
        self.running_error += current_loss * current_batch_size

    def update_epoch(self):
        self.current_epoch += 1
        self.current_epoch_progress = 0
        self.current_epoch_batch = 0

    def reset_running(self):
        self.running_count = 0
        self.running_correct_count = 0
        self.running_error = 0.0

    def should_log(self):
        if self.current_epoch_batch % self.log_batch == 0 or self.current_epoch_batch == self.train_batch_count:
            return True
        else:
            return False
    
    def log_train(self):
        log('Epoch %d [%d/%d] - training error: %.3f - accuracy: %.3f'
        % (self.current_epoch, self.current_epoch_progress, self.train_size, self.running_error / self.running_count, self.running_correct_count / self.running_count))

    def log_val(self):
        log('Epoch %d - validation error: %.3f - accuracy: %.3f'
        % (self.current_epoch, self.running_error / self.running_count, self.running_correct_count / self.running_count))

    def export_running_error(self, root_dir, filename):
        if os.path.exists(root_dir) == False:
            os.mkdir(root_dir)
        
        filepath = os.path.join(root_dir, filename)

        with open(filepath, mode='a') as f:
            f.write(str(self.running_error / self.running_count) + '\n')
            f.close()

    def get_current_batch(self):
        return self.current_batch

    def get_running_error(self):
        return self.running_error / self.running_count