class Early_Stopping():
    def __init__(self, patience):
        self.patience = patience
        self.best_error = float('inf')
        self.unimprovement_count = 0

    def update(self, error):
        if error >= self.best_error:
            self.unimprovement_count += 1
        else:
            self.best_error = error
            self.unimprovement_count = 0

    def should_stop(self):
        if self.patience > 0 and self.unimprovement_count >= self.patience:
            return True
        else:
            return False

    def has_improved(self):
        if self.unimprovement_count == 0:
            return True
        else:
            return False