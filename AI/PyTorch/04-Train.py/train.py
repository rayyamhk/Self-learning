import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.Progress import Progress
from utils.Checkpoint import Checkpoint
from utils.Early_Stopping import Early_Stopping
from utils.log import log

def train(model, device, train_dataset, dev_dataset, epochs=1, batch_size=256, lr=3e-4, weight_decay=1e-4, early_stopping=4, checkpoints_max=5, checkpoint_batches=1000, log_batch=100):
    log('Preparing data loader...')
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataLoader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    progress_monitor = Progress(
        train_size=len(train_dataset),
        train_batch_count=len(train_dataLoader),
        val_size=len(dev_dataset),
        val_batch_count=len(dev_dataLoader),
        log_batch=log_batch
    )
    early_stopping_monitor = Early_Stopping(patience=early_stopping)
    checkpoints_monitor = Checkpoint(model, checkpoints_max, checkpoint_batches)

    model.to(device)
    log('Training has started on %s' % device)

    progress_monitor.start()
    for epoch in range(epochs):
        progress_monitor.update_epoch()

        model.train()
        for imgs, labels in train_dataLoader:
            imgs, labels = imgs.to(device), labels.to(device)

            pred = model(imgs)
            loss = loss_fn(pred, labels)

            progress_monitor.update_train_batch(
                current_batch_size=labels.size(0),
                current_correct_count=(torch.argmax(pred, dim=1) == labels).sum().item(),
                current_loss=loss.item()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if progress_monitor.should_log():
                progress_monitor.log_train()
                progress_monitor.export_running_error('./exports', 'train-errors.txt')
                progress_monitor.reset_running()
            
            if checkpoints_monitor.should_save(current_batch=progress_monitor.get_current_batch()):
                checkpoints_monitor.save()

        model.eval()
        with torch.no_grad():
            for imgs, labels in dev_dataLoader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs)

                progress_monitor.update_val_batch(
                    current_batch_size=labels.size(0),
                    current_correct_count=(torch.argmax(pred, dim=1) == labels).sum().item(),
                    current_loss=loss_fn(pred, labels).item()
                )

        progress_monitor.log_val()
        progress_monitor.export_running_error('./exports', 'val-errors.txt')
        early_stopping_monitor.update(progress_monitor.get_running_error())
        progress_monitor.reset_running()

        if early_stopping_monitor.has_improved():
            export_model(model, 'best_weights.pth')
        
        if early_stopping_monitor.should_stop():
            break

    log('Exporting trained weights and errors to ./exports')
    export_model(model, 'final_weights.pth')
    progress_monitor.end()

def export_model(model, filename):
    root_dir = './exports'

    if os.path.exists(root_dir) == False:
        os.mkdir(root_dir)

    weights_path = os.path.join(root_dir, filename)
    torch.save(model.state_dict(), weights_path)
