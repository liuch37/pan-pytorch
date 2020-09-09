'''
Helper functions.
'''

def adjust_learning_rate(optimizer, dataloader, epoch, iter, schedule, lr, num_epoch):
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = num_epoch * len(dataloader)
        lr = lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr