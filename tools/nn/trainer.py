import time

import torch

from nn.logger import train_tf_logger


class Trainer:

    def __init__(self, net, N_seconds=1, DEBUG=False, seed=0, start=0):
        self.net = net
        self.N_seconds = N_seconds
        self.DEBUG = DEBUG
        self.start = start
        self.losses = []
        torch.manual_seed(seed)

    def get_losses(self):
        return self.losses
    
    def get_options(self, epoch):
        raise NotImplemented
    
    def get_data(self):
        raise NotImplemented
    
    def print_debug_info(self, X, y):
        print(X, y)
    
    def print_prediction(self, X, y, yhat):
        print('example prediction:', y, yhat)
    
    def print_log(self, epoch, batch, loss):
        print('[%d, %5d] loss: %.3f' % (epoch, batch, loss))        
    
    def save(self, epoch):
        pass
    
    def train_net(self, n_epochs, n_batches):
        """
        Этот метод тренирует нейросеть в течение n_epochs эпох 
        из n мини-батчей, а также выводит каждые N секунд средний лосс.
        """
    
        for epoch in range(self.start, n_epochs):
            # loop over the dataset multiple times
            loss_criterion, optimizer = self.get_options(epoch)
            running_loss = 0.0
            running_count = 0
            last_time = 0
              
            it = self.get_data()
    
            for i in range(n_batches):
                try:
                    X, y = next(it)
                except StopIteration:
                    # restart the loader
                    break
                
                if self.DEBUG and i == 0 and epoch == 0:
                    self.print_debug_info(X, y)

                # wrap them in Variable
                inputs, labels = X, y
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
              
                will_print = (i + 1 == n_batches)

                t = time.time()
                if t > last_time + self.N_seconds:
                    last_time = t
                    will_print = True

                if self.DEBUG and will_print:
                    yhat = outputs.data.numpy()

                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                # print(loss.data, type(loss), type(loss.data))
                l = loss.data.item()
                
                train_tf_logger.log(epoch * n_batches + i, l, 0)
                self.losses.append(l)
                running_loss += l
                running_count += 1
                
                if will_print:  # print every M seconds
                    if self.DEBUG:
                        self.print_prediction(X, y, yhat)

                    self.print_log(epoch, i + 1, running_loss / (running_count or 1))

                    running_loss = 0.0
                    running_count = 0
            
            self.save(epoch)
