from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys
#from .optimizer import Optimizer


class _scheduler_learning_rate(object):
    
	def __init__(self, optimizer=None, numberEpoch=0, lr_initial=0, lr_final=0, sigmoid_slope=0, sigmoid_shift=0, epoch=None):

		self.optimizer 		= optimizer
		self.numberEpoch	= numberEpoch
		self.lr_initial		= lr_initial
		self.lr_final 		= lr_final
		self.sigmoid_slope	= sigmoid_slope
		self.sigmoid_shift	= sigmoid_shift
		self.epoch 			= epoch	
		
		self.schedule 		= np.zeros(numberEpoch)

	def step(self):
	
		if self.epoch is None:
            
			self.epoch 	= 0 
			
		else:

			self.epoch 	= self.epoch + 1 
		
		lr = self.get_lr()
		# print(self.epoch)

		# for param_group in self.optimizer.param_groups:
        
		# 	param_group['lr'] = lr
		for optimizer in self.optimizer:

		    for param_group in self.optimizer[optimizer].param_groups:
			    param_group['lr'] = lr


	def get_lr(self):

		lr = self.schedule[self.epoch]

		return lr


	def plot(self):

		fig = plt.figure()
		ax	= fig.add_subplot(111)	
		
		ax.plot(self.schedule)
		
		plt.xlim(0, self.numberEpoch + 1)
		plt.xlabel('epoch')
		plt.ylabel('learning rate')
		plt.grid(linestyle='dotted')
		plt.tight_layout()
		plt.savefig('double_lr.png')
		plt.show()


# lr_initial 	= [initial]
# lr_final 		= [final]
# numberEpoch 	= [num]

class scheduler_learning_rate_sigmoid(_scheduler_learning_rate):

	def __init__(self, optimizer, numberEpoch, lr_initial, lr_final, sigmoid_slope=10, sigmoid_shift=0):
		
		super(scheduler_learning_rate_sigmoid, self).__init__(optimizer, numberEpoch, lr_initial, lr_final, sigmoid_slope, sigmoid_shift)

		if numberEpoch > 0:

			_index 		= np.linspace(-1, 1, numberEpoch)
			_sigmoid	= 1 / (1 + np.exp(sigmoid_slope * _index + sigmoid_shift))

			val_initial	= _sigmoid[0]
			val_final	= _sigmoid[-1]

			a = (lr_initial - lr_final) / (val_initial - val_final)
			b = lr_initial - a * val_initial 

			self.schedule		= a * _sigmoid + b


# lr_initial 	= [initial_1, initial_2, ..., initial_k]
# lr_final 		= [final_1, final_2, ..., final_k]
# alpha 		= [alpha_1, alpha_2, ..., alpha_k]
# beta 			= [beta_1, beta_2, ..., beta_k]

class scheduler_learning_rate_sigmoid_double(_scheduler_learning_rate):

	def __init__(self, optimizer=None, numberEpoch=0, lr_initial=[0, 0], lr_final=[0, 0], sigmoid_slope=[10, 10], sigmoid_shift=[0, 0]):
        
		super(scheduler_learning_rate_sigmoid_double, self).__init__(optimizer, numberEpoch, lr_initial, lr_final, sigmoid_slope, sigmoid_shift)

		num_epoch1	= int(np.round(numberEpoch / 2))
		num_epoch2 	= numberEpoch - num_epoch1
		num_epoch 	= [num_epoch1, num_epoch2]
		schedule 	= []

		if numberEpoch > 0:

			for i in range(len(num_epoch)):

				_numEpoch	= num_epoch[i]
				_initial 	= lr_initial[i]
				_final 		= lr_final[i]

				_index 		= np.linspace(-1, 1, _numEpoch)
				_sigmoid	= 1 / (1 + np.exp(sigmoid_slope[i] * _index + sigmoid_shift[i]))

				val_initial = _sigmoid[0]
				val_final	= _sigmoid[-1]

				a = (_initial - _final) / (val_initial - val_final)
				b = _initial - a * val_initial 

				schedule = np.concatenate((schedule, a * _sigmoid + b))

			self.schedule = schedule