import numpy as np
from keras.optimizers import *

class AdamAccumulate(Optimizer):
	"""Adam optimizer.

	Default parameters follow those provided in the original paper.

	# Arguments
		lr: float >= 0. Learning rate.
		beta_1: float, 0 < beta < 1. Generally close to 1.
		beta_2: float, 0 < beta < 1. Generally close to 1.
		epsilon: float >= 0. Fuzz factor.
		decay: float >= 0. Learning rate decay over each update.
		accum_iters: effective batch_size = accum_iters * real batch_size

	# References
		- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
	"""

	def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
				 epsilon=1e-8, decay=0., accum_iters=4, **kwargs):
		super(AdamAccumulate, self).__init__(**kwargs)
		self.iterations = K.variable(0)
		self.lr = K.variable(lr)
		self.beta_1 = K.variable(beta_1)
		self.beta_2 = K.variable(beta_2)
		self.epsilon = epsilon
		self.decay = K.variable(decay)
		self.initial_decay = decay
		self.accum_iters = K.variable(accum_iters)

	def get_updates(self, params, constraints, loss):
		grads = self.get_gradients(loss, params)
		self.updates = [K.update_add(self.iterations, 1)]

		#accum_switch = K.floor((self.accum_iters - K.mod(self.iterations + 1., self.accum_iters))/self.accum_iters)
		accum_switch = K.equal(self.iterations % self.accum_iters, 0)
		accum_switch = K.cast(accum_switch, dtype='float32')

		lr = self.lr
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * self.iterations))

		t = self.iterations + 1
		lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
					 (1. - K.pow(self.beta_1, t)))

		shapes = [K.get_variable_shape(p) for p in params]
		ms = [K.zeros(shape) for shape in shapes]
		vs = [K.zeros(shape) for shape in shapes]
		gs = [K.zeros(shape) for shape in shapes]
		self.weights = [self.iterations] + ms + vs

		for p, gp, m, v, ga in zip(params, grads, ms, vs, gs):
			g = (ga + gp)/self.accum_iters
			m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
			v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
			p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

			self.updates.append(K.update(m, (1-accum_switch)*m + accum_switch*m_t))
			self.updates.append(K.update(v, (1-accum_switch)*v + accum_switch*v_t))
			self.updates.append(K.update(ga, (1-accum_switch)*(ga + gp)))

			new_p = p_t
			# apply constraints
			if p in constraints:
				c = constraints[p]
				new_p = c(new_p)
			self.updates.append(K.update(p, (1-accum_switch)*p + accum_switch*new_p))
		return self.updates

	def get_config(self):
		config = {'lr': float(K.get_value(self.lr)),
				  'beta_1': float(K.get_value(self.beta_1)),
				  'beta_2': float(K.get_value(self.beta_2)),
				  'decay': float(K.get_value(self.decay)),
				  'epsilon': self.epsilon,
				  'accum_iters': int(K.get_value(self.accum_iters))}
		base_config = super(AdamAccumulate, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
