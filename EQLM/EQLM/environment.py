# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2020 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------
"""Contains the gym environment wrapper class"""

import gym


class Environment(gym.Wrapper):
	"""
	Gym environment wrapper with additional attributes and flattens states

	...

	Attributes
	----------
	state_size : int
		Size of the environment state space
	action size : int
		Size of the environment action space

	Methods
	-------
	reset()
		Flattens the state output
	step(a,render=False,**kwargs)
		Flattens the state output, gives option of rendering the environment

	The action space must be discrete to work with the agents implemented in this package
	"""
	state_size = None
	action_size = None
	def __init__(self,env_name='CartPole-v0'): # out-of-date, but not forcing using v1
	# def __init__(self,env_name='CartPole-v1'): #  max 500 time step for each eps(200 for v0)
		"""
		Parameters
		----------
		env_name : str, optional
			Name of the gym environment

		Raises
		------
		ValueError
			If the environment action space is not discrete
		"""
		super().__init__(gym.make(env_name))
		if not hasattr(self.action_space,'n'):
			raise ValueError(env_name + ' environment action space is incompatible')
		self.state_size = gym.spaces.flatdim(self.observation_space)
		self.action_size = self.action_space.n

	def reset(self):
		"""Additionally flatten and reshape the state output"""
		# s = super().reset()
		s, _ = super().reset() # New gym version(0.18.0->0.26.1), with GPU very slow
		return gym.spaces.flatten(self.observation_space,s).reshape(1,-1)

	def step(self,a,render=False,**kwargs):
		"""Additionally flatten and reshape the state output
		...
		Parameters
		----------
		a : int
			Action to execute in environment
		render : bool, optional
			Whether to render the environment at this step
		**kwargs
			Additional keyword arguments passed to `render`
		"""
		# s,r,d,info = super().step(a)
		# s,r,d,info,_ = super().step(a) #New gym version, this code no limited time step(200, or reward 195)
		# # why rewards higher than 200?(new gym problem, https://github.com/boyu-ai/Hands-on-RL/issues/72)
		s,r,terminated,truncated, info = super().step(a) # New gym version
		d = terminated or truncated # new done with all situations
		if render:
			super().render(**kwargs)
		return gym.spaces.flatten(self.observation_space,s).reshape(1,-1),r,d,info
