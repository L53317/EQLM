# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import tensorflow as tf
import numpy as np
import random
import math
import pandas as pd
from environment import Environment
from operator import itemgetter
import pdb

class QNetAgent():
	def __init__(self,agent_config,network_config,env):

		for key in agent_config.keys():
			setattr(self, key, agent_config[key])
		for key in network_config.keys():
			setattr(self, key, network_config[key])
		self.eps = self.eps0

		self.state_size = env.state_size
		self.action_size = env.action_size

		# Build network
		self.state_input = tf.placeholder(shape=[None,self.state_size],dtype=tf.float32)
		self.w_in = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,self.init_mag))
		self.b_in = tf.Variable(tf.random_uniform([1,self.N_hid],0,0))
		self.W = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,self.init_mag))
		try:
			act_fn = getattr(tf,self.activation)
		except AttributeError:
			act_fn = tf.tanh
		self.act = act_fn(tf.matmul(self.state_input,tf.add(self.w_in,self.b_in)), name=None)
		self.Q_est = tf.matmul(self.act,self.W)

		# Updating
		self.nextQ = tf.placeholder(shape=[None,self.action_size],dtype=tf.float32)
		loss = tf.reduce_sum(tf.square(self.nextQ - self.Q_est))
		trainer = tf.compat.v1.train.GradientDescentOptimizer(self.alpha)
		if isinstance(self.clip_norm, float):
			grads = trainer.compute_gradients(loss,[self.W,self.w_in,self.b_in])
			cap_grads = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in grads]
			self.updateModel = trainer.apply_gradients(cap_grads)
		else:
			self.updateModel = trainer.minimize(loss,var_list=[self.W,self.w_in,self.b_in])

		# Target network
		self.wt = tf.Variable(tf.random_uniform([self.state_size,self.N_hid],0,0.1))
		self.bt = tf.Variable(tf.random_uniform([1,self.N_hid],0,0.01))
		self.Wt = tf.Variable(tf.random_uniform([self.N_hid,self.action_size],0,0.1))
		self.actt = act_fn(tf.matmul(self.state_input,tf.add(self.wt,self.bt)), name=None)
		self.Qt = tf.matmul(self.actt,self.Wt)
		
		self.wt_assign = self.wt.assign(self.w_in)
		self.bt_assign = self.bt.assign(self.b_in)
		self.Wt_assign = self.Wt.assign(self.W)

		# Start tensorflow session
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# Memory initialisation
		self.prev_s=[]
		self.prev_a=[]
		self.memory=[]
		self.mem_df = pd.DataFrame(columns = ['s', 'a', 'r', 'sd', 'done']).astype({'s':'object', 
																					'a':'int64', 
																					'r':'float64', 
																					'sd':'object',
																				    'done':'bool'})
		self.step_count=0
		self.ep_no=0

	def action_select(self,env,state):

		if np.random.random(1)<self.eps:
			action=env.rand_action()
		else:
			q_s=self.sess.run(self.Q_est,feed_dict={self.state_input:state.reshape(1,-1)})
			action=np.argmax(q_s)
		self.prev_s=state
		self.prev_a=action
		return action

	def update_net(self,state,reward,done):

		# Update epsilon
		sdash = state
		if done:
			state=[]
			self.ep_no+=1
			if self.ep_no<self.n_eps:
				self.eps=float(self.eps0)-self.ep_no*(float(self.eps0)-float(self.epsf))/float(self.n_eps)
			else:
				self.eps=self.epsf

		# Update memory
		self.memory.append([self.prev_s,self.prev_a,reward,sdash,done])
		if len(self.memory)>self.max_mem:
			del self.memory[0]
# 		# pandas method
# 		self.mem_df = self.mem_df.append(pd.DataFrame({'s':[self.prev_s], 
# 													   'a':self.prev_a, 
# 													   'r':reward, 
# 													   'sd':[sdash],
# 													   'done':done}), ignore_index=True)

		# Select data from memory
		if len(self.memory)>self.minib & self.minib>1:
			sample_ind=random.sample(range(1,len(self.memory)),self.minib)
			df_ind = sample_ind
		elif self.minib==1:
			sample_ind=[len(self.memory)-1]
		else:
			sample_ind=range(len(self.memory))
			df_ind = sample_ind
		
		# update network
		with self.sess.as_default():
			# vectorised
			S1 = np.stack([self.memory[ind][0] for ind in sample_ind])
			Sd1 = np.stack([self.memory[ind][3] for ind in sample_ind])
			q_s1 = self.Qt.eval({self.state_input:S1})
			q_d1 = self.Qt.eval({self.state_input:Sd1})
			for i, ind in enumerate(sample_ind):
				r=self.memory[ind][2]
				a=self.memory[ind][1]
				if self.memory[ind][4]:
					q_s1[i][a] = r
				else:
					q_s1[i][a] = r + self.gamma*np.amax(q_d1[i])
			self.sess.run(self.updateModel,{self.state_input:S1,self.nextQ:q_s1})
			
# 			#iterative
# 			S_mat = []
# 			Q_mat = []
# 			for ind in sample_ind:
# 				s=self.memory[ind][0]
# 				r=self.memory[ind][2]
# 				a=self.memory[ind][1]
# 				q_s=self.Qt.eval({self.state_input:s.reshape(1,-1)})
# 				if not self.memory[ind][4]:
# 					q_dash=self.Qt.eval({self.state_input:self.memory[ind][3].reshape(1,-1)})
# 					q_s[0][a]=r+self.gamma*np.amax(q_dash)
# 				else:
# 					q_s[0][a]=r
# 				S_mat.append(np.array(s))
# 				Q_mat.append(np.array(q_s[0]))
# 			S = np.stack(S_mat)
# 			Q = np.stack(Q_mat)
# 			self.sess.run(self.updateModel,{self.state_input:S,self.nextQ:Q})
			
# 			# pandas method
# 			update_df = self.mem_df.loc[df_ind]
# 			S_df = np.stack(update_df.s.to_numpy())
# 			Sd_df = np.stack(update_df.sd.to_numpy())
# 			q_sdf = pd.DataFrame(self.Qt.eval({self.state_input:S_df}), columns = range(self.action_size))
# 			q_sddf = np.amax(self.Qt.eval({self.state_input:Sd_df}), axis=1)
# 			for i in range(len(q_sddf)):
# 				df_row = update_df.iloc[i]
# 				if df_row.done:
# 					q_sdf.loc[i,df_row.a] = df_row.r
# 				else:
# 					q_sdf.loc[i,df_row.a] = q_sddf[i]*self.gamma + df_row.r
# 			Q_df = np.stack(q_sdf.to_numpy())
# 			self.sess.run(self.updateModel,{self.state_input:S_df,self.nextQ:Q_df})

		# Update target network
		self.step_count += 1
		if self.step_count>=self.update_steps:
			self.sess.run(self.wt_assign)
			self.sess.run(self.bt_assign)
			self.sess.run(self.Wt_assign)
			self.step_count=0
