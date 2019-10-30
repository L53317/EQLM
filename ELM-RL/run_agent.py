# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */
# ------ Copyright (C) 2018 University of Strathclyde and Author ------
# ---------------------- Author: Callum Wilson ------------------------
# --------------- e-mail: callum.j.wilson@strath.ac.uk ----------------

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import pdb

from qnet_agent import QNetAgent
from eqlm_agent import EQLMAgent
from environment import Environment


def data_smooth(data,n_avg):
	# A function to average data over n_avg timesteps
	ind_vec = np.arange(n_avg,len(data)+1,n_avg)
	data_avg = [0]
	for ind in ind_vec:
		data_avg.append(np.mean(data[ind-n_avg:ind]))
	return data_avg

def display_frames_as_gif(frames, filename_gif = None):
	"""
	Displays a list of frames as a gif, with controls
	"""
	plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
	patch = plt.imshow(frames[0])
	plt.axis('off')
	
	def animate(i):
		patch.set_data(frames[i])
	
	anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
	if filename_gif: 
		anim.save(filename_gif, writer='imagemagick', fps=30)

def show_policy(N_ep, env, agent, fname = None):
	frames = []
	observation = env.reset()
	firstframe = env.gym_env.render(mode = 'rgb_array')
	fig,ax = plt.subplots()
	im = ax.imshow(firstframe)
	for ep_no in range(N_ep):
		observation = env.reset()
		done = False
		while not done:
			action = agent.action_select(env,observation)
			observation, _, done, _ = env.step(action)
			frame = env.gym_env.render(mode = 'rgb_array')
			im.set_data(frame)
			frames.append(frame)
	if fname:
		display_frames_as_gif(frames, filename_gif=fname)


def network_config():
	netcon = {}
	netcon['alpha'] = 0.01
	netcon['gamma_reg'] = 0.0621
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 15
	netcon['N_hid'] = 12
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.8
	agentcon['eps0'] = 0.75
	agentcon['epsf'] = 0.0
	agentcon['n_eps'] = 1000
	agentcon['minib'] = 10
	agentcon['max_mem'] = 10000
	agentcon['max_exp'] = 500
	return agentcon

N_ep = 1200
env = Environment('LunarLander-v2')
agent = QNetAgent(agent_config(),network_config(),env)

# Train the network for N_ep episodes
R_ep = []
for ep_no in range(N_ep):
	print('Episode: ' + repr(ep_no))
	observation = env.reset()
	done = False
	r = 0
	n_step = 0
	while not done:
		action = agent.action_select(env,observation)
		observation, reward, done, info = env.step(action)
		agent.update_net(observation,reward,done)
		r += reward
		n_step +=1
	R_ep.append(r)
	print('R: ' + repr(r) + ' Length: ' + repr(n_step))

show_policy(10, env, agent, fname = 'lander_anim0.gif')

agent.sess.close()

# Plot Reward
N_avg=100
R_plot=data_smooth(R_ep,N_avg)
plt.plot(np.arange(len(R_plot))*N_avg,R_plot,'r')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Total Discounted Reward', fontsize=12)
plt.show()