import numpy as np

def network_config():
	netcon = {}
	netcon['alpha'] = 0.01
	netcon['gamma_reg'] = 0.0621
	netcon['clip_norm'] = 1.0
	netcon['update_steps'] = 15
	netcon['N_hid'] = 11
	netcon['activation'] = 'tanh'
	netcon['init_mag'] = 0.01
	return netcon


def agent_config():
	agentcon = {}
	agentcon['gamma'] = 0.8
	agentcon['eps0'] = 0.8
	agentcon['epsf'] = 0.0
	agentcon['n_eps'] = 420
	agentcon['minib'] = 10
	agentcon['max_mem'] = 10000
	return agentcon