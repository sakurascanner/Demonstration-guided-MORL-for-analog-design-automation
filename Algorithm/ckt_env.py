import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
import numpy as np
import ctypes
import itertools
import json
import math
import pygame
import scipy.stats
from scipy.spatial import ConvexHull
from ckt_graph import GraphAMP
from math import ceil
from pathlib import Path
from typing import List, Optional
from ctypes import c_float, c_bool, c_int
from sim import Performance, Param
from dev_params import DeviceParams

MosArray = (c_float * 39) * 50
RArray = (c_float * 18) * 50
CArray = (c_float * 18) * 50
ParmArray = c_float * 20

CktGraph = GraphAMP

class AMPEnv(gym.Env, CktGraph, DeviceParams):

    def __init__(self, **kwargs):
        super(AMPEnv, self).__init__()
        gym.Env.__init__(self)
        CktGraph.__init__(self)
        DeviceParams.__init__(self, self.ckt_hierarchy)

        self.seed = 0
        self.CktGraph = CktGraph()
        self.sim_env = ctypes.CDLL("./simulation.so")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_shape, dtype=np.float64)
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.reward_shape, dtype=np.float64)

        self.parameters = (self.action_space_high + self.action_space_low ) / 2
        self.sim_time = 0


        self.reward_dim = 8

    def reset(self, seed=None):
        self.parameters = [1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 10, 1, 20, 1, 20, 1, 20, 5, 62300]
        self.sim_time = 0
        self.seed = seed
        performance = self.simulation(self.parameters, self.sim_time)
        obs = self.get_obs(performance.mos, performance.R, performance.C)
        return obs, {}
    
    def step(self, action):
        self.parameters += self.delta_action * action
        self.sim_time += 1

        performance = self.simulation(self.parameters, self.sim_time)
        obs =  self.get_obs(performance.mos, performance.R, performance.C)
        obs = np.clip(obs, -5, 5)

        done = False
        GBW_score = performance.gbw / self.GBW_target
        gain_score = performance.gain / self.dcgain_target
        pm_score = abs(performance.phase_margin - self.phase_margin_target) / 5
        cmrr_score = performance.cmrr / self.cmrr_target
        psrr_score = performance.psrr / self.psrr_target
        sr_score = performance.sr / self.sr_target

        idc_score = performance.idc / self.idc_target
        noise_score = performance.noise / self.noise_target

        reward = [GBW_score, gain_score, pm_score, idc_score, noise_score, cmrr_score, psrr_score, sr_score]
        
        return obs, reward, done, performance

    def render(self):
        return self

    def simulation(self, parameters, times):
        params = Param(parameters,times,ord('b'))
        self.sim_env.Simulate(params)
        self.perf = Performance.in_dll(self.sim_env,"final_perf")

    
    def get_obs(self, op_M):
        try:
            f = open(f'./AMP_NMCF_op_mean_std.json')
            op_mean_std = json.load(f)
            op_mean = op_mean_std['OP_M_mean']
            op_std = op_mean_std['OP_M_std']
            op_mean = np.array([op_mean['id'], op_mean['gm'], op_mean['gds'], op_mean['vth'], op_mean['vdsat'], op_mean['vds'], op_mean['vgs']])
            op_std = np.array([op_std['id'], op_std['gm'], op_std['gds'], op_std['vth'], op_std['vdsat'], op_std['vds'], op_std['vgs']])
        except:
            print('You need to run <_random_op_sims> to generate mean and std for transistor .OP parameters')
        
        OP_M0 = op_M[0]
        OP_M0_norm = (np.array([OP_M0[5],#'id'
                                OP_M0[1],#'gm'
                                OP_M0[2],#'gds'
                                OP_M0[4],#'vth'
                                OP_M0[3],#'vdsat'
                                OP_M0[19],#'vds'
                                OP_M0[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M1 = op_M[1]
        OP_M1_norm = (np.array([OP_M1[5],#'id'
                                OP_M1[1],#'gm'
                                OP_M1[2],#'gds'
                                OP_M1[4],#'vth'
                                OP_M1[3],#'vdsat'
                                OP_M1[19],#'vds'
                                OP_M1[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M2 = op_M[2]
        OP_M2_norm = (np.array([OP_M2[5],#'id'
                                OP_M2[1],#'gm'
                                OP_M2[2],#'gds'
                                OP_M2[4],#'vth'
                                OP_M2[3],#'vdsat'
                                OP_M2[19],#'vds'
                                OP_M2[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M3 = op_M[3]
        OP_M3_norm = (np.abs([OP_M3[5],#'id'
                                OP_M3[1],#'gm'
                                OP_M3[2],#'gds'
                                OP_M3[4],#'vth'
                                OP_M3[3],#'vdsat'
                                OP_M3[19],#'vds'
                                OP_M3[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M4 = op_M[4]
        OP_M4_norm = (np.abs([OP_M4[5],#'id'
                                OP_M4[1],#'gm'
                                OP_M4[2],#'gds'
                                OP_M4[4],#'vth'
                                OP_M4[3],#'vdsat'
                                OP_M4[19],#'vds'
                                OP_M4[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M5 = op_M[5]
        OP_M5_norm = (np.abs([OP_M5[5],#'id'
                                OP_M5[1],#'gm'
                                OP_M5[2],#'gds'
                                OP_M5[4],#'vth'
                                OP_M5[3],#'vdsat'
                                OP_M5[19],#'vds'
                                OP_M5[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M6 = op_M[6]
        OP_M6_norm = (np.array([OP_M6[5],#'id'
                                OP_M6[1],#'gm'
                                OP_M6[2],#'gds'
                                OP_M6[4],#'vth'
                                OP_M6[3],#'vdsat'
                                OP_M6[19],#'vds'
                                OP_M6[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M7 = op_M[7]
        OP_M7_norm = (np.array([OP_M7[5],#'id'
                                OP_M7[1],#'gm'
                                OP_M7[2],#'gds'
                                OP_M7[4],#'vth'
                                OP_M7[3],#'vdsat'
                                OP_M7[19],#'vds'
                                OP_M7[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M8 = op_M[8]
        OP_M8_norm = (np.array([OP_M8[5],#'id'
                                OP_M8[1],#'gm'
                                OP_M8[2],#'gds'
                                OP_M8[4],#'vth'
                                OP_M8[3],#'vdsat'
                                OP_M8[19],#'vds'
                                OP_M8[18]#'vgs'
                                ]) - op_mean)/op_std
        OP_M9 = op_M[9]
        OP_M9_norm = (np.array([OP_M9[5],#'id'
                                OP_M9[1],#'gm'
                                OP_M9[2],#'gds'
                                OP_M9[4],#'vth'
                                OP_M9[3],#'vdsat'
                                OP_M9[19],#'vds'
                                OP_M9[18]#'vgs'
                                ]) - op_mean)/op_std
        Ib = 0
        OP_C0_norm = 0
        OP_C1_norm = 0
        
        # state shall be in the order of node (node0, node1, ...)
        observation = np.array([
                               [0,0,0,0,      0,      OP_M0_norm[0],OP_M0_norm[1],OP_M0_norm[2],OP_M0_norm[3],OP_M0_norm[4],OP_M0_norm[5],OP_M0_norm[6]],
                               [0,0,0,0,      0,      OP_M1_norm[0],OP_M1_norm[1],OP_M1_norm[2],OP_M1_norm[3],OP_M1_norm[4],OP_M1_norm[5],OP_M1_norm[6]],
                               [0,0,0,0,      0,      OP_M2_norm[0],OP_M2_norm[1],OP_M2_norm[2],OP_M2_norm[3],OP_M2_norm[4],OP_M2_norm[5],OP_M2_norm[6]],
                               [0,0,0,0,      0,      OP_M3_norm[0],OP_M3_norm[1],OP_M3_norm[2],OP_M3_norm[3],OP_M3_norm[4],OP_M3_norm[5],OP_M3_norm[6]],
                               [0,0,0,0,      0,      OP_M4_norm[0],OP_M4_norm[1],OP_M4_norm[2],OP_M4_norm[3],OP_M4_norm[4],OP_M4_norm[5],OP_M4_norm[6]],
                               [0,0,0,0,      0,      OP_M5_norm[0],OP_M5_norm[1],OP_M5_norm[2],OP_M5_norm[3],OP_M5_norm[4],OP_M5_norm[5],OP_M5_norm[6]],
                               [0,0,0,0,      0,      OP_M6_norm[0],OP_M6_norm[1],OP_M6_norm[2],OP_M6_norm[3],OP_M6_norm[4],OP_M6_norm[5],OP_M6_norm[6]],
                               [0,0,0,0,      0,      OP_M7_norm[0],OP_M7_norm[1],OP_M7_norm[2],OP_M7_norm[3],OP_M7_norm[4],OP_M7_norm[5],OP_M7_norm[6]],
                               [0,0,0,0,      0,      OP_M8_norm[0],OP_M8_norm[1],OP_M8_norm[2],OP_M8_norm[3],OP_M8_norm[4],OP_M8_norm[5],OP_M8_norm[6]],
                               [0,0,0,0,      0,      OP_M9_norm[0],OP_M9_norm[1],OP_M9_norm[2],OP_M9_norm[3],OP_M9_norm[4],OP_M9_norm[5],OP_M9_norm[6]],
                               [self.Vdd,0,0,0,0,      0,0,0,0,0,0,0],
                               [0,self.GND,0,0,0,      0,0,0,0,0,0,0],
                               [0,0,Ib,0,0,       0,0,0,0,0,0,0],
                               [0,0,0,OP_C0_norm,0,    0,0,0,0,0,0,0],        
                               [0,0,0,0,OP_C1_norm,       0,0,0,0,0,0,0],                              
                               ])
        # clip the obs for better regularization
        observation = np.clip(observation, -5, 5)
        return

    def _init_random_sim(self, max_sims=20):
        '''
        
        This is NOT the same as the random step in the agent, here is basically 
        doing some completely random design variables selection for generating
        some device parameters for calculating the mean and variance for each
        .OP device parameters (getting a statistical idea of, how each ckt parameter's range is like'), 
        so that you can do the normalization for the state representations later.
    
        '''
        random_op_count = 0
        OP_M_lists = []
        OP_R_lists = []
        OP_C_lists = []
        OP_V_lists = []
        OP_I_lists = []
        
        while random_op_count <= max_sims :
            print(f'* simulation #{random_op_count} *')
            #action = np.random.uniform(self.action_space_low, self.action_space_high, self.action_dim) 
            action = np.array([1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 10, 1, 20, 1, 20, 1, 20, 5, 62300])
            print(f'action: {action}')
            self.sim_env.Get_Op_Range.argtypes = [Param]
            self.sim_env.Get_Op_Range(Param(action,random_op_count,ord('0')))
            op_results = Performance.in_dll(self.sim_env,"final_perf")
            
            OP_M_list = []
            OP_R_list = []
            OP_C_list = []
            OP_V_list = []
            OP_I_list = []

            for i in range(self.mos_num):
                OP_M_list.append(np.array(op_results.mos[i]))
            for i in range(self.r_num):
                OP_R_list.append(np.array(op_results.R[i]))
            for i in range(self.c_num):
                OP_C_list.append(np.array(op_results.C[i]))
            for i in range(self.v_num):
                OP_V_list.append(np.array(op_results.V[i]))
            for i in range(self.c_num):
                OP_I_list.append(np.array(op_results.I[i]))
                        
            OP_M_lists.append(np.array(OP_M_list))
            OP_R_lists.append(np.array(OP_R_list))
            OP_C_lists.append(np.array(OP_C_list))
            OP_V_lists.append(np.array(OP_V_list))
            OP_I_lists.append(np.array(OP_I_list))
            
            random_op_count = random_op_count + 1

        OP_M_lists = np.array(OP_M_lists)
        OP_R_lists = np.array(OP_R_lists)
        OP_C_lists = np.array(OP_C_lists)
        OP_V_lists = np.array(OP_V_lists)
        OP_I_lists = np.array(OP_I_lists)
        
        if OP_M_lists.size != 0:
            OP_M_mean = np.mean(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]), axis=0)
            OP_M_std = np.std(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]),axis=0)
            OP_M_mean_dict = {}
            OP_M_std_dict = {}
            for idx, key in enumerate(self.params_mos):
                OP_M_mean_dict[key] = float(OP_M_mean[idx])
                OP_M_std_dict[key] = float(OP_M_std[idx])
        
        # if OP_R_lists.size != 0:
        #     OP_R_mean = np.mean(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]), axis=0)
        #     OP_R_std = np.std(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]),axis=0)
        #     OP_R_mean_dict = {}
        #     OP_R_std_dict = {}
        #     for idx, key in enumerate(self.params_r):
        #         OP_R_mean_dict[key] = OP_R_mean[idx]
        #         OP_R_std_dict[key] = OP_R_std[idx]
                
        # if OP_C_lists.size != 0:
        #     OP_C_mean = np.mean(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]), axis=0)
        #     OP_C_std = np.std(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]),axis=0)
        #     OP_C_mean_dict = {}
        #     OP_C_std_dict = {}
        #     for idx, key in enumerate(self.params_c):
        #         OP_C_mean_dict[key] = OP_C_mean[idx]
        #         OP_C_std_dict[key] = OP_C_std[idx]     
                
        # if OP_V_lists.size != 0:
        #     OP_V_mean = np.mean(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]), axis=0)
        #     OP_V_std = np.std(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]),axis=0)
        #     OP_V_mean_dict = {}
        #     OP_V_std_dict = {}
        #     for idx, key in enumerate(self.params_v):
        #         OP_V_mean_dict[key] = OP_V_mean[idx]
        #         OP_V_std_dict[key] = OP_V_std[idx]
        
        # if OP_I_lists.size != 0:
        #     OP_I_mean = np.mean(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]), axis=0)
        #     OP_I_std = np.std(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]),axis=0)
        #     OP_I_mean_dict = {}
        #     OP_I_std_dict = {}
        #     for idx, key in enumerate(self.params_i):
        #         OP_I_mean_dict[key] = OP_I_mean[idx]
        #         OP_I_std_dict[key] = OP_I_std[idx]

        self.OP_M_mean_std = {
            'OP_M_mean': OP_M_mean_dict,         
            'OP_M_std': OP_M_std_dict
            }

        with open(f'./AMP_NMCF_op_mean_std.json','w') as file:
            json.dump(self.OP_M_mean_std, file)


if __name__ == "__main__" :
    testEnv = AMPEnv()
    testEnv._init_random_sim()