import torch
import numpy as np
import os
"""
Here you define the graph for a circuit
"""

class GraphAMP:
    """                                                                                                                           

    node 0 : M0 , node 1 : M1 , node 2 : M2 , node 3 : M3 , node 4 : M4 , node 5 : M5
    node 6 : M6 , node 7 : M7 , node 8 : M8 , node 9 : M9 , node 10 : M10 , node 11 : M11
    node 12 : M12 , node 13 : M13 , node 14 : M14 , node 15 : M15 , node 16 : M16 , node17 : M17 ,
    node 18 : M18 , node 19 : M19 , node 20 : M20 , node 21 : M21 , node 22 : M22 ,   
    node23 : M23 , node24 : Ib , node25 : VDD , node26 : GND , node27 : C0 , node28 : C1

    """
    def __init__(self):        
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu"
        # )
        
        self.device = torch.device(
           "cpu"
        )
        self.mos_num = 9
        self.r_num = 1
        self.c_num = 0
        self.i_num = 0
        self.v_num = 0
        self.ckt_hierarchy = (
                      ('M0','x1.XM0','pfet_01v8','m'),
                      ('M1','x1.XM1','pfet_01v8','m'),
                      ('M2','x1.XM2','pfet_01v8','m'),
                      ('M3','x1.XM3','pfet_01v8','m'),
                      ('M4','x1.XM4','pfet_01v8','m'),
                      ('M5','x1.XM5','pfet_01v8','m'),
                      ('M6','x1.XM6','pfet_01v8','m'),
                      ('M7','x1.XM7','pfet_01v8','m'),
                      ('M8','x1.XM8','pfet_01v8','m'),
                      ('M9','x1.XM9','pfet_01v8','m'),
                      ('M10','x1.XM10','pfet_01v8','m'),
                      ('M11','x1.XM11','pfet_01v8','m'),
                      ('M12','x1.XM12','nfet_01v8','m'),
                      ('M13','x1.XM13','nfet_01v8','m'),
                      ('M14','x1.XM14','nfet_01v8','m'),
                      ('M15','x1.XM15','nfet_01v8','m'),
                      ('M16','x1.XM16','nfet_01v8','m'),
                      ('M17','x1.XM17','nfet_01v8','m'),
                      ('M18','x1.XM18','nfet_01v8','m'),
                      ('M19','x1.XM19','nfet_01v8','m'),
                      ('M20','x1.XM20','nfet_01v8','m'),
                      ('M21','x1.XM21','nfet_01v8','m'),
                      ('M22','x1.XM22','nfet_01v8','m'),
                      ('M23','x1.XM23','nfet_01v8','m'),

                      ('Ib','','Ib','i'),
                      ('C0','x1.XC0','cap_mim_m3_1','c'),
                      ('C1','x1.XC1','cap_mim_m3_1','c')
                     )    

        self.op = {'M1':{},'M2':{},'M3':{},'M4':{},'M5':{},'M6':{},'M7':{},'M8':{},
                'Cc':{},'CL':{},'Mb':{},
                 }

        self.edge_index = torch.tensor([
          [0,1], [1,0], [0,2], [2,0], [0,3], [3,0], [0,4], [4,0],
          [1,3], [3,1], [1,4], [4,1], [1,5], [5,1], [1,8], [8,1],
          [2,3], [3,2], [2,5], [5,2], [2,10], [10,2],
          [3,5], [5,3], [3,8], [8,3], [3,10], [10,3],
          [4,6], [6,4], [4,7], [7,4], [4,10], [10,4], 
          [5,6], [6,5], [5,8], [8,5], [5,10], [10,5], 
          [6,7], [7,6], [6,10], [10,6], 
          [7,10], [10,7], 
          [8,9], [9,8],
            ], dtype=torch.long).t().to(self.device)
        
        self.num_relations = 2
        self.num_nodes = 29
        self.num_node_features = 12
        self.obs_shape = (self.num_nodes, self.num_node_features)

        """Select an action from the input state."""

        self.L_C1 = 30 
        self.W_C1 = 30
        M_C1_low = 1
        M_C1_high = 50 
        
        self.W_C0 = 30
        self.L_C0 = 30
        M_C0_low = 1
        M_C0_high = 50
        
        self.delta_action = np.array([0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             0.5, 0.5,
                             10,
                             100])
        self.action_space_low = np.array([0.5, 0.5,  # M1(L_low, W_low) 
                                        0.5, 0.5,     # M3/M4  
                                        0.5, 0.5,     # M5/M6
                                        0.5, 0.5,    # M7/M8
                                        0.5, 0.5,     # M9
                                        0.5, 0.5,     # Mb
                                        0.5, 0.5,    # M10
                                        0.5, 0.5,    # M11
                                        0.5, 0.5,    # M12 
                                        50,    # k
                                        100])   # r
        
        self.action_space_high = np.array([10, 50,  # M1(L_high, W_high) 
                                        10, 50,     # M3/M4  
                                        10, 50,     # M5/M6
                                        10, 50,    # M7/M8
                                        10, 50,     # M9
                                        10, 50,     # Mb
                                        10, 50,    # M10
                                        10, 50,    # M11
                                        10, 50,    # M12 
                                        2000,    # k
                                        100000])   # r
        
        self.action_dim = len(self.action_space_low)
        self.action_shape = (self.action_dim,)
        self.reward_dim = 8
        self.reward_shape = (self.reward_dim,)
        
        """Some target specifications for the final design"""
        self.PSRP_target = -90
        self.PSRN_target = -90 
        
        self.TC_target = 1e-6
        self.Power_target = 2e2
        self.vos_target = 4e-5
        
        self.cmrr_target = 80
        self.psrr_target = 80
        self.dcgain_target = 130
        self.GBW_target = 1e6
        self.phase_margin_target = 60 
        self.idc_target = 80#mA
        self.noise_target = 20#nV/ÌHz

        self.sr_target = 4e5
        self.settlingTime_target = 5e-6
        self.GND = 0
        self.Vdd = 1.8
        
        self.rew_eng = True        