import casadi as ca
import numpy as np

from systemmodels.stratstoragemodel import StratStorageModel
from systemmodels.systemModel import SystemModel
from utility import Constants, Data


class AveragedStratStorageModel(StratStorageModel):
    def __init__(self, s_n, g_n, distance, data: Data, constants: Constants):
        """ Stratified Model
        x (sto_n, g_n): T_s[s_n], T_g[g_n], x_soc
        u (5): P_hp [W], P_ch [W], P_dis [W], P_Grid_buy [W], P_Grid_sell [W]
        p_fix (5): - Storage size [m^3]
                   - HP capacity [W]
                   - PV Install capacity [W]
                   - Wind Install capacity [W]
                   - Battery capacity [Wh]

        p_data (5): - T_amb ambient temperature
                    - P_pv power of the pv
                    - P_wind power of the wind
                    - P_HH electric household power demand
                    - Qdot_HH heat demand of the household
            """
        super().__init__(s_n, g_n, distance, data, constants)

        # split the state
        self.x_sto = self.x[:-1]  # the states of the storage
        self.x_bat = self.x[-1]  # the state of the battery

        # evaluate the dynamics for the (later) average values of Tamb, Qdot_hh, Qdot_hp...
        _Tamb_SX = ca.SX.sym('T_amb')
        _Qdot_hh_SX = ca.SX.sym('Qdot_hh')
        _Qdot_hp_SX = ca.SX.sym('Qdot_hp')
        ode_sto = self.storage_model(self.x_sto, _Tamb_SX, _Qdot_hh_SX, _Qdot_hp_SX, self.p_fix)

        # split dynamics into coarse and fine
        self.f_sto_average = ca.Function('f', [self.x_sto, _Tamb_SX, _Qdot_hh_SX, _Qdot_hp_SX, self.p_fix], [ode_sto])
        self.f_bat = ca.Function('f', [self.u, self.p_fix], [self.battery_model(self.x, self.u, self.p_fix, self.p_data)])

        # fine constraints
        g_vec_fine = ca.vertcat(self.const_bat1,
                                self.const_bat2,
                                self.const_hp,
                                self.const_Grid)
        self.g_fine = ca.Function('g', [self.x_bat, self.u, self.p_fix, self.p_data], [g_vec_fine])
        self.lbg_fine = ca.vertcat(-ca.inf * ca.DM.ones((g_vec_fine.shape[0] - 1, 1)), 0)
        self.ubg_fine = ca.DM.zeros(g_vec_fine.shape)
        self.ng_fine = g_vec_fine.shape[0]

        # coarse constraints
        # g_vec_coarse = ca.vertcat(self.mdot_hp - self.constants.mdot_hp_max)
        # self.g_coarse = ca.Function('g', [self.x_sto,_Qdot_hp_SX], [g_vec_coarse])
        # self.lbg_coarse = ca.vertcat(-ca.inf)
        # self.ubg_coarse = ca.DM.zeros(g_vec_coarse.shape)
