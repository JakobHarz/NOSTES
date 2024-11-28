from typing import List

import casadi as ca
import numpy as np
import pandas as pd
import os

from casadi.tools import struct_symSX, entry, struct_SX

from utility import Constants


class Data:
    """ Class to preprocess and store the data for the NLP
        p_data (5): - T_amb ambient temperature
                    - P_pv power of the pv
                    - P_wind power of the wind
                    - P_load electric household power demand
                    - Qdot_load heat demand of the household
    """

    def __init__(self, file_path: str):
        # Load the data from the file
        raw_data = pd.read_csv(file_path)

        # Load the data
        self.data_T_amb = raw_data['T_amb'].values
        self.data_P_pv = raw_data['P_pv'].values
        self.data_P_wind = raw_data['P_wind'].values
        self.data_P_load = raw_data['P_load'].values
        self.data_Qdot_load = raw_data['Qdot_load'].values

    def getDataAtTime(self, time: float):
        """
        time: time in hours [0, ..., 24*365]
        """
        index = int(time) 
        return self.data_T_amb[index], self.data_P_pv[index], self.data_P_wind[index], self.data_P_load[index], self.data_Qdot_load[index]

    def getDataAtTimes(self, start, stop, step):
        """
        start: start time in hours
        stop: stop time in hours
        step: step size in hours
        """

        # check that the times are integer
        # assert np.all(np.mod(times, 1) == 0), "The times should be integers"
        # indices = times.astype(int)

        indices = np.arange(start, stop, step, dtype=int)
        return (self.data_T_amb[indices],
                self.data_P_pv[indices],
                self.data_P_wind[indices],
                self.data_P_load[indices],
                self.data_Qdot_load[indices])

class SystemModel:
    """ Base Class for a system model, defines the properties that are needed for the NLP (dynamics f, fixed costs, running costs, state and control bounds).
    The child classes should implement these properties.
        p_fix (5): - Storage size [m^3]
                   - HP capacity [W]
                   - PV Install capacity [W]
                   - Wind Install capacity [W]
                   - Battery capacity [Wh]
    """

    def __init__(self, nx, nu, ndata, ntheta, data: Data,
                 stateNames: List[str] = None,):

        self.nx = nx
        self.nu = nu
        self.ndata = ndata
        self.ntheta = ntheta
        self.data: Data = data

        # Define the symbolic variables
        self.x = ca.SX.sym('x', nx)
        self.u = ca.SX.sym('u', nu)
        self.p_fix = ca.SX.sym('theta', ntheta)  # fixed parameters to be optimized
        self.p_data = ca.SX.sym('data', ndata)  # data parameters

        # names for states and controls, can be overwritten
        self.stateNames = [f'x_{i}' for i in range(nx)]
        self.controlNames: List[str] = ['P_hp', 'P_ch', 'P_dis', 'P_Grid_buy', 'P_Grid_sell']
        self.p_fix_names: list = ['s_S', 's_hp', 's_pv', 's_wind', 's_bat']

        self.params = Constants()  # Initialize SystemParameters

        # initialization, overwrite in the subclass
        self.x0 = ca.DM.zeros(self.x.shape)


        # to be implemented in the subclass
        self.f: ca.Function = None
        """ Casadi function x' = f(x,u,t,p_fix,p_data) that computes the state dynamics"""

        # fixed costs: investment costs
        self.J_fix: ca.Function = None
        """ The fixed costs of the system J_fix = J_fix(p_fix)"""

        # running costs: operational costs
        self.cost_grid = self.compute_running_cost(self.u)
        self.J_running = ca.Function('J_running', [self.u], [self.cost_grid], ['u'], ['J_running'])
        """ The running costs over 20 years the system J_running = J_running(u)"""

        # lbp and ubp: bounds for the p_fix
        self.lbp = [1] * ca.DM.ones(self.p_fix.shape)
        self.ubp = [1] * ca.DM.ones(self.p_fix.shape)

        # upper and lower bounds for the states and controls
        self.lbx = -ca.inf * ca.DM.ones(self.x.shape)
        self.ubx = ca.inf * ca.DM.ones(self.x.shape)
        self.lbu = -ca.inf * ca.DM.ones(self.u.shape)
        self.ubu = ca.inf * ca.DM.ones(self.u.shape)

        # for outputs
        self.outputs: dict = None
        self._output_struct = None
        self._output_function = None

    def get_data(self, time: float):
        return self.data.getDataAtTime(time)

    def compute_Qdot_hp(self, P_hp, T_amb):
        T_lift = self.params.T_hp - T_amb  # [K] Temperature lift of the heat pump
        COP = self.params.eta_hp * self.params.T_hp / T_lift  # Coefficient of performance of the heat pump
        Qdot_hp = P_hp * COP  # Heat output of the heat pump
        return Qdot_hp

    def battery_model(self, x, u, p_fix, p_data):
        eta_ch, eta_dis = self.params.battery_params()
        P_ch = u[1]  # Charging power
        P_dis = u[2]  # Discharging power
        xdot_soc = (P_ch * eta_ch - P_dis / eta_dis) / (self.C_bat * 3600)  # [1/s]
        return xdot_soc

    def compute_running_cost(self,u):
        """ Compute the running cost of the system"""
        P_grid_pos = u[3] # [W]
        P_grid_neg = u[4] # [W]

        price_sell_Wh = self.params.price_sell/1e3  # EUR/Wh
        price_buy_Wh = self.params.price_buy/1e3  # EUR/Wh

        cost_grid_annual = price_sell_Wh*P_grid_neg + price_buy_Wh*P_grid_pos
        cost_grid = cost_grid_annual * self.params.n_years  # [EUR]
        return cost_grid

    @property
    def outputStruct(self) -> struct_symSX:
        if self._output_struct is None:
            entries = []
            for key, subdict in self.outputs.items():
                # check that the items have both a 'value' and a 'type' key
                assert 'value' in subdict, f"Key '{key}' in outputs does not have a 'value' key"
                assert 'type' in subdict, f"Key '{key}' in outputs does not have a 'type' key"
                single_entry = entry(key, expr=subdict['value'])
                entries.append(single_entry)
            self._output_struct = struct_SX(entries)
        return self._output_struct

    @property
    def outputFunction(self) -> ca.Function:
        """ Casadi function that computes the outputs of the system"""
        if self._output_function is None:
            self._output_function = ca.Function('f_output', [self.x, self.u, self.p_fix, self.p_data], [self.outputStruct])
        return self._output_function


