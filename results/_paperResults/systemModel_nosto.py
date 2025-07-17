from typing import List

import casadi as ca

from casadi.tools import struct_symSX, entry, struct_SX

from utility_nosto import Data


class SystemModel:
    """ Base Class for a system model,
    defines the properties that are needed for the NLP (dynamics f, fixed costs, running costs, state and control bounds).
    The child classes should implement these properties.
        p_fix (4): - HP capacity [W]
                   - PV Install capacity [W]
                   - Wind Install capacity [W]
                   - Battery capacity [Wh]
    """

    def __init__(self, nx, nu, ndata, ntheta, data: Data, constants):

        self.nx = nx
        self.nu = nu
        self.ndata = ndata
        self.ntheta = ntheta
        self.data: Data = data
        self.constants = constants

        # check that the user defined the needed constants
        assert all([constants.C_bat_default is not None,
                    constants.C_pv_default is not None,
                    constants.C_hp_default is not None,
                    constants.C_wind_default is not None]), \
            ("The user has to provide default values for the sizes of the components in the constants:"
             " [C_bat_default,C_pv_default,C_hp_default,C_wind_default ]!")

        # Define the symbolic variables
        self.x = ca.SX.sym('x', nx)
        self.u = ca.SX.sym('u', nu)
        self.p_fix = ca.SX.sym('theta', ntheta)  # fixed parameters to be optimized
        self.p_data = ca.SX.sym('data', ndata)  # data parameters

        # names for states and controls, can be overwritten
        self.stateNames = [f'x_{i}' for i in range(nx)]
        self.controlNames: List[str] = ['P_hp', 'P_ch', 'P_dis', 'P_Grid_buy', 'P_Grid_sell']
        self.p_fix_names: list = ['s_hp', 's_pv', 's_wind', 's_bat']

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
        """ The running costs over the system J_running = J_running(u)"""

        # lbp and ubp: bounds for the p_fix
        self.lbp = [0.1] * ca.DM.ones(self.p_fix.shape)
        self.ubp = [10] * ca.DM.ones(self.p_fix.shape)

        # upper and lower bounds for the states and controls
        self.lbx = -ca.inf * ca.DM.ones(self.x.shape)
        self.ubx = ca.inf * ca.DM.ones(self.x.shape)
        self.lbu = -ca.inf * ca.DM.ones(self.u.shape)
        self.ubu = ca.inf * ca.DM.ones(self.u.shape)

        # for outputs
        self.outputs: dict = {}
        self._output_struct = None
        self._output_function = None

    def get_data(self, time: float):
        return self.data.getDataAtTime(time)

    def compute_Qdot_hp(self, P_hp, T_amb):
        T_hp = 40 + 273.15  # [K] Heat pump temperature, can be adjusted
        T_lift = T_hp - T_amb  # [K] Temperature lift of the heat pump
        COP = self.constants.eta_hp * T_hp / T_lift  # Coefficient of performance of the heat pump
        Qdot_hp = P_hp * COP  # Heat output of the heat pump
        return Qdot_hp

    def battery_model(self, x, u, p_fix, p_data):
        P_ch = u[1]  # Charging power
        P_dis = u[2]  # Discharging power
        xdot_soc = (P_ch * self.constants.eta_ch - P_dis / self.constants.eta_dis) / (self.C_bat * 3600)  # [1/s]
        return xdot_soc

    def compute_running_cost(self,u):
        """ Compute the running cost of the system"""
        P_grid_pos = u[3] # [W]
        P_grid_neg = u[4] # [W]

        price_sell_Wh = self.constants.price_sell / 1e3  # EUR/Wh
        price_buy_Wh = self.constants.price_buy / 1e3  # EUR/Wh

        cost_grid_annual = price_sell_Wh*P_grid_neg + price_buy_Wh*P_grid_pos
        cost_grid = cost_grid_annual #* self.constants.n_years  # [EUR]
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


