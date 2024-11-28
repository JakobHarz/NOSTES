import casadi as ca
import numpy as np

from systemmodels.systemModel import SystemModel, Data
from utility import Constants


class SimpleStorageModel(SystemModel):
    def __init__(self, data: Data):
        """ Simple Model for a system that consideres a heatpump and a heat storage
        with a single state variable and simple heat loss

        x (2): T_s [K], x_soc [-]
        u (5): P_hp [W], P_ch [W], P_dis [W], P_Grid_buy [W], P_Grid_sell [W]
        p_fix (5): - Storage size [m^3]
                   - HP capacity [W]
                   - PV Install capacity [W]
                   - Wind Install capacity [W]
                   - Battery capacity [Wh]
        """
        super().__init__(2, 5, 5, 5, data)

        self.stateNames = ['T_s', 'SOC']
        self.params = Constants()  # Initialize SystemParameters

        # storage parameter definition
        self.V_s, self.A_s, self.C_s, self.R_s = self.cyl_sto_params()

        # grid use
        P_re = self.p_fix[2] * self.p_data[1] + self.p_fix[3] * self.p_data[2]  # P_pv + P_wind
        P_hh = self.p_data[3]
        P_hp = self.u[0]
        P_bat_charge = self.u[1]
        P_bat_discharge = self.u[2]
        P_bat = P_bat_charge - P_bat_discharge
        P_grid_buy = self.u[3]
        P_grid_sell = self.u[4]
        P_grid = P_grid_buy - P_grid_sell

        # Fixed cost function
        J_fix = self.compute_fixed_cost(self.p_fix, annuity = 0.04, n_year = 20)
        self.J_fix = ca.Function('J_fix', [self.p_fix], [J_fix], ['p_fix'], ['J_fix'])

        # dynamics
        ODE = ca.vertcat(self.storage_model(self.x, self.u, self.p_fix, self.p_data),
                         self.battery_model(self.x, self.u, self.p_fix, self.p_data))

        self.f = ca.Function('f', [self.x, self.u, self.p_fix, self.p_data], [ODE], ['x', 'u', 'p_fix', 'p_data'], ['xdot'])

        self.x0 = ca.vertcat([self.params.min_T, 0.5])

        # Overwrite the state and control bounds
        self.lbx = ca.vertcat([self.params.min_T, 0])
        self.ubx = ca.vertcat([self.params.max_T, 1])
        self.lbu = ca.DM.zeros(self.u.shape)
        self.ubu = ca.inf*ca.DM.ones(self.u.shape)

        # Heat pump and Battery, and Grid Balance constraints
        const_Grid = -P_grid_buy + P_grid_sell - P_re + P_hh + P_hp - P_bat_discharge + P_bat_charge
        const_bat1 = self.u[1] - self.C_bat/4
        const_bat2 = self.u[2] - self.C_bat/4
        const_hp = self.u[0] - self.C_hp  # W
        g_vec = ca.vertcat(const_bat1, const_bat2, const_hp, const_Grid)
        self.g = ca.Function('g', [self.x, self.u, self.p_fix, self.p_data], [g_vec], ['x', 'u', 'p_fix', 'p_data'], ['g'])
        self.lbg = ca.vertcat(-ca.inf*ca.DM.ones((g_vec.shape[0]-1,1)), 0)
        self.ubg = ca.DM.zeros(g_vec.shape)
        self.ng = g_vec.shape[0]

        # define output dictionary (still kind of stupid implementation)
        self.outputs = {'X': {'value': self.x, 'type': 'profile'},
                        'U': {'value': self.u, 'type': 'profile'},
                        'P_grid': {'value': P_grid, 'type': 'profile'},
                        'P_re': {'value': P_re, 'type': 'profile'},
                        'P_hh': {'value': P_hh, 'type': 'profile'},
                        'P_bat': {'value': P_bat, 'type': 'profile'},
                        'P_hp': {'value': self.u[0], 'type': 'profile'},
                        'J_running': {'value': cost_grid, 'type': 'profile'},
                        'J_fix': {'value': J_fix, 'type': 'single'},
                        'p_fix': {'value': self.p_fix, 'type': 'single'},
                        'V_s': {'value': self.V_s, 'type': 'single'},
                        'C_bat': {'value': self.C_bat, 'type': 'single'},
                        }

    def cyl_sto_params(self):
        radius = 63.1 * self.p_fix[0]
        V_cyl = np.pi * self.params.height * radius**2  # [m^3] Volume of the storage
        A_cyl = 2 * np.pi * (self.params.radius * self.params.height + radius**2)  # [m^2] Surface area of the storage
        C_s = self.params.rho * self.params.c_p * V_cyl  # [J/K] Thermal capacitance of the storage
        R_s = 1 / (self.params.U_wall_ins * A_cyl)  # [K/W] Thermal resistance of the wall
        return V_cyl, A_cyl, C_s, R_s

    def compute_fixed_cost(self, p_fix, annuity, n_year):
        """ Compute the investment cost of the system"""
        I_hp,  I_s, I_pv, I_wind, I_bat = self.params.investment()
        self.C_pv = 34.69 * 1e6 * self.p_fix[2]  # Wp
        self.C_wind = 7.14 * 5 * 1e6 * self.p_fix[3] #Wp
        self.C_bat = 2e6 * self.p_fix[4]  # Wh
        self.C_hp = 2e7 * self.p_fix[1]  # W
        CAPEX_hp = I_hp * self.C_hp
        CAPEX_s = I_s * self.V_s
        CAPEX_pv = I_pv * self.C_pv
        CAPEX_wind = I_wind * self.C_wind
        CAPEX_bat = I_bat * self.C_bat
        CAPEX = CAPEX_hp + CAPEX_s + CAPEX_pv + CAPEX_wind + CAPEX_bat
        OPEX = 0.01 * (CAPEX_hp + CAPEX_s + CAPEX_pv) + 0.02 * (CAPEX_wind + CAPEX_bat) 
        annuity_cost = annuity * CAPEX * (((1 + annuity)**n_year - 1) / annuity)
        fixed_cost = CAPEX + OPEX * n_year + annuity_cost
        return fixed_cost

    def storage_model(self, x, u, p_fix, p_data):
        """ Define the storage model dynamics."""
        Qdot_hp = self.compute_Qdot_hp(self.u[0] * self.p_fix[1], self.p_data[0]) # P_hp, T_amb
        Qdot_loss =  1/self.R_s * (self.x[0] - self.p_data[0])  # heat we lose to the environment
        Qdot_load = self.p_data[4]  # heat demand of the households
        Tdot = 1/self.C_s * (Qdot_hp - Qdot_loss - Qdot_load)  # [K/s]
        return Tdot
