import casadi as ca
import numpy as np

from systemmodels.systemModel import SystemModel
from utility import Constants, Data


class StratStorageModel(SystemModel):
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
        nx = s_n + g_n + 1
        super().__init__(nx, 5, 5, 5, data, constants)

        self.stateNames = [f'T_s_{i}' for i in range(s_n)] + [f'T_g_{i}' for i in range(g_n)] + ['SOC']
        self.s_n = s_n
        self.g_n = g_n  # Set g_n attribute
        self.distance = distance  # Set distance attribute

        # Load stratification parameters
        strat_params = self.constants.strat_storage()
        self.strat_params = strat_params[s_n]

        # Initialize storage parameters
        self.tl = self.strat_params['tl_init'] * self.p_fix[0]
        self.bl = self.strat_params['bl_init'] * self.p_fix[0]
        self.ml = [ml * self.p_fix[0] for ml in self.strat_params['ml_init']]
        self.mh = self.strat_params['mh']

        # Compute storage properties
        self.compute_sto_properties(s_n)

        # sizes of the sources, storages and sinks with their default size
        self.C_hp = constants.C_hp_default * self.p_fix[1]  # W
        self.C_pv = constants.C_pv_default* self.p_fix[2]  # Wp
        self.C_wind = constants.C_wind_default * self.p_fix[3] #Wp
        self.C_bat = constants.C_bat_default * self.p_fix[4]  # Wh

        # Fixed cost function
        fixed_cost = self.build_fixed_cost()
        self.J_fix = ca.Function('J_fix', [self.p_fix], [fixed_cost], ['p_fix'], ['J_fix'])

        # grid use
        P_re = self.p_fix[2] * self.p_data[1] + self.p_fix[3] * self.p_data[2]  # P_pv + P_wind
        P_pv = self.p_fix[2] * self.p_data[1]
        P_wind = self.p_fix[3] * self.p_data[2]
        P_hh = self.p_data[3]
        P_hp = self.u[0]
        P_bat_charge = self.u[1]
        P_bat_discharge = self.u[2]
        P_bat = P_bat_charge - P_bat_discharge
        P_grid_buy = self.u[3]
        P_grid_sell = self.u[4]
        P_grid = P_grid_buy - P_grid_sell

        # dynamics
        Qdot_hp = self.compute_Qdot_hp(self.u[0], self.p_data[0])  # P_hp, T_amb, T_hp
        ODE = ca.vertcat(self.storage_model(x_sto= self.x[:-1], Tamb=self.p_data[0], Qdot_hh=self.p_data[4],Qdot_hp=Qdot_hp,
                                            p_fix = self.p_fix),
                         self.battery_model(self.x, self.u, self.p_fix, self.p_data))

        self.f = ca.Function('f', [self.x, self.u, self.p_fix, self.p_data], [ODE], ['x', 'u', 'p_fix', 'p_data'], ['xdot'])

        x0list = [273.15 + 50]*self.s_n + self.g_n * [13.5 + 273.15] + [0.5]
        self.x0 = ca.vertcat(*x0list)
        assert self.x0.shape == self.x.shape

        # Overwrite the state and control bounds
        self.lbx = ca.vertcat(self.constants.min_T1 * ca.DM.ones(1), self.constants.min_T * ca.DM.ones(s_n - 1), self.constants.min_T * ca.DM.ones(g_n), 0) # min bounds for the top layer
        self.ubx = ca.vertcat(self.constants.max_T * ca.DM.ones(s_n), self.constants.max_T * ca.DM.ones(g_n), 1)
        self.lbu = ca.DM.zeros(self.u.shape)
        self.ubu = ca.DM.ones(self.u.shape)*ca.inf

        # Heat pump and Battery constraints + Add mass flow constraints,  Grid Balance constraints
        self.const_Grid = -P_grid_buy + P_grid_sell - P_re + P_hh + P_hp - P_bat_discharge + P_bat_charge
        self.const_bat1 = self.u[1] - self.C_bat/4
        self.const_bat2 = self.u[2] - self.C_bat/4
        self.const_hp = Qdot_hp - self.C_hp  # W
        # self.const_mdot_hp = self.mdot_hp - self.constants.mdot_hp_max


        g_vec = ca.vertcat(self.const_bat1,
                           self.const_bat2,
                           self.const_hp,
                           # self.const_mdot_hp,
                           self.const_Grid)
        self.g = ca.Function('g', [self.x, self.u, self.p_fix, self.p_data], [g_vec], ['x', 'u', 'p_fix', 'p_data'], ['g'])
        self.lbg = ca.vertcat(-ca.inf*ca.DM.ones((g_vec.shape[0]-1,1)), 0)
        self.ubg = ca.DM.zeros(g_vec.shape)
        self.ng = g_vec.shape[0]

        # define output dictionary (still kind of stupid implementation)
        self.outputs['X'] = {'value': self.x, 'type': 'profile'}
        self.outputs['U']= {'value': self.u, 'type': 'profile'}
        self.outputs['P_grid']= {'value': P_grid, 'type': 'profile'}
        self.outputs['P_re']= {'value': P_re, 'type': 'profile'}
        self.outputs['P_pv']= {'value': P_pv, 'type': 'profile'}
        self.outputs['P_wind']= {'value': P_wind, 'type': 'profile'}
        self.outputs['P_hh']= {'value': P_hh, 'type': 'profile'}
        self.outputs['P_bat']= {'value': P_bat, 'type': 'profile'}
        self.outputs['P_hp']= {'value': self.u[0], 'type': 'profile'}
        self.outputs['Qdot_hp'] = {'value': Qdot_hp, 'type': 'profile'}
        self.outputs['mdot_hp'] = {'value': self.mdot_hp, 'type': 'profile'}
        # self.outputs['self.mdot_load'] = {'value': self.mdot_load, 'type': 'profile'}
        self.outputs['Qdot_load'] = {'value': self.Qdot_load, 'type': 'profile'}
        self.outputs['J_running']= {'value': self.cost_grid, 'type': 'profile'}
        self.outputs['J_fix']= {'value': fixed_cost, 'type': 'single'}
        self.outputs['p_fix']= {'value': self.p_fix, 'type': 'single'}
        self.outputs['V_s']= {'value': self.V_s, 'type': 'single', 'unit': 'm^3'}
        self.outputs['C_bat']= {'value': self.C_bat, 'type': 'single', 'unit': 'Wh'}
        self.outputs['C_pv']= {'value': self.C_pv, 'type': 'single', 'unit': 'Wp'}
        self.outputs['C_wind']= {'value': self.C_wind, 'type': 'single', 'unit': 'Wp'}
        self.outputs['C_hp']= {'value': self.C_hp, 'type': 'single', 'unit': 'W'}

    def compute_sto_properties(self, s_n):
        heights = self.mh
        lengths = [self.tl] + self.ml + [self.bl]
        self.height = self.constants.height
        self.A_bot = self.bl ** 2
        self.A_top = self.tl ** 2
        self.A_v = self.A_top + self.A_bot + np.sqrt(self.A_top * self.A_bot)
        self.V_s = 1 / 3 * self.height * self.A_v  # [m^3] Volume of the storage

        self.A_surf = self.compute_surface_areas(s_n, heights, lengths)
        self.A_q = self.compute_cross_sectional_areas(lengths)
        self.p_free = self.compute_free_convection_params(s_n, heights, self.A_q)
        self.R_s = self.compute_resistance_storage(s_n, self.A_surf)

    def compute_surface_areas(self, s_n, heights, lengths):
        A_surf = []
        for i in range(s_n-1):
            A_surf.append(2 * heights[i] * (lengths[i] + lengths[i + 1]))
        A_surf.append(2 * (self.height - sum(heights)) * (self.bl + lengths[-2]) + self.A_bot)
        return A_surf

    def compute_cross_sectional_areas(self, lengths):
        A_q = []
        for i in range(len(lengths) - 1):
            A_q.append(((lengths[i] + lengths[i + 1]) / 2) ** 2)
        return A_q

    def compute_free_convection_params(self, s_n, heights, A_q):
        p_free = []
        for i in range(s_n-1):
            p_free.append(self.constants.lambda_eff * A_q[i] / heights[i])
        p_free.append(self.constants.lambda_eff * A_q[-1] / (self.height - sum(heights)))
        return p_free

    def compute_resistance_storage(self, s_n, A_surf):
        R_s = []
        for i in range(s_n):
            R_s.append(1 / (self.constants.U_wall_noins * self.A_surf[i]))
        return R_s

    def compute_ground_properties(self, g_n, distance):
        self.volume_ground = []
        self.C_g = []  # [J/K] Thermal capacitance of the ground
        self.R_g = []  # [K/W] Thermal resistance of the ground
        self.A_i = []
        self.A_i1 = []
        self.h_i = []
        self.h_i1 = []
        self.A_s = []

        for i in range(1, self.g_n + 1):
            A_i = (self.tl + i * self.distance) ** 2 + (self.bl + i * self.distance) ** 2 + np.sqrt(
                (self.tl + i * self.distance) ** 2 * (self.bl + i * self.distance) ** 2)
            A_i1 = (self.tl + (i - 1) * self.distance) ** 2 + (self.bl + (i - 1) * self.distance) ** 2 + np.sqrt(
                (self.tl + (i - 1) * self.distance) ** 2 * (self.bl + (i - 1) * self.distance) ** 2)
            self.A_i.append(A_i)
            self.A_i1.append(A_i1)
            h_i = self.height + i * self.distance
            h_i1 = self.height + (i - 1) * self.distance
            self.h_i.append(h_i)
            self.h_i1.append(h_i1)
            volume_i = 1 / 3 * (h_i * A_i - h_i1 * A_i1)
            self.volume_ground.append(volume_i)
            C_ground_i = volume_i * self.constants.rho_g * self.constants.c_pg
            self.C_g.append(C_ground_i)

        self.d = [0] + [self.distance / 2] + [self.distance / 2 + self.distance * (i + 1) for i in
                                              range(self.g_n - 1)] + [self.distance * self.g_n]
        for i in range(1, self.g_n + 2):
            A_s = (self.bl + self.d[i]) ** 2 + 2 * (self.height + self.d[i]) * (
                        (self.tl + self.d[i]) + (self.bl + self.d[i]))
            self.A_s.append(A_s)
        self.A_s = np.array(self.A_s)

        self.distances = np.array([self.distance / 2] + [self.distance] * (self.g_n - 1) + [self.distance / 2])
        R_ground_i = (1 / self.constants.lambda_ground) * self.distances / (self.A_s)
        self.R_g = R_ground_i.tolist()
        self.C_g = np.array(self.C_g)
        return self.C_g, self.R_g

    def build_fixed_cost(self):
        CAPEX_hp = self.constants.I_hp * self.C_hp
        CAPEX_s = self.constants.I_s * self.V_s
        CAPEX_pv = self.constants.I_pv * self.C_pv
        CAPEX_wind = self.constants.I_wind * self.C_wind
        CAPEX_bat = self.constants.I_bat * self.C_bat
        CAPEX = CAPEX_hp + CAPEX_s + CAPEX_pv + CAPEX_wind + CAPEX_bat
        OPEX = 0.01 * (CAPEX_s + CAPEX_pv) + 0.02 * (CAPEX_wind + CAPEX_bat) + 0.025 * CAPEX_hp


        n = self.constants.n_years
        r = self.constants.annuity
        ANF = (r * (1 + r)**n) / ((1 + r)**n - 1)

        ANI = CAPEX * ANF
        #fixed_cost = ANI * self.constants.n_years + OPEX * self.constants.n_years
        fixed_cost = ANI + OPEX


        # append to output
        self.outputs['cost_CAPEX_hp'] = {'value': CAPEX_hp, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_s'] = {'value': CAPEX_s, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_pv'] = {'value': CAPEX_pv, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_wind'] = {'value': CAPEX_wind, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_bat'] = {'value': CAPEX_bat, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX'] = {'value': CAPEX, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_OPEX'] = {'value': OPEX, 'unit': 'EUR', 'type': 'single'}
        self.outputs['ANI'] = {'value': ANI, 'unit': 'EUR', 'type': 'single'}

        return fixed_cost

    def storage_model(self, x_sto, Tamb, Qdot_hh, Qdot_hp, p_fix):
        T_s = x_sto[:self.s_n]
        T_g = x_sto[self.s_n:]
        C_g, R_g = self.compute_ground_properties(self.g_n, self.distance)
        C_s = self.V_s * self.constants.rho * self.constants.c_p / self.s_n

        R_g[0] = self.g_n * R_g[0]  # [K/W] consider parallel resistance
        R_top = 1 / (self.constants.U_top * self.A_top)  # [K/W] Thermal resistance of the top

        # Qdot_hp = self.compute_Qdot_hp(u[0] * p_fix[1], p_data[0])  # P_hp, T_amb
        mdot_hp = Qdot_hp / (self.constants.c_p * (self.constants.T_hp - T_s[-1]))  # [kg/s]
        # T_hp = T_s[0] + 5 # [K] Heat pump temperature
        # mdot_hp = Qdot_hp / (self.constants.c_p * (T_hp - T_s[-1]))  # [kg/s]
        self.mdot_hp = mdot_hp
        self.Qdot_load = Qdot_hh  # heat demand of the households
        self.mdot_load = self.Qdot_load / (
                self.constants.c_p * (self.constants.T_sup_hh - self.constants.T_ret_hh))  # [kg/s] fix sup & ret temp
        T_lr = T_s[0] - (
                self.constants.T_sup_hh - self.constants.T_ret_hh)  # [K] return temperature from the load (no heat exchange loss)

        # Storage Dynamics
        Tdot_s1 = 1 / C_s * (self.constants.c_p * mdot_hp * (self.constants.T_hp - T_s[0])
                             - self.constants.c_p * self.mdot_load * (T_s[0] - T_s[1])
                             - 1 / R_top * (T_s[0] - Tamb)  # T_amb
                             + self.p_free[0] * (T_s[1] - T_s[0])
                             - 1 / (self.R_s[0] + R_g[0]) * (T_s[0] - T_g[0]))  # [K/s]
        Tdot_si = []
        for i in range(1, self.s_n - 1):
            Tdot_stoi = 1 / C_s * (self.constants.c_p * mdot_hp * (T_s[i - 1] - T_s[i])
                                   - self.constants.c_p * self.mdot_load * (T_s[i] - T_s[i + 1])
                                   + self.p_free[i] * (T_s[i - 1] - 2 * T_s[i] + T_s[i + 1])
                                   - 1 / (self.R_s[i] + R_g[0]) * (T_s[i] - T_g[0]))  # [K/s]
            Tdot_si.append(Tdot_stoi)
        Tdot_sm = 1 / C_s * (self.constants.c_p * mdot_hp * (T_s[self.s_n - 2] - T_s[self.s_n - 1])
                             - self.constants.c_p * self.mdot_load * (x_sto[self.s_n - 1] - T_lr)
                             - self.p_free[-1] * (T_s[self.s_n - 1] - T_s[self.s_n - 2])
                             - 1 / (self.R_s[-1] + R_g[0]) * (T_s[self.s_n - 1] - T_g[0]))  # [K/s]
        # Ground Dynamics
        Tdot_g1 = 0
        for i in range(self.s_n):
            Tdot_g1 += 1 / (R_g[0] + self.R_s[i]) * (T_s[i] - T_g[0])
        Tdot_g1 = 1 / C_g[0] * Tdot_g1

        Tdot_gi = []
        for i in range(1, self.g_n - 1):
            Tdot_i = 1 / C_g[i] * ((1 / R_g[i]) * (T_g[i - 1] - T_g[i])
                                   + (1 / R_g[i + 1]) * (T_g[i + 1] - T_g[i]))
            Tdot_gi.append(Tdot_i)
        Tdot_gN = 1 / C_g[self.g_n - 1] * (1 / R_g[self.g_n - 1] * (T_g[self.g_n - 2] - T_g[self.g_n - 1])
                                           + 1 / R_g[self.g_n] * (self.constants.T_bc - T_g[self.g_n - 1]))

        Tdot_g = ca.vertcat(Tdot_g1, *Tdot_gi, Tdot_gN)
        Tdot_s = ca.vertcat(Tdot_s1, *Tdot_si, Tdot_sm)
        Tdot = ca.vertcat(Tdot_s, Tdot_g)
        return Tdot
