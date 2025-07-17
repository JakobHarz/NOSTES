import casadi as ca

from results._paperResults.systemModel_nosto import SystemModel
from utility_nosto import Constants, Data


class NoStorageModel(SystemModel):
    def __init__(self, data: Data, constants: Constants):
        """ No storage Model
        x (): x_soc
        u (5): P_hp [W], P_ch [W], P_dis [W], P_Grid_buy [W], P_Grid_sell [W]
        p_fix (4): - HP capacity [W]
                   - PV Install capacity [W]
                   - Wind Install capacity [W]
                   - Battery capacity [Wh]

        p_data (5): - T_amb ambient temperature
                    - P_pv power of the pv
                    - P_wind power of the wind
                    - P_HH electric household power demand
                    - Qdot_HH heat demand of the household
            """
        nx = 1
        super().__init__(nx, 5, 5, 4, data, constants)

        self.stateNames = ['SOC']

        # sizes of the sources, storages and sinks with their default size
        self.C_hp = constants.C_hp_default * self.p_fix[0]  # W
        self.C_pv = constants.C_pv_default* self.p_fix[1]  # Wp
        self.C_wind = constants.C_wind_default * self.p_fix[2] #Wp
        self.C_bat = constants.C_bat_default * self.p_fix[3]  # Wh

        # Fixed cost function
        fixed_cost = self.build_fixed_cost()
        self.J_fix = ca.Function('J_fix', [self.p_fix], [fixed_cost], ['p_fix'], ['J_fix'])

        # grid use
        P_re = self.p_fix[1] * self.p_data[1] + self.p_fix[2] * self.p_data[2]  # P_pv + P_wind
        P_pv = self.p_fix[1] * self.p_data[1]
        P_wind = self.p_fix[2] * self.p_data[2]
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
        Qdot_load = self.p_data[4]  # heat demand of the households
        ODE = ca.vertcat(self.battery_model(self.x, self.u, self.p_fix, self.p_data))

        self.f = ca.Function('f', [self.x, self.u, self.p_fix, self.p_data], [ODE], ['x', 'u', 'p_fix', 'p_data'], ['xdot'])

        x0list = [0.5]
        self.x0 = ca.vertcat(*x0list)
        assert self.x0.shape == self.x.shape

        # Overwrite the state and control bounds
        self.lbx = ca.vertcat(0) # min bounds for the top layer
        self.ubx = ca.vertcat(1)
        self.lbu = ca.DM.zeros(self.u.shape)
        self.ubu = ca.DM.ones(self.u.shape)*ca.inf

        # Heat pump and Battery constraints + Add mass flow constraints,  Grid Balance constraints
        self.const_Grid = -P_grid_buy + P_grid_sell - P_re + P_hh + P_hp - P_bat_discharge + P_bat_charge
        self.const_bat1 = self.u[1] - self.C_bat/4
        self.const_bat2 = self.u[2] - self.C_bat/4
        self.const_hp = Qdot_hp - self.C_hp  # W
        self.const_heat = Qdot_load - Qdot_hp  # heat demand of the households
        # self.const_mdot_hp = self.mdot_hp - self.constants.mdot_hp_max


        g_vec = ca.vertcat(self.const_bat1,
                           self.const_bat2,
                           self.const_hp,
                           # self.const_mdot_hp,
                           self.const_Grid,
                           self.const_heat)
        self.g = ca.Function('g', [self.x, self.u, self.p_fix, self.p_data], [g_vec], ['x', 'u', 'p_fix', 'p_data'], ['g'])
        self.lbg = ca.vertcat(-ca.inf, -ca.inf, -ca.inf, 0, 0)
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
        # self.outputs['mdot_hp'] = {'value': self.mdot_hp, 'type': 'profile'}
        # self.outputs['self.mdot_load'] = {'value': self.mdot_load, 'type': 'profile'}
        # self.outputs['Qdot_load'] = {'value': self.Qdot_load, 'type': 'profile'}
        self.outputs['J_running']= {'value': self.cost_grid, 'type': 'profile'}
        self.outputs['J_fix']= {'value': fixed_cost, 'type': 'single'}
        self.outputs['p_fix']= {'value': self.p_fix, 'type': 'single'}
        self.outputs['C_bat']= {'value': self.C_bat, 'type': 'single', 'unit': 'Wh'}
        self.outputs['C_pv']= {'value': self.C_pv, 'type': 'single', 'unit': 'Wp'}
        self.outputs['C_wind']= {'value': self.C_wind, 'type': 'single', 'unit': 'Wp'}
        self.outputs['C_hp']= {'value': self.C_hp, 'type': 'single', 'unit': 'W'}

    def build_fixed_cost(self):
        CAPEX_hp = self.constants.I_hp * self.C_hp
        CAPEX_pv = self.constants.I_pv * self.C_pv
        CAPEX_wind = self.constants.I_wind * self.C_wind
        CAPEX_bat = self.constants.I_bat * self.C_bat
        CAPEX = CAPEX_hp + CAPEX_pv + CAPEX_wind + CAPEX_bat
        OPEX = 0.01 * (CAPEX_pv) + 0.02 * (CAPEX_wind + CAPEX_bat) + 0.025 * CAPEX_hp


        n = self.constants.n_years
        r = self.constants.annuity
        ANF = (r * (1 + r)**n) / ((1 + r)**n - 1)

        ANI = CAPEX * ANF
        #fixed_cost = ANI * self.constants.n_years + OPEX * self.constants.n_years
        fixed_cost = ANI + OPEX


        # append to output
        self.outputs['cost_CAPEX_hp'] = {'value': CAPEX_hp, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_pv'] = {'value': CAPEX_pv, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_wind'] = {'value': CAPEX_wind, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX_bat'] = {'value': CAPEX_bat, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_CAPEX'] = {'value': CAPEX, 'unit': 'EUR', 'type': 'single'}
        self.outputs['cost_OPEX'] = {'value': OPEX, 'unit': 'EUR', 'type': 'single'}
        self.outputs['ANI'] = {'value': ANI, 'unit': 'EUR', 'type': 'single'}

        return fixed_cost