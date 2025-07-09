import numpy as np
import casadi as ca
from scipy.optimize import fsolve

class Constants:
    def __init__(self):
        self.height = 16.0
        self.lambda_eff = 0.644
        self.U_wall_noins = 90
        self.T_bc = 273.15 + 10.0
        self.rho_g = 2000.0
        self.c_pg = 700.0
        self.lambda_ground = 0.47

class ThermalModel:
    def __init__(self, nk, s_n, g_n, distance, constant=None):
        self.constants = Constants()
            
        self.nk = nk
        self.distance = distance
        self.g_n = g_n
        self.s_n = s_n

        self.initialize_parameters()
        self.simulation_params()
        self.mh, self.ml = self._calculate_geometric_properties(s_n)
        self.diffuser_map = self._map_diffusers_to_layers()
        
        self.define_symbolic_variables()
        self.compute_sto_properties(self.distance)
        self.C_g, self.R_g = self.compute_ground_properties(self.g_n, self.distance)
        self.f = self.create_ode_function()

    def initialize_parameters(self):
        self.rho = 1000
        self.c_p = 4200
        self.tl, self.bl = 91.0, 26.0
        self.height = 16.0
        self.lambda_eff = 0.644
        self.U_top = 0.186
        self.A_top = self.tl**2

    def simulation_params(self):
        self.h = 600 # 10-minute step size

    def _calculate_geometric_properties(self, s_n):
        # Solves for layer heights and intermediate lengths for s_n equal-volume segments
        V_total = (1/3) * self.height * (self.tl**2 + self.bl**2 + self.tl * self.bl)
        V_segment = V_total / s_n

        def equations(vars):
            eqs = []
            lengths = [self.tl] + list(vars[0::2])
            heights = list(vars[1::2])
            
            for i in range(s_n - 1):
                A_upper = lengths[i]**2
                A_lower = lengths[i+1]**2
                eq = (1/3) * heights[i] * (A_upper + A_lower + np.sqrt(A_upper * A_lower)) - V_segment
                eqs.append(eq)
            
            slope = (self.tl - self.bl) / self.height
            total_h = 0
            for i in range(s_n - 1):
                total_h += heights[i]
                l_calc = self.tl - slope * total_h
                eqs.append(l_calc - lengths[i+1])
            return eqs

        initial_guess = []
        for i in range(s_n - 1):
            initial_guess.extend([self.tl - (i+1)*(self.tl-self.bl)/(s_n), self.height/s_n])
        
        solution = fsolve(equations, initial_guess)
        
        solved_ml = solution[0::2]
        solved_mh = solution[1::2]
        mh_final = self.height - sum(solved_mh)
        
        return list(solved_mh) + [mh_final], list(solved_ml)

    def _map_diffusers_to_layers(self):
        # Maps physical diffuser depths to the calculated model layers
        diffuser_depths = {'top': 15.3, 'mid': 10.9, 'bot': 2.825}
        layer_boundaries_from_top = np.cumsum([0] + self.mh)
        
        mapping = {}
        for name, depth_from_bottom in diffuser_depths.items():
            depth_from_top = self.height - depth_from_bottom
            for i in range(self.s_n):
                if layer_boundaries_from_top[i] <= depth_from_top < layer_boundaries_from_top[i+1]:
                    mapping[name] = i
                    break
        print(f"Diffuser mapping to layers: {mapping}")
        return mapping

    def define_symbolic_variables(self):
        self.T_s = [ca.SX.sym(f'T_s{i}') for i in range(self.s_n)]
        self.T_g = [ca.SX.sym(f'T_g{i}') for i in range(self.g_n)]
        self.x = ca.vertcat(*self.T_s, *self.T_g)
        self.u = ca.SX.sym('u_dummy')
        
        self.T_amb, self.T_bc = ca.SX.sym('T_amb'), ca.SX.sym('T_bc')
        self.mdot_ch_top, self.T_in_ch_top, self.mdot_dis_top = ca.SX.sym('mdot_ch_top'), ca.SX.sym('T_in_ch_top'), ca.SX.sym('mdot_dis_top')
        self.mdot_ch_mid, self.T_in_ch_mid, self.mdot_dis_mid = ca.SX.sym('mdot_ch_mid'), ca.SX.sym('T_in_ch_mid'), ca.SX.sym('mdot_dis_mid')
        self.mdot_ch_bot, self.T_in_ch_bot, self.mdot_dis_bot = ca.SX.sym('mdot_ch_bot'), ca.SX.sym('T_in_ch_bot'), ca.SX.sym('mdot_dis_bot')
        
        self.par = ca.vertcat(self.T_amb, self.T_bc, self.mdot_ch_top, self.T_in_ch_top, self.mdot_dis_top, self.mdot_ch_mid, self.T_in_ch_mid, self.mdot_dis_mid, self.mdot_ch_bot, self.T_in_ch_bot, self.mdot_dis_bot)

    def compute_sto_properties(self, distance):
        lengths = [self.tl] + self.ml + [self.bl]
        self.V_layers = [(self.mh[i]/3) * (lengths[i]**2 + lengths[i+1]**2 + lengths[i]*lengths[i+1]) for i in range(self.s_n)]
        self.C_s = [v * self.rho * self.c_p for v in self.V_layers]
        self.A_q = [lengths[i+1]**2 for i in range(self.s_n - 1)]
        self.p_free = [self.lambda_eff * self.A_q[i] / ((self.mh[i] + self.mh[i+1])/2) for i in range(self.s_n - 1)]
        A_surf = [2 * self.mh[i] * (lengths[i] + lengths[i+1]) for i in range(self.s_n)]
        self.R_s = 1 / (self.constants.U_wall_noins * np.array(A_surf))
        R_int = 0.5 * self.distance / (self.constants.lambda_ground * np.array(A_surf))
        # 0.5*Δr because the centre of ground node T_g,1 is Δr/2 away

        self.R_int_layer = R_int           # store for later use

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

        # self.d = [0] + [self.distance / 2] + [self.distance / 2 + self.distance * (i + 1) for i in
        #                                       range(self.g_n - 1)] + [self.distance * self.g_n]
        # for i in range(1, self.g_n + 2):
        #     A_s = (self.bl + self.d[i]) ** 2 + 2 * (self.height + self.d[i]) * (
        #                 (self.tl + self.d[i]) + (self.bl + self.d[i]))
        #     self.A_s.append(A_s)
        # self.A_s = np.array(self.A_s)
        # perimeter of a square frustum at radius offset d_i
        def perimeter(side_len):          # side_len = tl + d_i  or  bl + d_i
            return 4 * side_len

        self.A_side = []
        for i in range(1, self.g_n + 2):
            # average perimeter of inner and outer square
            P_avg = 0.5 * (perimeter(self.tl + (i-1)*self.distance) +
                        perimeter(self.tl +  i   *self.distance))
            A_side = P_avg * self.height                      # lateral area only
            self.A_side.append(A_side)

        self.distances = np.array([self.distance/2] +
                                [self.distance]*(self.g_n-1) +
                                [self.distance/2])

        self.R_g = (self.distances /
                    (self.constants.lambda_ground * np.array(self.A_side)))

        # self.distances = np.array([self.distance / 2] + [self.distance] * (self.g_n - 1) + [self.distance / 2])
        # R_ground_i = (1 / self.constants.lambda_ground) * self.distances / (self.A_s)
        # self.R_g = R_ground_i.tolist()
        
        self.C_g = np.array(self.C_g)
        return self.C_g, self.R_g

    def create_ode_function(self):
        T_s, T_g = self.T_s, self.T_g
        T_amb, T_bc = self.par[0], self.par[1]
        mdot_ch_top, T_in_ch_top, mdot_dis_top = self.par[2], self.par[3], self.par[4]
        mdot_ch_mid, T_in_ch_mid, mdot_dis_mid = self.par[5], self.par[6], self.par[7]
        mdot_ch_bot, T_in_ch_bot, mdot_dis_bot = self.par[8], self.par[9], self.par[10]

        R_top = 1 / (self.U_top * self.A_top)
        cp = self.c_p

        # --- Net mass flow at the boundary BELOW each layer (positive is downward) ---
        v = [ca.SX(0)] * self.s_n
        v_flow = 0
        for i in range(self.s_n):
            if i == self.diffuser_map.get('top'): v_flow += mdot_ch_top - mdot_dis_top
            if i == self.diffuser_map.get('mid'): v_flow += mdot_ch_mid - mdot_dis_mid
            if i == self.diffuser_map.get('bot'): v_flow += mdot_ch_bot - mdot_dis_bot
            v[i] = v_flow
        
        # --- Advection energy transfer between layers ---
        Q_adv = []
        for i in range(self.s_n - 1):
            # Net energy flow across boundary i
            Q_net_adv = cp * (ca.fmax(0, v[i]) * T_s[i] - ca.fmax(0, -v[i]) * T_s[i+1])
            Q_adv.append(Q_net_adv)

        Tdot_s = []
        for i in range(self.s_n):
            # Advection from above/to below
            adv_in = Q_adv[i-1] if i > 0 else 0
            adv_out = Q_adv[i] if i < self.s_n - 1 else 0
            
            # Conduction terms
            cond_from_above = self.p_free[i-1] * (T_s[i-1] - T_s[i]) if i > 0 else 0
            cond_from_below = self.p_free[i] * (T_s[i+1] - T_s[i]) if i < self.s_n - 1 else 0
            
            # Loss terms
            # loss_to_ground = (1/(self.R_s[i] + self.R_g[0])) * (T_s[i] - T_g[0])
            loss_to_ground = (1/(self.R_s[i] + self.R_int_layer[i])) * (T_s[i] - T_g[0])
            loss_to_ambient = (1/R_top) * (T_s[0] - T_amb) if i == 0 else 0
            
            # Port source/sink terms
            source_sink = 0
            if i == self.diffuser_map.get('top'): source_sink += cp * (mdot_ch_top * T_in_ch_top - mdot_dis_top * T_s[i])
            if i == self.diffuser_map.get('mid'): source_sink += cp * (mdot_ch_mid * T_in_ch_mid - mdot_dis_mid * T_s[i])
            if i == self.diffuser_map.get('bot'): source_sink += cp * (mdot_ch_bot * T_in_ch_bot - mdot_dis_bot * T_s[i])

            dTi = (1/self.C_s[i]) * (adv_in - adv_out + cond_from_above + cond_from_below - loss_to_ground - loss_to_ambient + source_sink)
            Tdot_s.append(dTi)

        # --- Ground Dynamics ---
        Tdot_g0 = 0
        for i in range(self.s_n):
            # Tdot_g0 += 1 / (self.R_g[0] + self.R_s[i]) * (T_s[i] - T_g[0])
            Tdot_g0 += (T_s[i] - T_g[0]) / (self.R_s[i] + self.R_int_layer[i])
        Tdot_g0 += (T_g[1] - T_g[0]) / self.R_g[0]
        Tdot_g0 /= self.C_g[0]


        Tdot_gi = []
        for i in range(1, self.g_n - 1):
            Tdot_i = 1 / self.C_g[i] * ((1 / self.R_g[i]) * (T_g[i - 1] - T_g[i])
                                   + (1 / self.R_g[i + 1]) * (T_g[i + 1] - T_g[i]))
            Tdot_gi.append(Tdot_i)
        Tdot_gN = 1 / self.C_g[self.g_n - 1] * (1 / self.R_g[self.g_n - 1] * (T_g[self.g_n - 2] - T_g[self.g_n - 1])
                                           + 1 / self.R_g[self.g_n] * (self.constants.T_bc - T_g[self.g_n - 1]))

        Tdot_g = ca.vertcat(Tdot_g0, *Tdot_gi, Tdot_gN)
        Tdot_s_vec = ca.vertcat(*Tdot_s)
        Tdot = ca.vertcat(Tdot_s_vec, Tdot_g)
        
        return ca.Function('f', [self.x, self.u, self.par], [Tdot])
