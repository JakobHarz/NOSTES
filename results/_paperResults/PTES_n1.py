import numpy as np
import casadi as ca

class ThermalModel:
    def __init__(self, tl, bl, iter, distance, U_wall):
        self.tl = tl # top length
        self.bl = bl # bottom length
        self.distance = distance
        self.iter = iter
        self.U_wall = U_wall
        self.initialize_parameters()
        self.calculate_ground_properties()
        self.define_symbolic_variables()
        self.calculate_storage_properties()
        self.formulate_state_space_model()

    def initialize_parameters(self):
        self.rho = 1000 # [kg/m^3] Density of water
        # self.c_p = 4186 # [J/kgK] Specific heat capacity of water @ 50degC
        # self.rho_g = 2000 # [kg/m^3] Density of ground
        # self.c_pg = 880 # ground specific heat capacity J/kgK 
        # self.lambda_ground = 1.5 # W/mK

        """Toward efficient numerical modeling and analysis of large-scale thermal
        energy storage for renewable district heating
        0.166 W/m^2K ≤ Utop ≤ 0.30 W/m^2K -> 0.186
        0.3 W/mK ≤ λg ≤ 0.5 W/mk -> 0.47
        """
        self.c_p = 4200 # [J/kgK] Specific heat capacity of water @ 50degC
        self.rho_g = 2000 # [kg/m^3] Density of ground(dry sand) 
        self.c_pg = 700 # ground (dry sand) specific heat capacity J/kgK 
        self.lambda_ground = 0.47 # W/mK
        self.U_top = 0.186 # [W/m^2K] Heat transfer coefficient of the top
    
        #self.A_top = 8100 # [m^2] Surface area of the top (Dronninglund) 90, 26 
        self.A_top = 153.3**2 # [m^2] Surface area of the top (Vojens)
        self.A_bot = 73.2**2 # [m^2]
        self.height = 15 # [m] Height of the storage
        self.A_v = self.A_top + self.A_bot + np.sqrt(self.A_top * self.A_bot)
        self.A_surf = self.A_bot + 2 * self.height * self.tl + 2 * self.height * self.bl # [m^2] surface area connection to the ground
        self.V = 1/3 * self.height * self.A_v # [m^3] Volume of the storage

    def calculate_ground_properties(self):
        self.volume_ground = []
        self.C_ground = [] # [J/K] Thermal capacitance of the ground
        self.R_ground = [] # [K/W] Thermal resistance of the ground
        self.A_i = []
        self.A_i1 = []
        self.h_i = []
        self.h_i1 = []
        self.A_s = []
        #self.A_si = []

        volume_ground = 1/3 * self.height * (self.tl**2 + self.bl**2 + np.sqrt(self.tl**2 * self.bl**2))
        for i in range(1, self.iter + 1):
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
            C_ground_i = volume_i * self.rho_g * self.c_pg
            self.C_ground.append(C_ground_i)
        
        self.d = [0] + [self.distance / 2] + [self.distance / 2 + self.distance * (i+1) for i in range(self.iter-1)] + [self.distance *self.iter]
        for i in range(1, self.iter + 2):
            A_s = (self.bl + self.d[i]) ** 2 + 2 * (self.height + self.d[i]) * ((self.tl + self.d[i]) + (self.bl + self.d[i]))
            self.A_s.append(A_s)
        self.A_s = np.array(self.A_s)

        self.distances = np.array([self.distance / 2] + [self.distance] * (self.iter - 1) + [self.distance / 2])
        R_ground_i = (1 / self.lambda_ground) * self.distances / (self.A_s)
        self.R_ground = R_ground_i.tolist()
        self.C_ground = np.array(self.C_ground)

    def define_symbolic_variables(self):
        self.T_sto = ca.SX.sym('T_sto')
        self.T_ground = [ca.SX.sym(f'T_ground_{i}') for i in range(1, self.iter+1)]
        self.state = ca.vertcat(self.T_sto, *self.T_ground)
        self.Qdot_sto = ca.SX.sym('Qdot_sto')
        self.Qdot_HP = ca.SX.sym('Qdot_HP')
        self.con = ca.vertcat(self.Qdot_sto, self.Qdot_HP)
        self.T_bc = ca.SX.sym('T_bc')
        self.Qdot_load = ca.SX.sym('Qdot_load')
        self.c_el = ca.SX.sym('c_el')
        self.COP = ca.SX.sym('COP')
        self.T_amb = ca.SX.sym('T_amb')
        # one more parameter
        self.par = ca.vertcat(self.T_bc, self.Qdot_load, self.c_el, self.COP, self.T_amb)

    def calculate_storage_properties(self):
        self.C_sto = self.rho * self.V * self.c_p # [J/K] Thermal capacitance of the storage
        self.R_sto = 1 / (self.U_wall * self.A_surf) # [K/W] Thermal resistance of the wall
        self.R_top = 1 / (self.U_top * self.A_top) # [K/W] Thermal resistance of the top

    def formulate_state_space_model(self):
        A = [
            [-1 / self.C_sto * (1 / (self.R_sto + self.R_ground[0]) + 1 / self.R_top), 1 / (self.C_sto * (self.R_sto + self.R_ground[0]))],
            [1 / (self.C_ground * (self.R_sto + self.R_ground[0])), -1 / self.C_ground * (1 / (self.R_sto + self.R_ground[0]) + 1 / self.R_ground[1])]
        ]
        self.A = ca.SX(A)

        B = [
            [1 / self.C_sto, 0],
            [0, 0]
        ]
        self.B = ca.SX(B)

        D = [
            [0, 0, 0, 0, 1 / (self.C_sto * self.R_top)],
            [1 / (self.C_ground * self.R_ground[1]), 0, 0, 0, 0]
        ]
        self.D = ca.SX(D)

        self.Tdot = self.A @ self.state + self.B @ self.con + self.D @ self.par # [K/s]
        self.f = ca.Function('f', [self.state, self.con, self.par], [self.Tdot])


#model = ThermalModel(height = 16, radius = 34.6, iter=5, distance = 0.5, U_wall = 80)
#model = ThermalModel(height = 16, radius = 45, iter=5, distance = 0.5, U_wall = 80)
#model = ThermalModel(height = 16, radius = 55, iter=5, distance = 0.5, U_wall = 80)
#model = ThermalModel(height = 16, radius = 63.1, iter=5, distance = 0.5, U_wall = 80)
model = ThermalModel(153.3, 73.2, iter=1, distance = 0.5, U_wall = 80) # Dronninglund
