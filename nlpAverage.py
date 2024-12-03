import numpy as np

from systemmodels.averagedstratstoragemodel import AveragedStratStorageModel
from systemmodels.systemModel import SystemModel, Data
from utility import Constants, NumpyStruct
from nlp import STESNLP
import casadi as ca
from casadi.tools import struct_symSX, entry


class AverageSTESNLP(STESNLP):

    def build_NLP(self):

        averagesystem = self.system
        assert isinstance(averagesystem, AveragedStratStorageModel)

        N = self.N  # for readability
        T = 365 * 24
        h = T / N  # [h] fine grid step size in hours
        H = 24  # [h] coarse grid step size in hours
        M = int(T / H)  # number of coarse grid steps
        ratio = int(H / h)  # number of fine grid steps in one coarse grid step
        H_sec = H * 3600
        h_sec = h * 3600

        self.h = h
        self.T = T
        self.h_sec = h_sec

        self.params = Constants()  # Initialize SystemParameters

        # empty list for constraint, (could be replaced with casadi structure at some point)
        G = []
        lbG = []
        ubG = []

        # Define the decision variables
        w = struct_symSX([entry('X_bat', shape=averagesystem.x_bat.shape, repeat=N + 1),  # fine grid
                          entry('X_sto', shape=averagesystem.x_sto.shape, repeat=M + 1),  # coarse grid
                          entry('U', shape=averagesystem.u.shape, repeat=N),  # constant control over the interval
                          entry('p_fix', shape=averagesystem.p_fix.shape), ])  # constant parameter

        # Load every Data over the time horizon
        p_data_n = ca.horzcat(*[ca.vertcat(*self.data.getDataAtTime(n * h)) for n in range(N)])

        # initialization of the variables
        w0 = w(0)
        w0['X_bat'] = self.system.x0[-1]  # initial battery soc
        w0['X_sto'] = self.system.x0[:-1]  # initial temperature of storage
        w0['U', :, 0] = 3e6
        w0['U', :, 1] = 0
        w0['U', :, 2] = 0
        w0['p_fix'] = [1, 1, 1, 1, 1]
        w0['U', :, 0] = ca.horzsplit((p_data_n[1, :] + p_data_n[2, :] - p_data_n[3, :]) / 3)  # P_pv + P_wind - P_load
        w0['U', :, 1] = ca.horzsplit((p_data_n[1, :] + p_data_n[2, :] - p_data_n[3, :]) / 3)
        w0['U', :, 2] = ca.horzsplit((p_data_n[3, :] - p_data_n[1, :] - p_data_n[2, :]) / 3)  # P_load - P_pv - P_wind

        # create the upper and lower bounds for the variables
        lbw = w(-ca.inf)
        ubw = w(ca.inf)

        # overwrite upper and lower bounds for the states and controls
        lbw['X_sto'] = averagesystem.lbx[:-1]
        ubw['X_sto'] = averagesystem.ubx[:-1]
        lbw['X_bat'] = averagesystem.lbx[-1]
        ubw['X_bat'] = averagesystem.ubx[-1]
        lbw['U'] = averagesystem.lbu
        ubw['U'] = averagesystem.ubu
        lbw['p_fix'] = averagesystem.lbp
        ubw['p_fix'] = averagesystem.ubp

        # periodicty constraints
        G.append(w['X_sto', 0] - w['X_sto', M])
        G.append(w['X_bat', 0] - w['X_bat', N])
        lbG.append(ca.DM.zeros(averagesystem.x.shape))
        ubG.append(ca.DM.zeros(averagesystem.x.shape))

        # fixed costs
        J_fix = averagesystem.J_fix(w['p_fix'])
        J_running = 0
        # J_reg = (w['p_fix'] - w0['p_fix']).T@(w['p_fix'] - w0['p_fix'])
        J_reg = 0

        # iterate the intervals of the fine grid
        outputs = []
        p_fix = w['p_fix']
        for n in range(N):
            # implicit euler integration
            x_bat_n = w['X_bat', n]
            x_bat_np1 = w['X_bat', n + 1]
            u_n = w['U', n]  # constant control over the interval
            t_n = h * n
            p_data_n = ca.vertcat(*averagesystem.data.getDataAtTime(t_n))

            # constraint satisfaction for fine grid
            G.append(averagesystem.g_fine(x_bat_n, u_n, p_fix, p_data_n))
            lbG.append(averagesystem.lbg_fine)
            ubG.append(averagesystem.ubg_fine)

            # explicit euler step, linear dynamics
            G.append(x_bat_np1 - x_bat_n - h_sec * averagesystem.f_bat(u_n, p_fix))
            lbG.append(ca.DM.zeros(averagesystem.x_bat.shape))
            ubG.append(ca.DM.zeros(averagesystem.x_bat.shape))

            # cost of the interval
            J_running += averagesystem.J_running(u_n) * h

            # regularization
            W_reg = ca.diag([1E-7, 1E-7, 1E-7, 1E-7, 1E-7])
            J_reg += 1E-10 * u_n.T @ W_reg.T @ W_reg @ u_n * h

            # approximate the state of the storage with a linear interpolation between the two nodes
            m, dk = np.divmod(n, ratio)
            x_sto_n = w['X_sto', m] + dk / ratio * (w['X_sto', m + 1] - w['X_sto', m])

            # store the outputs of the interval
            x_bat_n = ca.vertcat(x_sto_n, x_bat_n)
            outputs.append(averagesystem.outputFunction(x_bat_n, u_n, p_fix, p_data_n))

        # iterate the intervals of the coarse grid
        for m in range(M):
            # times at which the coarse interval starts and ends
            t_m = H * m
            t_mp1 = H * (m + 1)

            # start index of the fine grid at the coarse grid
            n = m * ratio

            x_sto_m = w['X_sto', m]
            x_sto_mp1 = w['X_sto', m + 1]

            # obtain the average values of heatpump power, ambient temperature, heat demand and Qdot_hp
            u_vals = w['U', n:n + ratio, 0]  # heat pump power variables
            p_data_vals = self.data.getDataAtTimes(t_m, t_mp1, h)  # ambient temperature and heat demand
            assert u_vals.__len__() == p_data_vals[0].__len__()  # sanity check
            Tamb_aver = np.mean(p_data_vals[0])
            Qdot_hh_aver = np.mean(p_data_vals[4])
            Qdot_hp_aver = 1 / ratio * sum(
                [averagesystem.compute_Qdot_hp(P_hp, T_amb) for P_hp, T_amb in zip(u_vals, p_data_vals[0])])

            # implicit euler step of the average dynamics
            averageDynamicsEval = averagesystem.f_sto_average(x_sto_mp1, Tamb_aver, Qdot_hh_aver, Qdot_hp_aver, p_fix)
            G.append(x_sto_mp1 - x_sto_m - H_sec * averageDynamicsEval)
            lbG.append(ca.DM.zeros(averagesystem.x_sto.shape))
            ubG.append(ca.DM.zeros(averagesystem.x_sto.shape))

            # constraint satisfaction for coarse grid
            G.append(averagesystem.g_coarse(x_sto_m, Qdot_hp_aver))
            lbG.append(averagesystem.lbg_coarse)
            ubG.append(averagesystem.ubg_coarse)

        # store the nlp variables
        self.w = w
        self.lbw = lbw
        self.ubw = ubw
        self.J = (J_fix + J_running) / 1e6 + J_reg
        self.G = ca.vertcat(*G)
        self.lbG = ca.vertcat(*lbG)
        self.ubG = ca.vertcat(*ubG)

        self.w0 = w0
        self.f_outputs = ca.Function('f_outputs', [w], [ca.horzcat(*outputs)])
        self.f_Jrunning = ca.Function('f_Jrunning', [w], [J_running])
        self.f_Jfix = ca.Function('f_Jfix', [w], [J_fix])
        self.f_Jreg = ca.Function('f_Jreg', [w], [J_reg])