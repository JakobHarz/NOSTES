from typing import Union, List

import numpy as np

from systemmodels.systemModel import SystemModel, Data
from utility import Constants, NumpyStruct, Results
import casadi as ca
from casadi.tools import struct_symSX, entry

class STESNLP:
    def __init__(self, system: SystemModel, data: Data, N=365 * 24):

        self.system = system
        self.N = N
        self.data = data

        self.build_NLP()

    def build_NLP(self):
        # N = 365 * 24
        T = 365 * 24
        h = T / self.N  # [h] step size in hours
        h_sec = h * 3600
        N = self.N  # for readability
        self.h = h
        self.T = T
        self.h_sec = h_sec

        self.params = Constants()  # Initialize SystemParameters

        # empty list for constraint, (could be replaced with casadi structure at some point)
        G = []
        lbG = []
        ubG = []

        # Define the decision variables
        w = struct_symSX([entry('X', shape=self.system.x.shape, repeat=N + 1),  # MS shooting Nodes
                          entry('U', shape=self.system.u.shape, repeat=N),  # constant control over the interval
                          entry('p_fix', shape=self.system.p_fix.shape), ])  # constant parameter

        # Load every Data over the time horizon
        p_data_n = ca.horzcat(*[ca.vertcat(*self.data.getDataAtTime(n * h)) for n in range(N)])

        # initialization of the variables
        w0 = w(0)
        w0['X'] = self.system.x0  # initial temperature of storage
        w0['U', :, 0] = 3e6
        w0['U', :, 1] = 0
        w0['U', :, 2] = 0
        w0['p_fix'] = [1, 1, 1, 1, 20]
        w0['U', :, 0] = ca.horzsplit((p_data_n[1, :] + p_data_n[2, :] - p_data_n[3, :]) / 3)  # P_pv + P_wind - P_load
        w0['U', :, 1] = ca.horzsplit((p_data_n[1, :] + p_data_n[2, :] - p_data_n[3, :]) / 3)
        w0['U', :, 2] = ca.horzsplit((p_data_n[3, :] - p_data_n[1, :] - p_data_n[2, :]) / 3)  # P_load - P_pv - P_wind

        # create the upper and lower bounds for the variables
        lbw = w(-ca.inf)
        ubw = w(ca.inf)

        # overwrite upper and lower bounds for the states and controls
        lbw['X'] = self.system.lbx
        ubw['X'] = self.system.ubx
        lbw['U'] = self.system.lbu
        ubw['U'] = self.system.ubu
        lbw['p_fix'] = self.system.lbp
        ubw['p_fix'] = self.system.ubp

        # periodicty constraints
        G.append(w['X', 0] - w['X', self.N])
        lbG.append(ca.DM.zeros(self.system.x.shape))
        ubG.append(ca.DM.zeros(self.system.x.shape))

        # fixed costs
        J_fix = self.system.J_fix(w['p_fix'])
        J_running = 0
        # J_reg = (w['p_fix'] - w0['p_fix']).T@(w['p_fix'] - w0['p_fix'])*1E-1
        J_reg = 0

        # iterate the intervals
        outputs = []
        p_fix = w['p_fix']
        for n in range(N):
            # implicit euler integration
            x_n = w['X', n]
            x_np1 = w['X', n + 1]
            u_n = w['U', n]  # constant control over the interval
            t_n = h * n
            p_data_n = ca.vertcat(*self.system.data.getDataAtTime(t_n))

            # constraint satisfaction
            G.append(self.system.g(x_n, u_n, p_fix, p_data_n))
            lbG.append(self.system.lbg)
            ubG.append(self.system.ubg)

            # implicit euler step
            G.append(x_np1 - x_n - h_sec * self.system.f(x_np1, u_n, p_fix, p_data_n))
            lbG.append(ca.DM.zeros(self.system.x.shape))
            ubG.append(ca.DM.zeros(self.system.x.shape))

            # cost of the interval
            J_running += self.system.J_running(u_n) * h

            # regularization
            W_reg = ca.diag([1E-7, 1E-7, 1E-7, 1E-7, 1E-7])
            J_reg += 1E-8 * u_n.T @ W_reg.T @ W_reg @ u_n * h

            # store the outputs of the interval
            outputs.append(self.system.outputFunction(x_n, u_n, p_fix, p_data_n))

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

    def solve(self, additional_options: dict = {}) -> Results:

        print('Building NLP')
        # create the nlp
        nlp = {'x': self.w.cat,
               'f': self.J,
               'g': self.G}

        # create the solver
        opts = {'ipopt.max_iter': 0, 'ipopt.print_level': 5, 'print_time': 0, 'ipopt.linear_solver': 'ma97'}
        opts.update(additional_options)

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        print('Solving NLP')
        # solve the nlp
        res = solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbG, ubg=self.ubG)
        wopt = self.w(res['x'])

        # process the results into a single dictionary
        results = self.processOutput(wopt)

        return results

    def processOutput(self, wopt) -> Results:
        """ Processes the solution of the NLP into a dictionary.
        The keys in the dictionary are the defined outputs of the system.
        """

        # get the optimal outputs
        rawOutputs = NumpyStruct(self.system.outputStruct.repeated(self.f_outputs(wopt)))

        # empty results dictionary
        results = Results()

        # nlp cost
        results.addResult('NLP_J_running', float(self.f_Jrunning(wopt)), '-', 'Running Costs')
        results.addResult('NLP_J_fix', float(self.f_Jfix(wopt)), '-', 'Investment Costs')
        results.addResult('NLP_J_reg', float(self.f_Jreg(wopt)), '-', 'Regularization Costs')



        # time grid
        results['timegrid'] = np.linspace(0, self.T - self.h, self.N)

        # get the other outputs
        for key, subdict in self.system.outputs.items():
            values = rawOutputs[:, key]

            # check if all the values in the first dimension are the same
            if subdict['type'] == 'single':
                write_val = values[0]
            else:
                write_val = values

            results.addResult(key, write_val, subdict.get('unit','-'), subdict.get('description','-'))

        # information from the system
        results.addResult('nx', self.system.nx, '-', 'Number of states')
        results.addResult('nu', self.system.nu, '-', 'Number of controls')
        results.addResult('lbx', self.system.lbx.full(), '-', 'Lower Bounds for States')
        results.addResult('ubx', self.system.ubx.full(), '-', 'Upper Bounds for States')
        results.addResult('ubu', self.system.lbu.full(), '-', 'Lower Bounds for Controls')
        results.addResult('lbu', self.system.ubu.full(), '-', 'Upper Bounds for Controls')
        results.addResult('statenames', self.system.stateNames, '-', 'Names of the states')
        results.addResult('controlnames', self.system.stateNames, '-', 'Names of the controls')

        return results
