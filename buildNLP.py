import numpy as np

from systemmodels.systemModel import SystemModel, Data
from utility import Constants, NumpyStruct
import casadi as ca
from casadi.tools import struct_symSX, entry

class STESNLP:
    def __init__(self, system: SystemModel, data: Data, N = 365*24):

        self.system = system

        # N = 365 * 24
        T = 365 * 24
        h = T / N  # [h] step size in hours
        h_sec = h * 3600

        self.N = N
        self.h = h
        self.T = T
        self.h_sec = h_sec

        self.params = Constants()  # Initialize SystemParameters

        # empty list for constraint, (could be replaced with casadi structure at some point)
        G = []
        lbG = []
        ubG = []

        # Define the decision variables
        w = struct_symSX([entry('X', shape=system.x.shape, repeat=N + 1),  # MS shooting Nodes
                          entry('U', shape=system.u.shape, repeat=N),  # constant control over the interval
                          entry('p_fix', shape=system.p_fix.shape), ])  # constant parameter


        # Load every Data over the time horizon
        p_data_n = ca.horzcat(*[ca.vertcat(*data.getDataAtTime(n*h)) for n in range(N)])

        # initialization of the variables
        w0 = w(0)
        w0['X'] = self.system.x0  # initial temperature of storage
        w0['U', :, 0] = 3e6
        w0['U', :, 1] = 0
        w0['U', :, 2] = 0
        w0['p_fix'] = [1,1,1,1,20]
        w0['U', :, 0] = ca.horzsplit((p_data_n[1,:] + p_data_n[2,:] - p_data_n[3,:])/3)  # P_pv + P_wind - P_load
        w0['U', :, 1] = ca.horzsplit((p_data_n[1,:] + p_data_n[2,:] - p_data_n[3,:])/3)
        w0['U', :, 2] = ca.horzsplit((p_data_n[3,:] - p_data_n[1,:] - p_data_n[2,:])/3) # P_load - P_pv - P_wind

        # create the upper and lower bounds for the variables
        lbw = w(-ca.inf)
        ubw = w(ca.inf)

        # overwrite upper and lower bounds for the states and controls
        lbw['X'] = system.lbx
        ubw['X'] = system.ubx
        lbw['U'] = system.lbu
        ubw['U'] = system.ubu
        lbw['p_fix'] = system.lbp
        ubw['p_fix'] = system.ubp

        # periodicty constraints
        G.append(w['X', 0] - w['X', N])
        lbG.append(ca.DM.zeros(system.x.shape))
        ubG.append(ca.DM.zeros(system.x.shape))

        # fixed costs
        J_fix = system.J_fix(w['p_fix'])
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
            p_data_n = ca.vertcat(*system.data.getDataAtTime(t_n))

            # constraint satisfaction
            G.append(system.g(x_n, u_n, p_fix, p_data_n))
            lbG.append(system.lbg)
            ubG.append(system.ubg)

            # implicit euler step
            G.append(x_np1 - x_n - h_sec * system.f(x_np1, u_n, p_fix, p_data_n))
            lbG.append(ca.DM.zeros(system.x.shape))
            ubG.append(ca.DM.zeros(system.x.shape))

            # cost of the interval
            J_running += system.J_running(u_n)*h

            # regularization
            W_reg = ca.diag([1E-7,1E-7,1E-7,1E-7,1E-7])
            J_reg += 1E-8*u_n.T@W_reg.T@W_reg@u_n*h

            # store the outputs of the interval
            outputs.append(system.outputFunction(x_n, u_n, p_fix, p_data_n))

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

    def solve(self, additional_options: dict = {}):

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


    def processOutput(self, wopt) -> dict:
        """ Processes the solution of the NLP into a dictionary.
        The keys in the dictionary are the defined outputs of the system.
        """


        # get the optimal outputs
        rawOutputs = NumpyStruct(self.system.outputStruct.repeated(self.f_outputs(wopt)))

        # empty results dictionary
        results = {}
        results['rawOutputs'] = rawOutputs

        # nlp cost
        results['NLP_costs'] = {'J_running': float(self.f_Jrunning(wopt)),
                                'J_fix': float(self.f_Jfix(wopt)),
                                'J_reg': float(self.f_Jreg(wopt))}

        # time grid
        results['time_grid'] = np.linspace(0, self.T-self.h, self.N)

        # get the other outputs
        for key, subdict in self.system.outputs.items():
            values = rawOutputs[:, key]

            # check if all the values in the first dimension are the same
            if subdict['type'] == 'single':
                results[key] = values[0]
            else:
                results[key] = values

        return results
