from typing import List, Union
import numpy as np
import pandas as pd
from tabulate import tabulate


class Constants:
    rho = 1000  # [kg/m^3] Density of water
    rho_g = 2000  # [kg/m^3] Density of ground(dry sand)
    c_p = 4200  # [J/kgK] Specific heat capacity of water @ 50degC
    c_pg = 700  # ground (dry sand) specific heat capacity J/kgK

    # ----- STORAGE -----
    U_wall_ins = 0.3  # [W/m^2K] Heat transfer coefficient
    U_wall_noins = 80  # [W/m^2K] Heat transfer coefficient
    height = 15  # [m] Height of the storage
    lambda_ground = 0.47  # W/mK
    radius = 63.1  # [m] scaled radius of the storage
    U_top = 0.186  # [W/m^2K] Heat transfer coefficient of the top
    T_ret_hh = 273.15 + 20
    T_sup_hh = 273.15 + 40
    lambda_eff = 0.644  # [W/mK] Effective thermal conductivity of the storage
    eta_hp = 0.5
    min_T = 273.15 + 10
    min_T1 = 273.15 + 40
    max_T = 273.15 + 85
    T_bc = 273.15 + 13.5

    # ----- HEAT PUMP -----
    T_hp = 86 + 273.15  # [K] Temperature of the heat pump
    mdot_hp_max = 700  # [kg/s] Maximum mass flow rate of the heat pump

    # ---- PRICING ----
    price_sell = -0.01  # EUR/kWh
    #price_sell = 0
    price_buy = 0.3  # EUR/kWh
    DOLLAR_TO_EURO = 0.92
    I_hp = 0.375  # EUR/W
    I_s = 30  # EUR/m^3
    I_pv = 1.491 * DOLLAR_TO_EURO  # EUR/Wp
    I_wind = 1.569 * DOLLAR_TO_EURO  # EUR/Wp
    I_bat = 0.476 * DOLLAR_TO_EURO  # EUR/Wh

    # --- stuff ---
    n_years = 30  # Number of years the system is running
    annuity = 0.04  # Annuity rate

    # ---- Default Scaling of size params, HAS TO BE SET BY THE USER ----
    C_bat_default = None  # Wh
    C_hp_default = None  # W (thermal)
    C_wind_default = None  # Wp
    C_pv_default = None  # Wp (Use instead of 20% rather 18% efficiency)

    # --- Battery Parameters ---
    eta_ch = 0.95 # Charging efficiency
    eta_dis = 0.95  # Discharging efficiency


    def strat_storage(self):
        # Store the params for different layers of the storage
        strat_params = {
            2: {
                'tl_init': 153.3,
                'bl_init': 73.2,
                'ml_init': [125.94],
                'mh': [5.12]
            },
            3: {
                'tl_init': 153.3,
                'bl_init': 73.2,
                'ml_init': [136.31, 113.51],
                'mh': [3.18, 4.27]
            },
            4: {
                'tl_init': 153.3,
                'bl_init': 73.2,
                'ml_init': [140.95, 125.94, 106.11],
                'mh': [2.31, 2.81, 3.71]
            }
        }
        return strat_params


class Results:
    """ A class to store the results of the optimization problem,
     including the optimal trajectories, the optimal parameters ...
    Also includes stats from the NLP solver, sizing, cost function values etc.

    Can be saved and loaded from a file using .save() and .fromFile().

    Use results.printAll() to print an overview in the console.

    """

    # Expose some results, for easier access, not sure if this is useful
    X: np.ndarray = np.nan
    timegrid: np.ndarray = np.nan
    U: np.ndarray = np.nan
    J_running: np.ndarray = np.nan
    J_fix: np.ndarray  = np.nan
    V_s: np.ndarray = np.nan
    C_bat: np.ndarray = np.nan
    C_hp: np.ndarray = np.nan
    C_pv: np.ndarray = np.nan
    C_wind: np.ndarray = np.nan

    def __init__(self):
        self._valueDict = {}
        self._unitDict = {}
        self._descriptionDict = {}
        self.keys: List[str] = []

        # add some empty results
        self.addResult('X', np.nan, 'K', 'State Variables')
        self.addResult('timegrid', np.nan, 'h', 'Time Grid')
        self.addResult('U', np.nan, 'W', 'Control Variables')
        self.addResult('J_running', np.nan, 'EUR', 'Running Costs')
        self.addResult('J_fix', np.nan, 'EUR', 'Investment Costs')

        # do for V_s, C_bat, C_hp, C_pv, C_wind
        self.addResult('V_s', np.nan, 'm^3', 'Size of the thermal storage')
        self.addResult('C_bat', np.nan, 'Wh', 'Capacity of the battery')
        self.addResult('C_hp', np.nan, 'W', 'Capacity of the heat pump')
        self.addResult('C_pv', np.nan, 'Wp', 'Capacity of the PV')
        self.addResult('C_wind', np.nan, 'Wp', 'Capacity of the wind turbine')

    @property
    def units(self) -> dict:
        return self._unitDict

    def addResult(self, name: str, value, unit: Union[str, List[str]] = None, description: str = None):
        """ Adds a result to the results dictionary, with the given name, value, unit and description."""
        if type(value) is not np.ndarray:
            value = np.array(value)
        self._valueDict[name] = value

        if unit is not None:
            self._unitDict[name] = unit
        else:
            # dont overwrite the unit if it is already set
            self._unitDict[name] = self._unitDict.get(name, '-')

        if description is not None:
            self._descriptionDict[name] = description
        else:
            # dont overwrite the description if it is already set
            self._descriptionDict[name] = self._descriptionDict.get(name, '-')

        if name not in self.keys:
            self.keys.append(name)

        # check if the name is also an attribute of the class, if yes, set
        if hasattr(self, name):
            setattr(self, name, value)

    def save(self, filename):
        # dumps the object into a npz file
        outdict = {}
        for name in self.keys:
            outdict[name + '_value'] = self._valueDict[name]
            outdict[name + '_unit'] = self._unitDict[name]
            outdict[name + '_description'] = self._descriptionDict[name]

        # store
        # make sure that the filename ends with .npz
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename, **outdict, names=self.keys)

    @staticmethod
    def fromFile(filename):
        # loads the object from a npz file
        inDict = np.load(filename, allow_pickle=False)
        res = Results()
        for name in inDict['names']:
            res.addResult(name, inDict[name + '_value'], str(inDict[name + '_unit']),
                          str(inDict[name + '_description']))
        return res

    def printAll(self, comparewith: 'Results' = None):
        self.printValues(self.keys, comparewith=comparewith)

    def printValues(self, keys: List[str], comparewith: 'Results' = None):
        """ Print a formatted table with the values of the given keys.
         A second 'Results' instance can be provided to compare the values side by side."""
        table_data = []
        for key in keys:
            # Truncate entries that exceed max_width
            description = self._descriptionDict[key]
            unit = self._unitDict[key]
            value_array: np.ndarray = self._valueDict[key]
            value_str = self._formatValue(value_array)
            row_data = [key, description, unit, value_str]

            if comparewith is not None:
                if key in comparewith.keys:
                    value_str_compar = self._formatValue(comparewith[key])
                else:
                    value_str_compar = '-'
                row_data.append(value_str_compar)

            table_data.append(row_data)

        # Print the formatted table
        headers = ['Key', 'Description', 'Unit', 'Value']
        colaling = ['left', 'left', 'left', 'right']
        if comparewith is not None:
            headers.append('Value (Comp)')
            colaling.append('right')
        print(tabulate(table_data, headers=headers, tablefmt='rst', maxcolwidths=50, colalign=colaling))

    def printSizings(self, comparewith: 'Results' = None):
        self.printValues(['V_s', 'C_bat', 'C_hp', 'C_pv', 'C_wind'], comparewith=comparewith)

    def printNLPStats(self, comparewith: 'Results' = None):
        self.printValues(['nlp_w_size', 'solver_iter_count', 'solver_t_wall_total'], comparewith=comparewith)

    def printCosts(self, comparewith: 'Results' = None):
        self.printValues([key for key in self.keys if key.startswith('cost_')], comparewith=comparewith)

    def __getitem__(self, item):
        return self._valueDict[item]

    def __setitem__(self, key, value):
        self.addResult(key, value)

    def __repr__(self):
        return f"Results Instance, use results.printAll() to print the results."

    def _formatValue(self, value: np.ndarray):
        assert type(value) is np.ndarray, "The value should be a numpy array"
        if value.size > 8:
            return "NumpyArray " + str(value.shape)
        elif value.size == 1 and np.issubdtype(value.dtype, np.number):
            # format with engineering notation (1E3, 1E6, 1E9, ...)
            return self._formatEngineering(value)
        else:
            value_str = str(value).replace("\n", "")
            # cut away after 20 letters
            if len(value_str) > 23:
                value_str = value_str[:20] + '...'
            return value_str

    def _formatEngineering(self, value):
        """ formats the given value in engineering notation"""

        exponent = np.floor(np.log10(np.abs(value))) // 3 * 3
        significand = value / 10 ** exponent

        # look up the exponent letter in a dictionary from -12 to 12
        exponent_dict = {3: 'k', 6: 'M', 9: 'G', 12: 'T', 0: '', -3: 'm', -6: 'u', -9: 'n', -12: 'p'}
        exponent_letter = exponent_dict.get(int(exponent), f'E{exponent}')

        return f"{significand:.2f} {exponent_letter}"


class Data:
    """ Class to preprocess and store the data for the NLP.
        The provided data file should have the following columns:
        - T_amb: ambient temperature [K (!)]
        - P_pv: power of the pv [W]
        - P_wind: power of the wind [W]
        - P_load: electric household power demand [W]
        - Qdot_load: heat demand of the household [W]

        p_data (5): - T_amb ambient temperature
                    - P_pv power of the pv
                    - P_wind power of the wind
                    - P_load electric household power demand
                    - Qdot_load heat demand of the household
    """

    def __init__(self, file_path: str):
        # Load the data from the file
        raw_data = pd.read_csv(file_path)
        assert all([col in raw_data.columns for col in ['T_amb', 'P_pv', 'P_wind', 'P_load', 'Qdot_load']]),\
            "The data file should have the columns: T_amb, P_pv, P_wind, P_load, Qdot_load"

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
