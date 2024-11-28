import numpy as np


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
    price_sell = -0.05  # EUR/kWh
    price_buy = 0.4  # EUR/kWh

    # --- stuff ---
    n_years = 20  # Number of years the system is running

    def battery_params(self):
        eta_ch = 0.95
        eta_dis = 0.95  # Discharging efficiency
        return eta_ch, eta_dis

    def investment(self):
        # Dollar to Euro 
        Dollar_to_Euro = 0.9

        I_hp = 0.375  # EUR/W
        I_s = 30  # EUR/m^3
        I_pv = 1.491 * 0.9  # EUR/Wp
        I_wind = 1.569 * Dollar_to_Euro  # EUR/Wp
        I_bat = 0.476 * Dollar_to_Euro  # EUR/Wh
        # OPEX_pv = 21 * 0.9 / 1e3 
        # OPEX_wind = 31 * 0.9 / 1e3
        # OPEX_battery = 39 * 0.9 /1e3 
        return I_hp, I_s, I_pv, I_wind, I_bat

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


class NumpyStruct:
    """
    Helper class to convert the values of a DM struct to numpy arrays, especially for plotting.
    Indexing the instance will index the DM struct and return a numpy array of the result, with proper shapes.

    Example:

    >>> Xsim = model.x_struct.repeated(Xsim)
    >>> Xsim_plotting = NumpyStruct(Xsim)
    >>> plt.plot(Xsim_plotting[:, 't'], Xsim_plotting[:, 'x', 0])

    """

    def __init__(self, DMStruct):
        self.DMStruct = DMStruct

    def __getitem__(self, slice) -> np.ndarray:
        slice_result = self.DMStruct[slice]
        # convert to numpy array and squeeze
        return np.array(slice_result).squeeze()
