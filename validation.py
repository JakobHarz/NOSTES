import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
import re
from systemmodels.PTES_validation import ThermalModel as PTES
from utility import Constants

constants = Constants()

# --- Load Data ---
df = pd.read_csv('data/Dronninglund_treated_data_and_flow_rates_2014.csv', index_col=[0], parse_dates=True)


# --- Step 1: Prepare Full-Year 10-Minute Data ---
print("Preparing data with a 10-minute frequency...")
full_year_index = pd.date_range(start='2014-01-01', end='2014-12-31 23:50', freq='10T', tz='CET')

# --- Main Parameters ---
T_amb_K_full = df['temp_dry'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill').fillna(method='bfill').values + 273.15
T_bc_full = np.full_like(T_amb_K_full, 273.15 + 10) # Constant deep ground temperature

# --- Prepare 3-Port Flow and Temperature Data from Notebook Output ---
rename_dict = {'SO.LA.TT.414': 'T_top', 'SO.LA.TT.415': 'T_mid', 'SO.LA.TT.416': 'T_bot'}
df.rename(columns=rename_dict, inplace=True, errors='ignore')

# Resample the key flow and temperature columns to 10 minutes
F_top_q_full = df['F_topq'].resample('10T').mean().reindex(full_year_index).fillna(0)
F_mid_q_full = df['F_midq'].resample('10T').mean().reindex(full_year_index).fillna(0)
F_bot_n_full = df['F_bot_n'].resample('10T').mean().reindex(full_year_index).fillna(0)
T_top_full = df['T_top'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill')
T_mid_full = df['T_mid'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill')
T_bot_full = df['T_bot'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill')

# Convert flow rates from kg/hr to kg/s and separate into charge/discharge
kg_per_hr_to_kg_per_s = 1 / 3600.0
mdot_ch_top_full = F_top_q_full.clip(lower=0).values * kg_per_hr_to_kg_per_s
mdot_dis_top_full = -F_top_q_full.clip(upper=0).values * kg_per_hr_to_kg_per_s
T_in_ch_top_full = T_top_full.values + 273.15
mdot_ch_mid_full = F_mid_q_full.clip(lower=0).values * kg_per_hr_to_kg_per_s
mdot_dis_mid_full = -F_mid_q_full.clip(upper=0).values * kg_per_hr_to_kg_per_s
T_in_ch_mid_full = T_mid_full.values + 273.15
mdot_ch_bot_full = F_bot_n_full.clip(lower=0).values * kg_per_hr_to_kg_per_s
mdot_dis_bot_full = -F_bot_n_full.clip(upper=0).values * kg_per_hr_to_kg_per_s
T_in_ch_bot_full = T_bot_full.values + 273.15

# Ground temperatures
Tg_10_full = df['Tg_10'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill').fillna(method='bfill')
Tg_15_full = df['Tg_15'].resample('10T').mean().reindex(full_year_index).fillna(method='ffill').fillna(method='bfill')


# --- Step 2: Select Measured Temperatures for Direct Comparison ---
print("Mapping model layers to closest physical sensors...")
num_storage_layers = 4

# Instantiate a temporary model to get the calculated layer heights
temp_model = PTES(nk=1, s_n=num_storage_layers, g_n=3, distance=2)
# Get layer boundaries (cumulative height from the top)
model_layer_boundaries_from_top = np.cumsum([0] + temp_model.mh)
# Calculate the center point of each model layer (depth from top)
model_layer_centers_from_top = (model_layer_boundaries_from_top[:-1] + model_layer_boundaries_from_top[1:]) / 2
# Convert to height from bottom for comparison with sensor names
model_layer_centers_from_bottom = 16.0 - model_layer_centers_from_top
model_layer_boundaries_from_bottom = 16.0 - model_layer_boundaries_from_top

# Get the available physical sensor heights
physical_sensor_heights = np.linspace(0.5, 16.0, 32)

df_meas_aggr_full = pd.DataFrame(index=full_year_index)
for i in range(num_storage_layers):
    # The boundaries are ordered from top to bottom, so we reverse for bottom-up height
    boundary_upper = model_layer_boundaries_from_bottom[i]
    boundary_lower = model_layer_boundaries_from_bottom[i+1]
    
    cols_in_layer = []
    for c in df.columns:
        # Use regex to safely parse sensor names like 'T_01.5'
        match = re.match(r'T_(\d+\.\d+)', c)
        if match:
            z_probe = float(match.group(1)) # Height of this sensor from the bottom
            # Check if the sensor's height falls within the current model layer's boundaries
            if boundary_lower <= z_probe < boundary_upper:
                cols_in_layer.append(c)
                
    print(f"Model Layer {i+1} (H: {boundary_lower:.2f}m - {boundary_upper:.2f}m) includes {len(cols_in_layer)} physical sensors.")
    
    # Average all sensors found within that layer
    df_meas_aggr_full[f'T_meas_layer_{i+1}'] = (
        df[cols_in_layer].mean(axis=1)
            .resample('10T').mean()
            .reindex(full_year_index)
            .fillna(method='ffill')
        )


# Fill any remaining NaNs at the very top or bottom
df_meas_aggr_full.fillna(method='ffill', inplace=True)
df_meas_aggr_full.fillna(method='bfill', inplace=True)

print("\nInterpolating temperatures for layers without sensors...")
# Interpolate across columns (axis=1) to fill gaps between layers
df_meas_aggr_full = df_meas_aggr_full.interpolate(axis=1, method='linear')

# --- Step 3: Slice Data for Simulation Period ---
start_date = '2014-06-04'
print(f"\nSlicing data to start simulation from: {start_date}")
df_meas_aggr = df_meas_aggr_full.loc[start_date:].copy()
start_index = df_meas_aggr_full.index.searchsorted(start_date)

params_to_slice = {
    'T_amb_K': T_amb_K_full, 'T_bc': T_bc_full,
    'mdot_ch_top': mdot_ch_top_full, 'T_in_ch_top': T_in_ch_top_full, 'mdot_dis_top': mdot_dis_top_full,
    'mdot_ch_mid': mdot_ch_mid_full, 'T_in_ch_mid': T_in_ch_mid_full, 'mdot_dis_mid': mdot_dis_mid_full,
    'mdot_ch_bot': mdot_ch_bot_full, 'T_in_ch_bot': T_in_ch_bot_full, 'mdot_dis_bot': mdot_dis_bot_full
}
sim_params = {key: value[start_index:] for key, value in params_to_slice.items()}


# --- Step 4: Configure and Run the Simulation ---
N = len(df_meas_aggr)
num_ground_layers = 4
print(f"New simulation period: {df_meas_aggr.index[0]} to {df_meas_aggr.index[-1]}")
print(f"Total simulation steps (N): {N}")

def run_validation_simulation(model_class, initial_states, sim_steps, **kwargs):
    model = model_class(nk=sim_steps, g_n=num_ground_layers, s_n=num_storage_layers, distance=5)
    x, u, par, f = model.x, model.u, model.par, model.f
    x0 = ca.DM(initial_states)
    
    FE = ca.Function('x1', [x, u, par], [x + model.h * f(x, u, par)])
    
    con = np.zeros((u.shape[0], sim_steps))
    par_input = np.vstack([
        kwargs['T_amb_K'], kwargs['T_bc'],
        kwargs['mdot_ch_top'], kwargs['T_in_ch_top'], kwargs['mdot_dis_top'],
        kwargs['mdot_ch_mid'], kwargs['T_in_ch_mid'], kwargs['mdot_dis_mid'],
        kwargs['mdot_ch_bot'], kwargs['T_in_ch_bot'], kwargs['mdot_dis_bot']
    ])
    
    Xsim = FE.mapaccum(sim_steps)(x0, con, par_input)
    return Xsim

# Set Initial Conditions
initial_T_storage = df_meas_aggr.iloc[0].values + 273.15
initial_Tg_10 = Tg_10_full.loc[start_date].iloc[0] + 273.15 + 50
initial_Tg_15 = Tg_15_full.loc[start_date].iloc[0] + 273.15 + 30
initial_T_ground = [initial_Tg_10] + [initial_Tg_15] * (num_ground_layers-1)
x_initial = list(initial_T_storage) + list(initial_T_ground)

print(f"\nTotal initial states provided: {len(x_initial)}")
print(f"Initial Storage Temps (°C): {[f'{T-273.15:.1f}' for T in initial_T_storage]}")
print(f"Initial Ground Temps (°C): {initial_Tg_10-273.15:.1f}, {initial_Tg_15-273.15:.1f}")

# Run the Simulation
print("\nRunning simulation...")
X_sim_results = run_validation_simulation(PTES, x_initial, N, **sim_params)
print("Simulation successful.")


# --- Step 5: Compare and Visualize Results ---
results_index = df_meas_aggr.index
sim_results_df = pd.DataFrame(
    X_sim_results.full().T,
    index=results_index,
    columns=[f'T_sim_layer_{i+1}' for i in range(num_storage_layers)] + [f'T_ground_{i+1}' for i in range(num_ground_layers)]
)
T_sim_C = sim_results_df[[f'T_sim_layer_{i+1}' for i in range(num_storage_layers)]] - 273.15

plt.figure(figsize=(14, 8))
# Use a colormap for plotting many lines
colors = plt.cm.viridis(np.linspace(0, 1, num_storage_layers))
for i in range(num_storage_layers):
    layer_num = i + 1
    plt.plot(df_meas_aggr.index, df_meas_aggr[f'T_meas_layer_{layer_num}'], linestyle='--', color=colors[i])
    plt.plot(T_sim_C.index, T_sim_C[f'T_sim_layer_{layer_num}'], linestyle='-', color=colors[i], label=f'Layer {layer_num}')

plt.xlabel('Date')
plt.ylabel('Temperature [°C]')
plt.title(f'Model Validation: Simulation vs. Measurement ({num_storage_layers}-Layer Model)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

for i in range(num_storage_layers):
    layer_num = i + 1
    T_meas = df_meas_aggr[f'T_meas_layer_{layer_num}'].values
    T_sim = T_sim_C[f'T_sim_layer_{layer_num}'].values
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((T_sim - T_meas)**2))
    
    # Calculate NRMSE (normalized by the range of the measured data)
    meas_range = np.max(T_meas) - np.min(T_meas)
    if meas_range == 0:
        nrmse = np.inf # Avoid division by zero if data is constant
    else:
        nrmse = rmse / meas_range
        
    print(f"Layer {layer_num}: RMSE = {rmse:.2f} °C,  NRMSE = {nrmse:.2%}")