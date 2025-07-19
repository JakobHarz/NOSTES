# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from utility import Results
from cost_analysis import CostAnalyzer # ## ADDED: Import your new analyzer

# --- Constants ---
# Define constants at the top for easy configuration and clarity.
NUM_SCENARIOS = 6
DISTRICT_AREA_M2 = 1.1e6
BASE_LOAD_WS = 32 * 1e9  # 32 GWh
ELECTRICITY_PRICES = np.linspace(0.1, 0.6, 6) # Corresponds to c_el_values

# Define file patterns in a dictionary for easy access.
FILE_PATTERNS = {
    'full_model': 'dietenbach_average_varyPrice_{i}.npz',
    'no_stes': 'dietenbach_nosto_varyPrice_{i}.npz',
    'no_wind': 'dietenbach_nowind_average_varyPrice_{i}.npz',
}

FINANCIAL_CONSTANTS = {
    'n_years': 30,
    'discount_rate': 0.04,
    'investment_costs': {
        'hp': 0.651,      # EUR/W
        's': 30,          # EUR/m^3
        'pv': 1.371,      # EUR/Wp (assuming 0.92 conversion)
        'wind': 1.443,    # EUR/Wp
        'bat': 0.438      # EUR/Wh
    },
    'opex_percents': {
        'hp': 0.025,
        's': 0.01,
        'pv': 0.01,
        'wind': 0.02,
        'bat': 0.02
    }
}

def setup_latex_plotting():
    """Configures matplotlib to use LaTeX for rendering text in plots."""
    params = {
        'text.usetex': True,
        'font.family': 'serif',
        "axes.labelsize": 14,
        "font.size": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        'text.latex.preamble': r'\usepackage{eurosym}'
    }
    plt.rcParams.update(params)


def load_and_process_data(file_pattern: str, num_files: int) -> Tuple[list, list, np.ndarray]:
    """
    Loads raw simulation results. Now returns:
    1. A list of full cost breakdown dictionaries.
    2. A list of capacity breakdown dictionaries.
    3. An array of autonomy levels.
    """
    cost_breakdowns_npv = []
    capacity_breakdowns = [] # ## ADDED: New list to store capacities
    autonomy_levels = []
    total_costs_annual = []  # To store annual costs for each scenario
    
    analyzer = CostAnalyzer(
        n_years=FINANCIAL_CONSTANTS['n_years'],
        discount_rate=FINANCIAL_CONSTANTS['discount_rate'],
        investment_costs=FINANCIAL_CONSTANTS['investment_costs'],
        opex_percents=FINANCIAL_CONSTANTS['opex_percents']
    )

    for i in range(num_files):
        result = Results.fromFile(file_pattern.format(i=i))

        capacities = {
            'hp': np.nan_to_num(result.get('C_hp', 0)),
            's': np.nan_to_num(result.get('V_s', 0)),
            'pv': np.nan_to_num(result.get('C_pv', 0)),
            'wind': np.nan_to_num(result.get('C_wind', 0)),
            'bat': np.nan_to_num(result.get('C_bat', 0))
        }
        capacity_breakdowns.append(capacities) # ## ADDED: Append capacities to the list
        
        p_grid_profile = result.get('P_grid', np.array([0]))
        grid_prices = {'buy': ELECTRICITY_PRICES[i], 'sell': -0.01}
        
        cost_breakdown = analyzer.calculate_all_costs(capacities, p_grid_profile, grid_prices)
        cost_breakdowns_npv.append(cost_breakdown)

        # --- Calculate Simple Annual Cost (for the plot) ---
        # Get the pre-calculated annual costs from the result file
        j_fix = np.nan_to_num(result.get('J_fix', 0))
        j_running = np.nan_to_num(result.get('J_running', 0))
        
        # Note: If j_running is a profile, use .sum(). If it's already a total, .sum() is harmless.
        total_annual_cost = j_fix + j_running.sum()
        total_costs_annual.append(total_annual_cost)

        hp_consumption_wh = np.nan_to_num(result.get('P_hp', np.array([0]))).sum()
        grid_purchase_wh = p_grid_profile[p_grid_profile > 0].sum()
        total_demand_wh = BASE_LOAD_WS + hp_consumption_wh
        
        autonomy = (total_demand_wh - grid_purchase_wh) / total_demand_wh if total_demand_wh > 0 else 0
        autonomy_levels.append(autonomy)
        
    return cost_breakdowns_npv, capacity_breakdowns, np.array(total_costs_annual), np.array(autonomy_levels)


def plot_cost_comparison(costs_full, autonomy_full, costs_nosto, autonomy_nosto, costs_nowind, autonomy_nowind):
    # ... (this function is unchanged) ...
    plt.figure(figsize=(9, 5))
    plt.plot(ELECTRICITY_PRICES, costs_full / DISTRICT_AREA_M2, marker='o', color='C0', label='(i) Full Model')
    plt.plot(ELECTRICITY_PRICES, costs_nosto / DISTRICT_AREA_M2, marker='x', color='C1', label='(ii) No PTES')
    plt.plot(ELECTRICITY_PRICES, costs_nowind / DISTRICT_AREA_M2, marker='s', color='r', label='(iii) No Wind')
    plt.xlabel(r'Electricity Price (\euro/kWh)')
    plt.ylabel(r'Yearly Energy Cost (\euro/m²)')
    plt.ylim([4, 13])
    plt.xlim([0.05, 0.65])
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.axvline(x=0.3, color='g', linestyle='--', linewidth=1)
    
    def annotate_autonomy(prices, costs, autonomy_levels):
        for i, (price, cost) in enumerate(zip(prices, costs)):
            if i == 0: continue
            plt.text(price + 0.019, cost - 0.08, f'({autonomy_levels[i]:.2f})', fontsize=11, ha='center', va='top')
    
    annotate_autonomy(ELECTRICITY_PRICES, costs_full / DISTRICT_AREA_M2, autonomy_full)
    annotate_autonomy(ELECTRICITY_PRICES, costs_nosto / DISTRICT_AREA_M2, autonomy_nosto)
    annotate_autonomy(ELECTRICITY_PRICES, costs_nowind / DISTRICT_AREA_M2, autonomy_nowind)
    
    plt.tight_layout()
    plt.savefig('cost_comparison.pdf')
    plt.show()

    
def print_comparative_breakdown(full_data, nosto_data, nowind_data, target_price=0.3):
    """
    Creates a single, formatted table comparing the capacities and cost breakdowns
    of the three main scenarios at a specific target electricity price.
    """
    print(f"\n==================================================================================")
    print(f"        CAPACITY & COST BREAKDOWN AT {target_price:.2f} €/kWh")
    print(f"==================================================================================")

    try:
        idx = np.where(np.isclose(ELECTRICITY_PRICES, target_price))[0][0]
    except IndexError:
        print(f"Error: Target price {target_price} not found in ELECTRICITY_PRICES.")
        return

    # --- Unpack all the data for the target price ---
    costs_full, caps_full, autonomy_full = full_data[0][idx], full_data[1][idx], full_data[2][idx]
    costs_nosto, caps_nosto, autonomy_nosto = nosto_data[0][idx], nosto_data[1][idx], nosto_data[2][idx]
    costs_nowind, caps_nowind, autonomy_nowind = nowind_data[0][idx], nowind_data[1][idx], nowind_data[2][idx]
    
    # --- Print Header ---
    header = f"{'Component':<22} | {'(i) Full Model':>15} | {'(ii) No PTES':>15} | {'(iii) No Wind':>15}"
    print(header)
    print("----------------------------------------------------------------------------------")

    # --- ## ADDED: Print Capacities Section ## ---
    print(f"{'CAPACITIES':<22} | {'':>15} | {'':>15} | {'':>15}")
    capacity_units = {'hp': 'MW_th', 's': 'm³', 'pv': 'MWp', 'wind': 'MWp', 'bat': 'MWh'}
    techs = ['hp', 's', 'pv', 'wind', 'bat']

    for tech in techs:
        # Divide by 1e6 for MW/MWh, except for storage volume
        unit_divisor = 1 if tech == 's' else 1e6
        
        val_full = caps_full.get(tech, 0) / unit_divisor
        val_nosto = caps_nosto.get(tech, 0) / unit_divisor
        val_nowind = caps_nowind.get(tech, 0) / unit_divisor
        
        label = f"  {tech.upper()} ({capacity_units[tech]})"
        if any(v > 0 for v in [val_full, val_nosto, val_nowind]):
            print(f"{label:<22} | {val_full:15,.2f} | {val_nosto:15,.2f} | {val_nowind:15,.2f}")
    
    print("----------------------------------------------------------------------------------")
    print(f"{'LIFETIME COSTS (NPV)':<22} | {'':>15} | {'':>15} | {'':>15}")

    # --- Print Cost Rows (Unchanged) ---
    cost_keys = [
        ('hp', 'CAPEX'), ('s', 'CAPEX'), ('pv', 'CAPEX'), ('wind', 'CAPEX'), ('bat', 'CAPEX'),
        ('hp', 'OPEX'), ('s', 'OPEX'), ('pv', 'OPEX'), ('wind', 'OPEX'), ('bat', 'OPEX'),
        ('import', 'GRID'), ('export', 'GRID'),
        ('total', 'NPV')
    ]
    for component, type in cost_keys:
        # ... (rest of the cost printing logic is the same as before) ...
        if type == 'CAPEX':
            key, label = f"CAPEX_{component}", f"  CAPEX {component.upper()}"
        elif type == 'OPEX':
            key, label = f"OPEX_lifetime_pv_{component}", f"    OPEX PV {component.upper()}"
        elif type == 'GRID':
            key, label = f"GRID_{component.upper()}_lifetime_pv", f"  Grid {component.capitalize()} PV"
        else:
            print("----------------------------------------------------------------------------------")
            key, label = "TOTAL_NPV", "TOTAL NPV"
            
        val_full = costs_full.get(key, 0)
        val_nosto = costs_nosto.get(key, 0)
        val_nowind = costs_nowind.get(key, 0)

        if any(v != 0 for v in [val_full, val_nosto, val_nowind]):
            print(f"{label:<22} | {val_full:15,.0f} | {val_nosto:15,.0f} | {val_nowind:15,.0f}")
    
    print("----------------------------------------------------------------------------------")
    
    # --- Print Autonomy Levels (Unchanged) ---
    print(f"{'Autonomy Level (%)':<22} | {autonomy_full:>14.1%} | {autonomy_nosto:>14.1%} | {autonomy_nowind:>14.1%}")
    print("==================================================================================")

def main():
    """Main function to run data processing, print breakdowns, and then plot."""
    setup_latex_plotting()

    # --- Step 1: Data Loading - now returns three items ---
    costs_full_b, caps_full, costs_full_plot, auto_full = load_and_process_data(FILE_PATTERNS['full_model'], NUM_SCENARIOS)
    costs_nosto_b, caps_nosto, costs_nosto_plot, auto_nosto = load_and_process_data(FILE_PATTERNS['no_stes'], NUM_SCENARIOS)
    costs_nowind_b, caps_nowind, costs_nowind_plot, auto_nowind = load_and_process_data(FILE_PATTERNS['no_wind'], NUM_SCENARIOS)

    # --- Step 2: Print the Single, Comparative Breakdown Table ---
    print_comparative_breakdown(
        (costs_full_b, caps_full, auto_full),
        (costs_nosto_b, caps_nosto, auto_nosto),
        (costs_nowind_b, caps_nowind, auto_nowind)
    )

    plot_cost_comparison(
        costs_full_plot, auto_full,
        costs_nosto_plot, auto_nosto,
        costs_nowind_plot, auto_nowind
    )
    print(costs_full_plot, costs_nosto_plot, costs_nowind_plot)

if __name__ == '__main__':
    main()
# %%