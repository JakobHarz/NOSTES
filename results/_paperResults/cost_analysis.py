
import numpy as np

class CostAnalyzer:
    """
    Handles all post-processing financial calculations.
    Derives all costs from raw physical results (capacities and power profiles).
    """
    def __init__(self, n_years: int, discount_rate: float, investment_costs: dict, opex_percents: dict):
        self.n = n_years
        self.r = discount_rate
        self.I = investment_costs
        self.O_p = opex_percents
        self.pv_annuity_factor = (1 - (1 + self.r)**-self.n) / self.r

    def calculate_all_costs(self, capacities: dict, p_grid_profile: np.ndarray, grid_prices: dict) -> dict:
        """
        Calculates the full, granular cost breakdown for a given set of results.

        Args:
            capacities: Dictionary of the optimized physical capacities (e.g., {'hp': 20e6}).
            p_grid_profile: The time-series array of grid power in Watts.
            grid_prices: Dictionary with {'buy': price, 'sell': price}.
        """
        # --- 1. Calculate CAPEX for each component ---
        capex = {tech: self.I[tech] * cap for tech, cap in capacities.items()}

        # --- 2. Calculate Annual and Lifetime Discounted OPEX ---
        lifetime_opex = {}
        for tech in capacities.keys():
            annual_opex = capex[tech] * self.O_p.get(tech, 0)
            lifetime_opex[tech] = annual_opex * self.pv_annuity_factor

        # --- 3. Calculate Annual and Lifetime Grid Costs from the Power Profile ---
        # Assuming 1-hour timesteps, so Power (W) equals Energy (Wh)
        energy_imported_kWh = p_grid_profile[p_grid_profile > 0].sum() / 1000
        energy_exported_kWh = -p_grid_profile[p_grid_profile < 0].sum() / 1000

        annual_import_cost = energy_imported_kWh * grid_prices['buy']
        annual_export_revenue = energy_exported_kWh * grid_prices['sell'] # price_sell is negative

        lifetime_import_pv = annual_import_cost * self.pv_annuity_factor
        lifetime_export_pv = annual_export_revenue * self.pv_annuity_factor

        # --- 4. Assemble the final results dictionary ---
        results = {}
        total_npv = 0
        
        for tech in capacities.keys():
            results[f'CAPEX_{tech}'] = capex[tech]
            results[f'OPEX_lifetime_pv_{tech}'] = lifetime_opex[tech]
            npv_component = capex[tech] + lifetime_opex[tech]
            results[f'NPV_{tech}'] = npv_component
            total_npv += npv_component

        results['GRID_IMPORT_lifetime_pv'] = lifetime_import_pv
        results['GRID_EXPORT_lifetime_pv'] = lifetime_export_pv
        total_npv += lifetime_import_pv + lifetime_export_pv
        
        results['TOTAL_NPV'] = total_npv
        
        return results