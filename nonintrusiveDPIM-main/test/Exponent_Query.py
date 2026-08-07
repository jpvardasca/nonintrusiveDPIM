import numpy as np
import pandas as pd
#from Multiindex_exponents import MultiExponentGenerator

class ExponentQuery:
    def __init__(self, exponent_table, order_table, exponent_dict, num_modes, num_forcing):
        self.exponent_table = exponent_table
        self.order_table = order_table
        self.exponent_dict = exponent_dict
        self.num_modes = num_modes
        self.num_forcing = num_forcing

    def find_combination(self, target):
        target_tuple = tuple(target)
        if target_tuple not in self.exponent_dict:
            return None
        idx = self.exponent_dict[target_tuple]
        return self.exponent_table.iloc[[idx]]

    def find_less_combinations(self, target):
        target_order = sum(target)
        lower_indices = self._get_lower_order_indices(target_order)
        
        if lower_indices.empty:
            return pd.DataFrame(columns=self.exponent_table.columns)
            
        first, last = self._get_index_bounds(lower_indices)
        candidates = self._get_candidates(first, last)
        return self._filter_candidates(candidates, np.array(target))

    def find_less_combinations_with_order1(self, target):
        target_tuple = tuple(target)
        #print(f"target tuple: {target_tuple}")
        if target_tuple not in self.exponent_dict:
            return pd.DataFrame(columns=self.exponent_table.columns)
        
        # Get global index of target gamma
        idx = self.exponent_dict[target_tuple]
        gamma_global_index = self.exponent_table.iloc[idx]['global_index']
        #print(f"gamma_global_index: {gamma_global_index}")
        
        # Get order=1 exponents
        order1_row = self.order_table[self.order_table['order'] == 1]
        if order1_row.empty:
            return pd.DataFrame(columns=self.exponent_table.columns)
            
        first_index = order1_row['first_index'].values[0]
        #print(first_index)
        last_index = order1_row['last_index'].values[0]
        #print(last_index)
        # Filter order=1 exponents with higher global index than gamma
        candidates = self.exponent_table[
            (self.exponent_table['global_index'] >= first_index) &
            (self.exponent_table['global_index'] <= last_index) &
            (self.exponent_table['global_index'] > gamma_global_index)
        ]
        #print(candidates)
        # Filter to only include unit vectors in forcing terms
        mask = []
        for exp_tuple in candidates['exponents']:
            # Split into mode and forcing parts
            modes = exp_tuple[:self.num_modes]
            forcings = exp_tuple[self.num_modes:self.num_modes+self.num_forcing]
            
            # Check mode part is all zeros and forcing has exactly one 1
            if all(m == 0 for m in modes) and sum(forcings) == 1:
                mask.append(True)
            else:
                mask.append(False)
                
        return candidates[mask].copy()

    def _get_lower_order_indices(self, target_order):
        return self.order_table[
            (self.order_table['order'] >= 2) & 
            (self.order_table['order'] < target_order)
        ]

    def _get_index_bounds(self, indices_df):
        return (
            indices_df['first_index'].min(),
            indices_df['last_index'].max()
        )

    def _get_candidates(self, first_idx, last_idx):
        return self.exponent_table[
            (self.exponent_table['global_index'] >= first_idx) &
            (self.exponent_table['global_index'] <= last_idx)
        ]

    def _filter_candidates(self, candidates, target_arr):
        mask = np.all(np.array(list(candidates['exponents'])) <= target_arr, axis=1)
        return candidates[mask].copy()

