import numpy as np
import pandas as pd
from Exponent_Query import ExponentQuery  # Ensure this import is used elsewhere

class MultiExponentGenerator:
    def __init__(self, num_variables, num_forcing, max_order, max_order_modes, maximum_order_forcing):
        """Initialize the multi-exponent generator with system parameters and constraints.
        
        Args:
            num_variables: Total number of variables (modes + forcing terms)
            num_forcing: Number of forcing variables
            max_order: Maximum total polynomial order
            max_order_modes: Maximum order for mode variables
            maximum_order_forcing: Maximum order for forcing variables
        """
        self.num_variables = num_variables
        self.num_forcing = num_forcing
        self.num_modes = num_variables - num_forcing
        # Validate both blocks can be split evenly
        if self.num_modes % 2 != 0 or num_forcing % 2 != 0:
            raise ValueError(
                "Both the number of modes (num_variables - num_forcing) and "
                "the number of forcing variables must be even"
            )
        self.max_order = max_order
        self.max_order_modes = max_order_modes
        self.maximum_order_forcing = maximum_order_forcing
        self.exponent_table = None
        self.order_table = None
        self.exponent_dict = None
        self.exponent_list = None

    def _generate_combinations(self, remaining_variables, remaining_order):
        """Recursively generate all exponent combinations for given variables and order.
        
        Args:
            remaining_variables: Number of variables left to assign exponents
            remaining_order: Total exponent sum remaining to distribute
        
        Returns:
            np.array: Array of exponent combinations (each row is a combination)
        """
        if remaining_variables == 1:
            return np.array([[remaining_order]])
        combinations = []
        for current_power in range(remaining_order, -1, -1):
            sub_combinations = self._generate_combinations(
                remaining_variables - 1, remaining_order - current_power
            )
            if sub_combinations.size:
                new_row = np.hstack([
                    np.full((len(sub_combinations), 1), current_power), 
                    sub_combinations
                ])
                combinations.append(new_row)
        return np.vstack(combinations) if combinations else np.array([])

    def build_tables(self):
        """Main method to build exponent and order tables with constraints.
        
        Generates all valid exponent combinations, applies constraints, and builds:
        - Exponent table with detailed metadata
        - Order table with summary statistics
        - Lookup structures for conjugates
        - Special columns for solver integration
        """
        all_combos = []
        counts = []
        firsts = []
        lasts = []
        next_index = 1  # Track next available global index
        
        for order in range(1, self.max_order + 1):
            current = self._generate_combinations(self.num_variables, order)
            if len(current) > 0:
                # Filter combinations based on order constraints
                modes_sum = current[:, :self.num_modes].sum(axis=1)
                forcing_sum = current[:, self.num_modes:self.num_modes+self.num_forcing].sum(axis=1)
                mask = (modes_sum <= self.max_order_modes) & (forcing_sum <= self.maximum_order_forcing)
                current = current[mask]
            
            count_current = len(current)
            counts.append(count_current)
            
            if count_current > 0:
                # First index for this order is next available index
                fi = next_index
                li = fi + count_current - 1
                next_index = li + 1
                all_combos.append(current)
            else:
                # Handle empty orders
                fi = 0
                li = -1
            
            firsts.append(fi)
            lasts.append(li)
                
        if all_combos:
            combined = np.vstack(all_combos)
            self._create_exponent_table(combined, all_combos)
            self._create_order_table(counts, firsts, lasts)
            self._create_lookup_tables(combined)
            self._add_conjugate_column()
            self._add_abaqus_and_k_tg_columns()
        else:
            self._create_empty_tables()

    def _create_exponent_table(self, combined, all_combos):
        """Create main exponent table DataFrame with metadata columns.
        
        Args:
            combined: Numpy array of all exponent combinations
            all_combos: List of arrays separated by order
        """
        # Create local indices per order
        local_idx = []
        for c in all_combos:
            local_idx.extend(range(1, len(c) + 1))
        
        # Create order labels
        orders = []
        for i, c in enumerate(all_combos):
            orders.extend([i + 1] * len(c))
        
        # Convert to list of tuples for DataFrame
        exponents_list = list(map(tuple, combined))
        
        self.exponent_table = pd.DataFrame({
            'exponents': exponents_list,
            'global_index': range(1, len(combined) + 1),
            'local_index': local_idx,
            'order': orders
        })

    def _create_order_table(self, counts, firsts, lasts):
        """Create order summary table with count and index ranges per order.
        
        Args:
            counts: List of term counts per order
            firsts: List of first global indices per order
            lasts: List of last global indices per order
        """
        self.order_table = pd.DataFrame({
            'order': range(1, self.max_order + 1),
            'count': counts,
            'first_index': firsts,
            'last_index': lasts
        })

    def _create_lookup_tables(self, combined):
        """Create auxiliary lookup structures for exponent tuples.
        
        Args:
            combined: Numpy array of all exponent combinations
        """
        self.exponent_list = [tuple(x) for x in combined]
        self.exponent_dict = {k: v for v, k in enumerate(self.exponent_list)}

    def _add_conjugate_column(self):
        """Add conjugate index column by swapping variable blocks.
        
        For each term, computes its conjugate by swapping halves of:
        - Mode variables (first num_modes/2 and last num_modes/2)
        - Forcing variables (first num_forcing/2 and last num_forcing/2)
        Stores global index of conjugate term.
        """
        n_modes = self.num_modes
        n_forcing = self.num_forcing
        half_mode = n_modes // 2
        half_forcing = n_forcing // 2
        
        conjugates = []
        for t in self.exponent_table['exponents']:
            # Split into modes and forcing blocks
            modes = t[:n_modes]
            forcing = t[n_modes:n_modes+n_forcing]
            # Swap halves within each block
            new_modes = modes[half_mode:] + modes[:half_mode]
            new_forcing = forcing[half_forcing:] + forcing[:half_forcing]
            conjugate_tuple = tuple(new_modes + new_forcing)
            conjugate_idx = self.exponent_dict[conjugate_tuple]
            conjugates.append(conjugate_idx + 1)  # Convert to 1-based index
        self.exponent_table['global_index_cc'] = conjugates

    def _add_abaqus_and_k_tg_columns(self):
        """Add solver-specific columns for complex conjugates.
        
        Generates:
        - 'Abaqus' flag (currently unused placeholder)
        - K_tg columns mapping terms to real/imaginary components
          handling both self-conjugate terms and conjugate pairs
        """
        self.exponent_table['Abaqus'] = False
        
        n = len(self.exponent_table)
        processed = [False] * n
        col1 = [None] * n  # K_tg(+Re(w))
        col2 = [None] * n  # K_tg(-Re(w))
        col3 = [None] * n  # K_tg(+Im(w))
        col4 = [None] * n  # K_tg(-Im(w))
        current_counter = 0
        
        for i in range(n):
            if processed[i]:
                continue
                
            cc_global = self.exponent_table.iloc[i]['global_index_cc']
            j = cc_global - 1  # Convert to 0-based index
            
            if j == i:  # Self-conjugate
                col1[i] = current_counter
                col2[i] = current_counter + 1
                col3[i] = current_counter + 2
                col4[i] = current_counter + 3
                current_counter += 4
                processed[i] = True
            else:  # Conjugate pair
                col1[i] = current_counter
                col2[i] = current_counter + 1
                col3[i] = current_counter + 2
                col4[i] = current_counter + 3
                col1[j] = current_counter
                col2[j] = current_counter + 1
                col3[j] = current_counter + 3  # Swap imaginary components
                col4[j] = current_counter + 2
                current_counter += 4
                processed[i] = True
                processed[j] = True
                
        # Add columns
        self.exponent_table['K_tg(+Re(w))'] = col1
        self.exponent_table['K_tg(-Re(w))'] = col2
        self.exponent_table['K_tg(+Im(w))'] = col3
        self.exponent_table['K_tg(-Im(w))'] = col4

    def _create_empty_tables(self):
        """Initialize empty tables when no valid combinations exist."""
        self.exponent_table = pd.DataFrame(columns=[
            'exponents', 'global_index', 'local_index', 'order', 
            'global_index_cc', 'Abaqus', 
            'K_tg(+Re(w))', 'K_tg(-Re(w))', 'K_tg(+Im(w))', 'K_tg(-Im(w))'
        ])
        self.order_table = pd.DataFrame(columns=['order', 'count', 'first_index', 'last_index'])

if __name__ == "__main__":
    max_order = 2
    num_variables = 4
    num_forcing = 2
    max_order_modes = 2
    maximum_order_forcing = 2
    
    generator = MultiExponentGenerator(
        num_variables, 
        num_forcing, 
        max_order,
        max_order_modes,
        maximum_order_forcing
    )
    generator.build_tables()
    
    print("Exponent Combinations:\n", generator.exponent_table)
    print("\nOrder Information:\n", generator.order_table)
    print("\ngenerator.exponent_dict:\n", generator.exponent_dict)
    

    
    query = ExponentQuery(
        generator.exponent_table,
        generator.order_table,
        generator.exponent_dict,
        generator.num_modes,
        generator.num_forcing
    )
    

    # Test find_less_or_equal_combinations
    target = (1, 1, 1, 1, 1, 1)
    print(f"\nAll combinations less than or equal to {target} (order >= 2):\n", 
          query.find_less_combinations(target))

    # Test new function: find order=1 forcing unit vectors with global index > gamma's index
    gamma = (1, 0, 0, 0, 0, 0)  # First order=1 exponent
    print(f"\nOrder=1 forcing unit vectors with higher global index than {gamma}:")
    result = query.find_less_combinations_with_order1(gamma)
    print(result)
    
    # Additional test: use a middle-indexed gamma
    gamma2 = (0, 0, 1, 0, 0, 0)  # Global index should be 3
    print(f"\nOrder=1 forcing unit vectors with higher global index than {gamma2}:")
    result2 = query.find_less_combinations_with_order1(gamma2)
    print(result2)