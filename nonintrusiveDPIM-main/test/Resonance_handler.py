
import numpy as np
import sys


class ResonanceHandler:
    def __init__(self, dpim):
        """Initialize resonance detector with DPIM system data.
        
        Args:
            dpim: Data structure containing eigenvalues and parameters
        """
        self.dpim = dpim

    def detect_outer_resonances(self):
        """Detect critical resonances between external frequencies and system eigenvalues.
        
        Prints a warning if any outer resonance is detected but does not exit the program.
        """
        resonance_detected = False
        for idx, row in self.dpim.mexp.exponent_table.iterrows():
            gamma = row['exponents']
            s = np.dot(self.dpim.lambda_tilde, gamma)
            for lambda_outer in self.dpim.lambda_outer:
                if abs(s.imag - lambda_outer.imag) < self.dpim.resonance_tol:
                    resonance_detected = True
                    break  # Exit inner loop early
            if resonance_detected:
                break  # Exit outer loop early after first detection

        if resonance_detected:
            print("WARNING: There are outer resonances. THE SYSTEM MAY NOT BE RESOLVABLE.")
        else:
            print("No outer resonances detected. Proceeding with computation.")

    def _detect_resonances(self, gamma, s):
        """Detect resonances between a term's eigenvalue combination and system eigenvalues.
        
        Args:
            gamma: Exponent vector of the current term
            s: Dot product of gamma and extended eigenvalues (γ·λ̃)
            
        Returns:
            List of resonant mode indices, or None if no resonances
        """
        resonant_modes = []
        for i in range(self.dpim.n_modes):
            if abs(s.imag - self.dpim.lambda_[i].imag) < self.dpim.resonance_tol:
                print("+++++++++++++++++++++++++++++RESONANCE+++++++++++++++++++++++++")
                self._handle_resonance(gamma, i)
                resonant_modes.append(i)
        print(f"resonance mpdes: {resonant_modes}")
        return resonant_modes if resonant_modes else None

    def _handle_resonance(self, gamma, mode):
        """Record detected resonance in system data structures.
        
        Args:
            gamma: Exponent vector of resonant term
            mode: Index of resonant mode
        """
        print(f"Resonance detected between term {gamma} and mode {mode}")
        self.dpim.resonance_relations[mode].append(gamma)
        self.dpim.resonant_terms.add(tuple(gamma))