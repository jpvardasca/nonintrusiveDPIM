# Non-Intrusive DPIM for Reduced Order Modelling in Nonlinear Structural Dynamics

A fully automated framework for constructing nonlinear reduced-order models (ROMs) directly from Abaqus finite element models using the Direct Parametrisation of Invariant Manifolds (DPIM), Enhanced Enforced Displacement Stiffness Evaluation Procedure (EED-StEP), and Harmonic Balance Method (HBM). Developed as part of the work presented in *Non-intrusive Direct Parametrisation of Invariant Manifolds for Reduced Order Modelling in Nonlinear Structural Dynamics*. 【1-1de457】

---

## Overview

Nonlinear finite element simulations are often prohibitive for design iterations, optimization studies, uncertainty quantification, digital twins, and real-time applications. While advanced nonlinear model reduction methods based on invariant manifolds provide highly accurate reduced-order models, they typically require intrusive access to the finite element solver.

This repository bridges that gap by providing a **fully non-intrusive workflow for Abaqus**, allowing commercial finite element models to be reduced without requiring access to the solver source code. The framework treats Abaqus as a black box and automatically extracts the required nonlinear information through static analyses and tangent stiffness matrices. 【1-1de457】

---

## Main Features

- Automated extraction of nonlinear stiffness coefficients from Abaqus
- Enhanced Enforced Displacement Stiffness Evaluation Procedure (EED-StEP)
- Tangent-stiffness-based coefficient identification
- Direct Parametrisation of Invariant Manifolds (DPIM)
- Support for arbitrary expansion orders
- Complex Normal Form and Graph Style parametrizations
- Realification of complex-valued ROMs
- Harmonic Balance frequency response analysis
- Coefficient and stiffness matrix caching
- Abaqus User Element (UEL) implementation
- Total Lagrangian Saint-Venant Kirchhoff formulation
- Benchmark examples for beams and torsional structures

---

## Methodology

The workflow consists of four main stages:

```text
Abaqus Finite Element Model
            │
            ▼
    Modal Analysis
            │
            ▼
      EED-StEP
   Coefficient Extraction
            │
            ▼
        DPIM
 Manifold Construction
            │
            ▼
     Realification
            │
            ▼
 Harmonic Balance Method
            │
            ▼
 Nonlinear Dynamic Response
```

### 1. Linear Modal Analysis

Mass and stiffness matrices are extracted from Abaqus and used to compute:

- Natural frequencies
- Mode shapes
- Reduced modal basis

These modes define the master coordinates of the invariant manifold. 【1-1de457】

### 2. Enhanced StEP

The nonlinear stiffness coefficients are obtained by prescribing carefully selected displacement fields and extracting Abaqus tangent stiffness matrices.

Compared with classical Enforced Displacement methods, EED-StEP significantly reduces the number of required simulations, making high-order ROM construction practical for industrial-scale models. 【1-1de457】

### 3. Direct Parametrisation of Invariant Manifolds

DPIM constructs:

- A nonlinear manifold mapping
- Reduced dynamics on the manifold

through the recursive solution of cohomological equations.

The resulting ROM accurately reproduces the nonlinear dynamics of the original finite element model while drastically reducing computational cost. 【1-1de457】

### 4. Harmonic Balance Analysis

The reduced model can be solved using the Harmonic Balance Method (HBM) to efficiently compute:

- Frequency response curves
- Backbone curves
- Stable and unstable branches
- Saddle-node bifurcations
- Superharmonic resonances

without expensive transient simulations. 【1-1de457】

---

## Repository Structure

```text
├── abaqus/
│   ├── input_generation/
│   ├── modal_analysis/
│   ├── stiffness_extraction/
│   └── postprocessing/
│
├── uel/
│   └── svk_total_lagrangian.for
│
├── step/
│   ├── eed.py
│   ├── coefficient_extraction.py
│   └── cache_manager.py
│
├── dpim/
│   ├── eigenproblem.py
│   ├── multiindex.py
│   ├── resonances.py
│   ├── cohomological_equations.py
│   ├── manifold.py
│   └── realification.py
│
├── hbm/
│   ├── frequency_response.py
│   └── continuation.py
│
├── examples/
│   ├── clamped_clamped_beam/
│   ├── cantilever_beam/
│   └── torsional_resonator/
│
├── docs/
│
└── README.md
```

---

## Requirements

### Software

- Abaqus/Standard
- Python 3.10+
- Fortran compiler (Intel Fortran or gfortran)
- NumPy
- SciPy
- Matplotlib

### Optional

- pyHBM
- pandas
- Jupyter Notebook

---

## Quick Start

### 1. Compute Linear Modes

```bash
python run_modal_analysis.py
```

Generates modal data from the Abaqus model.

### 2. Extract Nonlinear Coefficients

```bash
python run_step.py
```

This step automatically:

- Generates Abaqus input files
- Prescribes displacement fields
- Executes nonlinear static analyses
- Extracts tangent stiffness matrices
- Computes quadratic and cubic stiffness coefficients

### 3. Construct the ROM

```bash
python run_dpim.py
```

Computes:

- Invariant manifold coefficients
- Reduced dynamics coefficients

### 4. Compute Frequency Response

```bash
python run_hbm.py
```

Produces:

- Backbone curves
- Frequency response curves
- Stability information

---

## Abaqus User Element

The framework includes a custom Abaqus User Element (UEL) implementing a:

- Total Lagrangian formulation
- Saint-Venant Kirchhoff constitutive model
- Consistent tangent stiffness matrix

The UEL provides the polynomial nonlinearities required by DPIM and ensures compatibility with the EED-StEP extraction procedure. 【1-1de457】

Compile and run using:

```bash
abaqus job=model user=uel_subroutine.for
```

---

## Validation Examples

### Clamped-Clamped Beam

Demonstrates:

- Geometric stiffening
- Backbone curves
- Saddle-node bifurcations

### Cantilever Beam

Demonstrates:

- Large-amplitude bending
- Invariant manifold folding
- Hardening frequency response

### Torsional Resonator

Demonstrates:

- Torsional nonlinearities
- Warping effects
- Convergent nonlinear frequency responses

The ROM predictions show excellent agreement with intrusive reference solutions while requiring significantly lower computational effort. 【1-1de457】

---

## Applications

This framework is suitable for:

- Nonlinear structural dynamics
- Digital twins
- Virtual prototyping
- Design optimization
- MEMS and resonators
- Aerospace structures
- Reduced-order modelling research
- Frequency response prediction

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{pereira2025dpim,
  title={Non-intrusive Direct Parametrisation of Invariant Manifolds for Reduced Order Modelling in Nonlinear Structural Dynamics},
  author={Pereira, Joana R. and Martins, Tiago S. and Trainotti, Francesco and Moleiro, Filipa and Vizzaccaro, Alessandra},
  journal={Journal of Structural Dynamics},
  year={2025}
}
```

---

## Acknowledgements

This work was developed within the framework of the research presented in:

**Joana R. Pereira**, *Non-intrusive Direct Parametrisation of Invariant Manifolds for Reduced Order Modelling in Nonlinear Structural Dynamics*, Instituto Superior Técnico, 2025. 【1-1de457】

Special thanks to:

- Tiago Martins
- Francesco Trainotti
- Filipa Moleiro
- Alessandra Vizzaccaro

for their contributions to the development and validation of the methodology. 【1-1de457】

---

## License

LICENSE

---

### Repository Description

> Automated non-intrusive reduced-order modelling framework for nonlinear structural dynamics in Abaqus using Enhanced StEP, Direct Parametrisation of Invariant Manifolds (DPIM), and Harmonic Balance analysis.
