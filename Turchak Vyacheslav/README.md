# Bachelor's Qualification Thesis

## Author
* **Full Name:** Vyacheslav Turchak
* **Group:** PMP-42
* **Department:** Department of Applied Mathematics
* **Faculty:** Faculty of Applied Mathematics and Informatics
* **University:** Ivan Franko National University of Lviv

---

## Thesis Title
**Numerical solution of two-dimensional problems of pollution transport in soils**

### Abstract
This work focuses on the mathematical modeling and numerical analysis of contaminant migration processes in soil environments. The **Finite Element Method (FEM)** is applied for the spatial discretization of the two-dimensional mathematical model (governing convection-diffusion equations incorporating adsorption and decay factors). The simulation software is developed in Python, enabling computer experiments, forecasting the spread of pollution fronts, and visual evaluation of the transport dynamics.

---

## Folder Structure
* `dyplom.pdf` — The full text of the bachelor's thesis in PDF format.
* `main.py` — Source code files of the application (scripts for mesh generation, stiffness/mass matrix assembly, linear solvers, and plotting).
* `README.md` — This file containing the project description.

---

## Execution Guide

The simulation framework is built using **Python 3**. It relies on standard scientific computing and data science libraries to assemble the FEM matrices and render the results.

### 1. Prerequisites
Before running the simulation, ensure you have the required packages installed. You can install them via Terminal using the following command:

```bash
pip install numpy scipy matplotlib pandas scikit-learn