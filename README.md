[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7473300.svg)](https://doi.org/10.5281/zenodo.7646063)
# Ebola epidemic model with dynamic population and memory

## Description
This repository contains Julia code used to generate figures for the paper titled "Ebola epidemic model with dynamic population and memory". The code generates the figures based on data stored in the `datoswho.csv` file.

## Dependencies
The code requires the following Julia packages to be installed:
- FdeSolver.jl
- CSV.jl
- DataFrames.jl
- DelimitedFiles.jl
- SpecialFunctions.jl
- Plots.jl
- Optim.jl
- StatsBase.jl
- LinearAlgebra.jl
- Interpolations.jl
- LaTeXStrings.jl 

## Usage
1. Clone the repository to your local machine.
```bash
git clone https://github.com/moeinkh88/Ebola_modelling.git
```
2. Install the required packages listed above

```julia
using Pkg
Pkg.add("PackageName")
```

3. Run Allplots_Models280.jl to generate the figures.

## License
This code is released under the MIT License.
