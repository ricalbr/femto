<div id="top"></div>

<!-- PROJECT LOGO -->
<div>
  <a href="https://github.com/github_username/repo_name">
    <h1> 
      <img src="https://mir-s3-cdn-cf.behance.net/project_modules/disp/511fdf30195555.560572b7c51e9.gif" 
        alt="femto logo" 
        width="32">
      femto 
    </h1>
  </a>
</div>

> Python library for design and fabrication of femtosecond laser written integrated photonic circuits.

## Table of Contents

* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Issues](#issues)

<!-- * [License](#license) -->

## Features

femto is an open-source package for the design of integrated optical circuits. It allows to define optical components
and to compile a .pgm file for the microfabrication of the circuit. The library consists of growing list of parts, which
can be composed into larger circuits.
The following components are implemented:

* Waveguide class, allowing easy chaining of bends and straight segments. Including:
    * Circular and sinusoidal arcs
    * Directional couplers
    * Mach-Zehnder interferometers
    * Spline segments and 3D bridges
* Class for different markers
* Trench class, allowing easy path generation for trenches with various geometries
* G-Code compiler class

<p align="right">(<a href="#top">back to top</a>)</p>

## Setup

The preferred way to install `femto` is using `conda`.
First, install `anaconda` or `miniconda` following the instruction [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda). 

Then create a virtual environment and install all the package dependencies using the following command:

``` bash
conda env create -n femto --file env.yml
```

The virtual environment can be updated using the `env.yml` file as well:

```bash
conda activate femto
# conda activate <env_name>
conda env update --file env.yml --prune
```

Alternatively, `femto` can also be installed via `pip` via:

```bash
git clone git@github.com:ricalbr/femto.git
cd femto
pip install -e .
```

or using GitHub Desktop and Spyder IDE as follow:

1. Download the repository on GitHub Desktop
2. Run the following command in the Spyder console

```bash
pip install -e C:\Users\<user>\Documents\GitHub\femto
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Here a brief example on how to use the library.

First, import all the required packages

```python
import matplotlib.pyplot as plt
from femto import Cell, PGMCompiler, Waveguide
```

Define a `Cell` which represents a circuit

```python
circuit = Cell()
```

Set waveguide and farbication parameters

```python
PARAMETERS_WG = dict(
    scan=6,
    speed=20,
    depth=0.035,
    radius=15,
)

PARAMETERS_GC = dict(
    filename='MZIs.pgm',
    lab='CAPABLE',
    samplesize=(25, 25),
    rotation_angle=0.0
)
```

Start by adding waveguides to the circuit

```python
# SWG
wg = Waveguide(PARAMETERS_WG)
wg.start([-2, 0, 0.035])
wg.linear([11, 0.0, 0.0], mode='ABS')
wg.end()
circuit.add(wg)

# MZI
for i in range(6):
    wg = Waveguide(PARAMETERS_WG)
    wg.start([-2, 0.3 + i * 0.08, 0.035])
        .linear([3.48, 0, 0])
        .arc_mzi((-1) ** (i) * 0.037)
        .linear([3.48, 0, 0])
    wg.end()
    circuit.add(wg)
```

Make a plot of the circuit

```python
circuit.plot2d()
plt.show()
```

Export the G-Code with the following commands

```python
# Waveguide G-Code
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(6):
        for wg in circuit.waveguides:
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)
```

Other example files can be found [here](https://github.com/ricalbr/femto/tree/main/examples)

<p align="right">(<a href="#top">back to top</a>)</p>

## Issues

To request features or report bugs open an issue [here](https://github.com/ricalbr/femto/issues)

<p align="right">(<a href="#top">back to top</a>)</p>
