<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://user-images.githubusercontent.com/45992199/169658724-72260dc7-26c5-4ff4-bdbb-a0ff6635d893.png" alt="femto" width="480">
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
femto is an open-source package for the design of integrated optical circuits. It allows to define optical components and to compile a .pgm file for the microfabrication of the circuit. The library consists of growing list of parts, which can be composed into larger circuits.
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
Fetmo can be installed via pip via
```bash
git clone git@github.com:ricalbr/femto.git
cd femto
pip install -e .
```

Alternatively, using GitHub Desktop and Spyder IDE, follow

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
from femto import PGMCompiler, Waveguide
from femto.Parameters import GcodeParameters, WaveguideParameters
```

Define a circuit as a list of waveguides
```python
waveguides = []
```

Set waveguide and farbication parameters
```python
PARAMETERS_WG = WaveguideParameters(
    scan=6,
    speed=20,
    depth=0.035,
    radius=15,
)

PARAMETERS_GC = GcodeParameters(
    filename='MZIs.pgm',
    lab='CAPABLE',
    samplesize=(25, 25),
    angle=0.0
)
```

Start by adding waveguides to the circuit
```python
# SWG
wg = Waveguide(PARAMETERS_WG)
wg.start([-2, 0, 0.035])
wg.linear([25, 0.0, 0.0])
wg.end()
waveguides.append(wg)

# MZI
for i in range(6):
    wg = Waveguide(PARAMETERS_WG)
    wg.start([-2, 0.3 + i * 0.08, 0.035]) \
        .linear([1.48, 0, 0]) \
        .arc_mzi((-1) ** (i) * 0.037) \
        .linear([1.48, 0, 0])
    wg.end()
    waveguides.append(wg)
```

Export the G-Code with the following commands
```python
# Waveguide G-Code
with PGMCompiler(PARAMETERS_GC) as gc:
    with gc.repeat(6):
        for wg in waveguides:
            gc.comment(f' +--- Modo: {i + 1} ---+')
            gc.write(wg.points)
```
Other example files can be found [here](https://github.com/ricalbr/femto/tree/main/examples)

<p align="right">(<a href="#top">back to top</a>)</p>

## Issues
To request features or report bugs open an issue [here](https://github.com/ricalbr/femto/issues)

<p align="right">(<a href="#top">back to top</a>)</p>
