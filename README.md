<div id="top"></div>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/45992199/205449527-d349ee82-39fb-4e1f-b25b-dbd2260ad9a4.svg" width="480">
    <img alt="Logo" src="https://user-images.githubusercontent.com/45992199/205449385-341529d6-0575-430c-a0b4-62f50579db19.svg" width="480">
  </picture>
</div>

<p align="center">
  <br>
  <br>
  Python library for the design of femtosecond laser-written integrated photonic circuits.
</p>

<div align="center">

[![Latest version](https://img.shields.io/static/v1?label=version&message=2.0.0&color=yellowgreen)](https://github.com/ricalbr/femto/releases/latest)
[![Python 3.9 +](https://img.shields.io/static/v1?label=python&message=3.9%20%2B&color=blueviolet)](https://www.python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Codecov](https://codecov.io/gh/ricalbr/femto2/branch/main/graph/badge.svg?token=1E69VOP4ED)](https://codecov.io/gh/ricalbr/femto2)
[![Tests](https://github.com/ricalbr/femto/actions/workflows/tests.yml/badge.svg)](https://github.com/ricalbr/femto/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/femto2/badge/?version=latest)](https://femto2.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ricalbr/femto2/main.svg)](https://results.pre-commit.ci/latest/github/ricalbr/femto2/main)

</div>

`femto` is an open-source package for the design of integrated optical circuits.
The library consists of a growing list of parts and modules, which can be composed to construct complex optical components and large circuits.
The optical components can be plotted and exported to a .pgm file for the fabrication of the circuit.

## <img src="https://mir-s3-cdn-cf.behance.net/project_modules/disp/511fdf30195555.560572b7c51e9.gif" alt="femto logo" width="32"> Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Issues](#issues)
<!-- * [License](#license) -->

## Installation

The package can be installed using `pip` (use Python 3.11)

```bash
python3 -m venv .venvs/femto
source .venvs/minimal/bin/activate
pip install git+ssh://git@github.com/ricalbr/femto2.git
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Here a brief example on how to use the library.

First, import all the required modules

```python
from femto.curves import sin
from femto.device import Device
from femto.pgmcompiler import PGMCompiler
from femto.waveguide import Waveguide
```

Set waveguide parameters

```python
PARAM_WG = dict(scan=4,
                speed=7.5,
                depth=0.050,
                radius=25)
```

Create a list of waveguides as

```python
wgs = []

# SWG
wg = Waveguide(**PARAM_WG)
wg.start([-2, 4.5, 0.050])
wg.linear([27, 4.5, 0.050], mode='ABS')
wg.end()
wgs.append(wg)

# MZI
for i in range(6):
    wg = Waveguide(**PARAM_WG)
    wg.start([-2, 5+i*0.080, 0.500])
    wg.linear([10, 0, 0], mode='INC')
    wg.mzi(dy=(-1)**i * 0.037, dz=0, fx=sin)
    wg.linear([27, 5+i*0.080, 0.500], mode='ABS')
    wg.end()
    wgs.append(wg)
```

Now that the waveguides are defined, we set the fabrication parameters

```python
PARAM_GC = dict(filename='MZIs.pgm',
                laser='PHAROS',
                samplesize=(25, 10),
                rotation_angle=0.0)
```

Create a `Device` object that allows to store the waveguides and plot them

```python
# CIRCUIT
circuit = Device(**PARAM_GC)

# Add waveguides to circuit
circuit.extend(wgs)

# Make a plot of the circuit
circuit.plot2d()
```

Export the G-Code with the following commands

```python
# Export G-Code file
with PGMCompiler(**PARAM_GC) as G:
    G.tic()
    with G.repeat(6):
        for i, wg in enumerate(wgs):
            G.comment(f' +--- Mode: {i + 1} ---+')
            G.write(wg.points)
    G.toc()
    G.go_origin()

```

Other example files can be found [here](https://github.com/ricalbr/femto/tree/main/examples).

<p align="right">(<a href="#top">back to top</a>)</p>

## Documentation
The complete documentation can be found [here](https://femto.readthedocs.io/en/latest/).

<p align="right">(<a href="#top">back to top</a>)</p>

## Issues

To request features or report bugs open an issue [here](https://github.com/ricalbr/femto/issues)

<p align="right">(<a href="#top">back to top</a>)</p>

---

![image](https://github.com/user-attachments/assets/8f952e70-cde9-41cd-81f8-58a0ab947620)

You can finally rest, femto-writer...
