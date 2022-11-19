# Author and license
__author__ = "Riccardo Albiero"
__license__ = "MIT"

# Optional variables
__description__ = 'Python library for design and fabrication of femtosecond laser written integrated photonic circuits.'
__keywords__ = 'python g-code integrated-circuits quantum-optics'

# Support for "from femto import *"
# @see: https://stackoverflow.com/a/41895257
# @see: https://stackoverflow.com/a/35710527
__all__ = [
    'Parameters',
    'LaserPathParameters',
    'WaveguideParameters',
    'TrenchParameters',
    'GcodeParameters',
    'LaserPath',
    '_Waveguide',
    'Waveguide',
    'coupler',
    '_Marker',
    'Marker',
    '_RasterImage',
    'RasterImage',
    'Trench',
    'TrenchColumn',
    'PGMCompiler',
    'PGMTrench',
    'Device',
    'Cell',
    'helpers',
]

from .helpers import *
from .Parameters import LaserPathParameters, WaveguideParameters, MarkerParameters, RasterImageParameters, \
    TrenchParameters, GcodeParameters
from .LaserPath import LaserPath
from .Waveguide import _Waveguide, Waveguide, coupler
from .Marker import _Marker, Marker
from .RasterImage import _RasterImage, RasterImage
from .Trench import Trench, TrenchColumn
from .PGMCompiler import PGMCompiler, PGMTrench
from .Cell import Cell, Device
