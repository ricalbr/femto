from __future__ import annotations

import copy
import math
from collections import deque
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import dill
import numpy as np
import pytest
from femto.helpers import flatten
from femto.helpers import listcast
from femto.pgmcompiler import farcall
from femto.pgmcompiler import PGMCompiler
from femto.pgmcompiler import sample_warp


@pytest.fixture
def param() -> dict:
    p = {
        'filename': 'test.pgm',
        'n_glass': 1.50,
        'n_environment': 1.33,
        'export_dir': 'G-Code',
        'samplesize': (25, 25),
        'rotation_angle': 1.0,
        'long_pause': 1.0,
        'short_pause': 0.025,
        'speed_pos': 10,
        'flip_x': True,
        'minimal_gcode': True,
    }
    return p


@pytest.fixture
def empty_mk(param) -> PGMCompiler:
    return PGMCompiler(**param)


def test_default_values() -> None:
    G = PGMCompiler(filename='prova', n_glass=1.50, n_environment=1.33)
    assert G.filename == 'prova'
    assert G.export_dir == ''
    assert G.samplesize == (100, 50)
    assert G.laser == 'PHAROS'
    assert G.home is False
    assert G.shift_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(0.0)
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(0.5)
    assert G.short_pause == float(0.05)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(5.0)
    assert G.flip_x is False
    assert G.flip_y is False
    assert G.minimal_gcode is False
    assert G.verbose is True


def test_gcode_values(param) -> None:
    G = PGMCompiler(**param)
    assert G.filename == 'test.pgm'
    assert G.export_dir == 'G-Code'
    assert G.samplesize == (25, 25)
    assert G.laser == 'PHAROS'
    assert G.home is False
    assert G.shift_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(math.radians(1.0))
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(1.0)
    assert G.short_pause == float(0.025)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(10)
    assert G.flip_x is True
    assert G.flip_y is False
    assert G.minimal_gcode is True
    assert G.verbose is True


def test_pgm_from_dict(param) -> None:
    G = PGMCompiler.from_dict(param)
    assert G.filename == 'test.pgm'
    assert G.export_dir == 'G-Code'
    assert G.samplesize == (25, 25)
    assert G.laser == 'PHAROS'
    assert G.home is False
    assert G.shift_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(math.radians(1.0))
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(1.0)
    assert G.short_pause == float(0.025)
    assert G.output_digits == int(6)
    assert G.speed_pos == float(10)
    assert G.flip_x is True
    assert G.flip_y is False
    assert G.minimal_gcode is True
    assert G.verbose is True


def test_pgm_from_dict_update(param) -> None:
    G = PGMCompiler.from_dict(param, output_digits=9, laser='UWE', samplesize=(25, 30), long_pause=1.0)

    assert G.filename == 'test.pgm'
    assert G.export_dir == 'G-Code'
    assert G.samplesize == (25, 30)
    assert G.laser == 'UWE'
    assert G.home is False
    assert G.shift_origin == (0.0, 0.0)
    assert G.warp_flag is False
    assert G.n_glass == float(1.50)
    assert G.n_environment == float(1.33)
    assert G.rotation_angle == float(math.radians(1.0))
    assert G.aerotech_angle == float(0.0)
    assert G.long_pause == float(1.0)
    assert G.short_pause == float(0.025)
    assert G.output_digits == int(9)
    assert G.speed_pos == float(10)
    assert G.flip_x is True
    assert G.flip_y is False
    assert G.minimal_gcode is True
    assert G.verbose is True


def test_repr(param) -> None:
    r = PGMCompiler(**param).__repr__()
    print()
    print(r)
    cname, _ = r.split('@')
    assert cname == 'PGMCompiler'


def test_filename_error(param) -> None:
    param['filename'] = None
    with pytest.raises(ValueError):
        PGMCompiler(**param)


def test_enter_exit_method_default(param) -> None:
    with PGMCompiler(**param) as G:
        print(G._instructions)
        assert G._instructions == deque(
            [
                '; SETUP PHAROS - CAPABLE LAB\n',
                '\n',
                'ENABLE X Y Z\n',
                'PSOCONTROL X RESET\n',
                'PSOOUTPUT X CONTROL 3 0\n',
                'PSOCONTROL X OFF\n',
                '\n',
                'G71     ; DISTANCE UNITS: METRIC\n',
                'G76     ; TIME UNITS: SECONDS\n',
                'G90     ; ABSOLUTE MODE\n',
                'G359    ; WAIT MODE NOWAIT\n',
                'G108    ; VELOCITY ON\n',
                'G17     ; ROTATIONS IN XY PLANE\n',
                '\n',
                '; NSCOPETRIG\n',
                '; MSGCLEAR -1\n',
                '\n',
                'G4 P1.0 ; DWELL\n',
                '\n',
            ]
        )
    assert G._instructions == deque([])

    fold = Path('.') / param['export_dir']
    file = fold / param['filename']
    file.unlink()
    fold.rmdir()


def test_enter_exit_method_verbose() -> None:
    p = dict(
        filename='testPGM.pgm',
        n_glass=1.5,
        n_environment=1.33,
        laser='ant',
        samplesize=(25, 25),
        home=True,
        aerotech_angle=2.0,
        rotation_angle=1.0,
        flip_x=True,
        verbose=False,
    )
    with PGMCompiler(**p) as G:
        print(G._instructions)
        assert G._instructions == deque(
            [
                '; SETUP ANT - DIAMOND LAB\n',
                '\n',
                'ENABLE X Y Z\n',
                'PSOCONTROL Z RESET\n',
                'PSOOUTPUT Z CONTROL 0 1\n',
                'PSOCONTROL Z OFF\n',
                '\n',
                'G71     ; DISTANCE UNITS: METRIC\n',
                'G76     ; TIME UNITS: SECONDS\n',
                'G90     ; ABSOLUTE MODE\n',
                'G359    ; WAIT MODE NOWAIT\n',
                'G108    ; VELOCITY ON\n',
                'G17     ; ROTATIONS IN XY PLANE\n',
                '\n',
                '; NSCOPETRIG\n',
                '; MSGCLEAR -1\n',
                '\n',
                'G4 P1.0 ; DWELL\n',
                '\n',
                '\n; ACTIVATE AXIS ROTATION\n',
                'G1 X0.000000 Y0.000000 Z0.000000 F5.000000\n',
                'G84 X Y\n',
                'G4 P0.05 ; DWELL\n',
                'G84 X Y F2.0\n\n',
                'G4 P0.05 ; DWELL\n',
            ]
        )
    assert G._instructions == deque([])

    file = Path('.') / p['filename']
    assert file.is_file()
    file.unlink()


def test_enter_exit_method() -> None:
    p = dict(
        filename='testPGM.pgm',
        n_glass=1.5,
        n_environment=1.33,
        laser='ant',
        samplesize=(25, 25),
        home=True,
        aerotech_angle=2.0,
        flip_x=True,
    )
    with PGMCompiler(**p) as G:
        print(G._instructions)
        assert G._instructions == deque(
            [
                '; SETUP ANT - DIAMOND LAB\n',
                '\n',
                'ENABLE X Y Z\n',
                'PSOCONTROL Z RESET\n',
                'PSOOUTPUT Z CONTROL 0 1\n',
                'PSOCONTROL Z OFF\n',
                '\n',
                'G71     ; DISTANCE UNITS: METRIC\n',
                'G76     ; TIME UNITS: SECONDS\n',
                'G90     ; ABSOLUTE MODE\n',
                'G359    ; WAIT MODE NOWAIT\n',
                'G108    ; VELOCITY ON\n',
                'G17     ; ROTATIONS IN XY PLANE\n',
                '\n',
                '; NSCOPETRIG\n',
                '; MSGCLEAR -1\n',
                '\n',
                'G4 P1.0 ; DWELL\n',
                '\n',
                '\n; ACTIVATE AXIS ROTATION\n',
                'G1 X0.000000 Y0.000000 Z0.000000 F5.000000\n',
                'G84 X Y\n',
                'G4 P0.05 ; DWELL\n',
                'G84 X Y F2.0\n\n',
                'G4 P0.05 ; DWELL\n',
            ]
        )
    assert G._instructions == deque([])

    file = Path('.') / p['filename']
    assert file.is_file()
    file.unlink()


@pytest.mark.parametrize('p, n, expected', [(1, 1, 1), (0.5, 1, 0.5), (405, 1, 405), (2, 3, 6), (15, 15, 225)])
def test_total_dwell_time(param, p, n, expected) -> None:
    G = PGMCompiler(**param)
    for _ in range(n):
        G.dwell(p)
    assert G.total_dwell_time == expected


@pytest.mark.parametrize('xs, expected', [(1, 1), (0.5, 0.5), (-5, 5)])
def test_xsample(param, xs, expected) -> None:
    param['samplesize'] = (xs, xs)  # set ysample to the same value of xsample, just for testing purposes
    G = PGMCompiler(**param)
    assert G.xsample == expected


@pytest.mark.parametrize('ys, expected', [(-1, 1), (3.2, 3.2), (-18, 18)])
def test_ysample(param, ys, expected) -> None:
    param['samplesize'] = (ys, ys)  # set xsample to the same value of ysample, just for testing purposes
    G = PGMCompiler(**param)
    assert G.ysample == expected


@pytest.mark.parametrize(
    'ng, ne, expected',
    [(1.5, 1.5, 1), (1.5, 1.33, 1.5 / 1.33)],
)
def test_neff(param, ng, ne, expected) -> None:
    param['n_glass'] = ng
    param['n_environment'] = ne
    G = PGMCompiler(**param)
    assert G.neff == expected


@pytest.mark.parametrize(
    'laser, expected',
    [('ant', 'Z'), ('pharos', 'X'), ('UWE', 'X'), ('CaRbIdE', 'X')],
)
def test_pso_label(param, laser, expected) -> None:
    param['laser'] = laser
    G = PGMCompiler(**param)
    assert G.pso_axis == expected


@pytest.mark.parametrize(
    'laser, expectation',
    [
        ('ant', does_not_raise()),
        ('pharos', does_not_raise()),
        ('UWE', does_not_raise()),
        ('CARbide', does_not_raise()),
        ('satsuma', pytest.raises(ValueError)),
        ('PHAROS', does_not_raise()),
    ],
)
def test_pso_label_raise(param, laser, expectation) -> None:
    param['laser'] = laser
    G = PGMCompiler(**param)
    with expectation:
        print(G.pso_axis)


@pytest.mark.parametrize(
    'laser, expected',
    [('pharos', 0.00), ('PHAROS', 0.00), ('carbide', 0.00), ('CARBIDE', 0.00), ('uwe', 0.005), ('UWE', 0.005)],
)
def test_tshutter(param, laser, expected) -> None:
    param['laser'] = laser
    G = PGMCompiler(**param)
    assert pytest.approx(G.tshutter) == expected


@pytest.mark.parametrize(
    'laser, expectation',
    [
        ('pharos', does_not_raise()),
        ('UWE', does_not_raise()),
        ('', pytest.raises(ValueError)),
        ('CAPABLE', pytest.raises(ValueError)),
    ],
)
def test_tshutter_error(param, laser, expectation) -> None:
    param['laser'] = laser
    G = PGMCompiler(**param)
    with expectation:
        assert G.tshutter is not None


@pytest.mark.parametrize(
    'p, t, expected',
    [(1, 0, 0), (0, 0, 0), (-1, 5, 5), (0.5, 100, 50), (0.0, 79, 0)],
)
def test_dwell_time(param, p, t, expected) -> None:
    G = PGMCompiler(**param)
    for _ in range(t):
        G.dwell(p)
    assert pytest.approx(G.dwell_time) == expected
    del G


@pytest.mark.parametrize(
    'xy, expected',
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (50, 0),
        (-1, 0),
        (-2, 0),
        (-3, 0),
        (-4, 0),
        (-5, 0),
        (-50, 0),
    ],
)
def test_antiwarp_management(param, xy, expected) -> None:
    from pathlib import Path

    function_pickle = Path.cwd() / 'fwarp.pickle'
    G = PGMCompiler(**param)
    f = G.warp_management(opt=False)
    assert f(xy) == expected
    if function_pickle.is_file():
        function_pickle.unlink()


@pytest.mark.parametrize(
    'samplesize, expectation',
    [
        ((30, None), pytest.raises(ValueError)),
        ((None, 45), pytest.raises(ValueError)),
        ((None, None), pytest.raises(ValueError)),
    ],
)
def test_antiwarp_error(param, samplesize, expectation) -> None:
    from pathlib import Path

    function_pickle = Path.cwd() / 'fwarp.pickle'
    param['samplesize'] = samplesize
    G = PGMCompiler(**param)
    with expectation:
        assert G.warp_management(opt=True) is not None

    if function_pickle.is_file():
        function_pickle.unlink()


def test_antiwarp_pos_file_error(param) -> None:
    G = PGMCompiler(**param)
    with pytest.raises(FileNotFoundError):
        assert G.warp_management(opt=True)


@pytest.mark.parametrize('x, y', [(4, 5), (2, 4), (5, 2), (5, 0), (5, 3), (7, 7), (7, 0), (10, 5), (6, 10), (0, 5)])
def test_fwarp_load(param, x, y) -> None:
    G = PGMCompiler(**param)

    def fun(h, k):
        return h**2 * k

    file = Path.cwd() / 'fwarp.pickle'
    if not file.is_file():
        with open(file, 'wb') as f:
            dill.dump(fun, f)
    assert file.is_file()

    G_fun = G.warp_management(opt=True)
    assert G_fun(x, y) == fun(x, y)
    file.unlink()


def test_antiwarp_creation(param) -> None:
    from pathlib import Path

    fn = 'POS.txt'
    funpath = Path.cwd() / 'fwarp.pickle'
    pospath = Path.cwd() / fn

    x_in = np.linspace(0, 100, 50)
    y_in = np.linspace(0, 25, 10)
    X, Y = np.meshgrid(x_in, y_in)
    z_in = np.random.uniform(-0.030, 0.030, X.shape)

    M = np.stack([X.ravel(), Y.ravel(), z_in.ravel()], axis=-1)
    np.savetxt(fn, M, fmt='%.6f', delimiter=' ')

    G = PGMCompiler(**param)
    G_fun = G.warp_management(opt=True)
    assert callable(G_fun)
    assert funpath.is_file()
    funpath.unlink()
    pospath.unlink()


def test_antiwarp_plot(param) -> None:
    from pathlib import Path

    import matplotlib.pyplot as plt

    fn = 'POS.txt'
    pospath = Path.cwd() / fn

    x_in = np.linspace(0, 100, 50)
    y_in = np.linspace(0, 25, 10)
    X, Y = np.meshgrid(x_in, y_in)
    z_in = np.random.uniform(-0.030, 0.030, X.shape)

    M = np.stack([X.ravel(), Y.ravel(), z_in.ravel()], axis=-1)
    np.savetxt(fn, M, fmt='%.6f', delimiter=' ')

    G = PGMCompiler(**param)
    G.warp_generation(pospath, show=True)
    plt.close('all')
    assert True  # the plot is closed safely
    pospath.unlink()


@pytest.mark.parametrize(
    'xp, yp, mm',
    [
        (4, 5, 3.2),
        (6.7, 5.5, 3),
        (22, 12, 2),
    ],
)
def test_sample_warp(xp, yp, mm, param):
    from pathlib import Path

    param['aerotech_angle'] = 0.033

    sample_warp(pts_x=xp, pts_y=yp, margin=mm, parameters=param)

    sampling_script = Path('.') / param['export_dir'] / 'SAMPLE_WARP.pgm'
    assert sampling_script.is_file()

    with open(sampling_script) as f:
        for line in f:
            if line.startswith('$sizeX'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = float(var)
                assert var == float(param['samplesize'][0])
            elif line.startswith('$sizeY'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = float(var)
                assert var == float(param['samplesize'][1])
            elif line.startswith('$margin'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = float(var)
                assert var == mm
            elif line.startswith('$pointsX'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = int(float(var))
                assert var == int(xp)
            elif line.startswith('$pointsY'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = int(float(var))
                assert var == int(yp)
            elif line.startswith('$angle'):
                var = line.split('=')[-1].split(';')[0].strip('\t')
                var = float(var)
                assert var == float(param['aerotech_angle'])
            else:
                pass

    sampling_script.unlink()
    Path(Path('.') / param['export_dir']).rmdir()


def test_sample_warp_param_error(param):
    param['aerotech_angle'] = 0.55
    sampling_script = Path('.') / param['export_dir'] / 'SAMPLE_WARP.pgm'

    p1 = copy.deepcopy(param)
    p1['laser'] = None
    with pytest.raises(ValueError):
        assert sample_warp(7, 7, 3, p1) is not None
        assert not sampling_script.is_file()

    p2 = copy.deepcopy(param)
    p2['aerotech_angle'] = None
    with pytest.raises(TypeError):
        assert sample_warp(7, 7, 3, p2) is not None
        assert not sampling_script.is_file()

    p3 = copy.deepcopy(param)
    p3['samplesize'] = (None, 4)
    with pytest.raises(TypeError):
        assert sample_warp(7, 7, 3, p3) is not None
        assert not sampling_script.is_file()

    p4 = copy.deepcopy(param)
    p4['samplesize'] = (None, None)
    with pytest.raises(TypeError):
        assert sample_warp(7, 7, 3, p4) is not None
        assert not sampling_script.is_file()


def test_sample_warp_pos_file(param):
    param['aerotech_angle'] = 0.55

    x = np.linspace(0, 7, 15)
    y = np.linspace(0, 7, 15)
    z = np.random.uniform(-0.005, 0.005, x.size)
    M = np.stack([x, y, z], axis=-1)
    np.savetxt('POS.txt', M)

    pos_file = Path(__file__).cwd() / 'POS.txt'
    sampling_script = Path('.') / param['export_dir'] / 'SAMPLE_WARP.pgm'

    assert pos_file.is_file()
    sample_warp(7, 7, 3, param)
    assert not pos_file.is_file()
    sampling_script.unlink()
    Path(Path('.') / param['export_dir']).rmdir()


@pytest.mark.parametrize(
    'laser, expectation',
    [
        (None, pytest.raises(ValueError)),
        ('PAHROS', pytest.raises(ValueError)),
        ('Pharos', does_not_raise()),
        ('uva', pytest.raises(ValueError)),
        ('UWE', does_not_raise()),
    ],
)
def test_header_error(param, laser, expectation) -> None:
    param['laser'] = laser
    G = PGMCompiler(**param)
    with expectation:
        assert G.header() is None


def test_header(param) -> None:
    param['laser'] = 'ant'
    G = PGMCompiler(**param)
    G.header()
    assert G._instructions[4] == 'PSOOUTPUT Z CONTROL 0 1\n'

    param['laser'] = 'carbide'
    G = PGMCompiler(**param)
    G.header()
    assert G._instructions[4] == 'PSOOUTPUT X CONTROL 2 0\n'

    param['laser'] = 'pharos'
    G = PGMCompiler(**param)
    G.header()
    assert G._instructions[4] == 'PSOOUTPUT X CONTROL 3 0\n'

    param['laser'] = 'uwe'
    G = PGMCompiler(**param)
    G.header()
    assert G._instructions[9] == 'G359    ; WAIT MODE NOWAIT\n'


@pytest.mark.parametrize('v', [['V1'], ['V1', 'V2', 'V3'], [], 'VAR', ['V1', 'V2', ['V3', ['V4', 'V5']], 'V6']])
def test_dvar(param, v):
    G = PGMCompiler(**param)
    G.dvar(v)
    v = listcast(flatten(v))
    var_str = ' '.join(['${}'] * len(v)).format(*v)
    assert G._instructions[0] == f'DVAR {var_str}\n\n'
    assert G._dvars == [v.lower() for v in v]


@pytest.mark.parametrize(
    'mode, expectation',
    [
        (None, pytest.raises(ValueError)),
        ('ABS', does_not_raise()),
        ('abl', pytest.raises(ValueError)),
        ('Inc', does_not_raise()),
    ],
)
def test_mode(param, mode, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        assert G.mode(mode) is None


def test_mode_abs(param) -> None:
    G = PGMCompiler(**param)
    G.mode(mode='abs')
    assert G._instructions[-1] == 'G90 ; ABSOLUTE\n'
    assert G._mode_abs is True


def test_mode_inc(param) -> None:
    G = PGMCompiler(**param)
    G.mode(mode='inc')
    assert G._instructions[-1] == 'G91 ; INCREMENTAL\n'
    assert G._mode_abs is False


def test_comment(param) -> None:
    G = PGMCompiler(**param)
    c_str = 'this is a comment'
    G.comment(c_str)
    assert G._instructions[-1] == f'\n; {c_str}\n'

    G = PGMCompiler(**param)
    c_str = ''
    G.comment(c_str)
    assert G._instructions[-1] == '\n'

    G = PGMCompiler(**param)
    c_str = None
    G.comment(c_str)
    assert G._instructions[-1] == '\n'


@pytest.mark.parametrize(
    's, expectation',
    [
        (None, pytest.raises(ValueError)),
        ('on', does_not_raise()),
        ('OfF', does_not_raise()),
        ('onn', pytest.raises(ValueError)),
    ],
)
def test_shutter_inputs_error(param, s, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        assert G.shutter(s) is None


@pytest.mark.parametrize(
    's, expected',
    [
        (['on'], True),
        (['off'], False),
        (['off', 'off'], False),
        (['on', 'off'], False),
        (['off', 'on'], True),
        (['off', 'on', 'on'], True),
        (['on', 'on', 'off'], False),
        (['off', 'on', 'on', 'on', 'on'], True),
        (['off', 'on', 'on', 'on', 'off'], False),
    ],
)
def test_shutter_inputs(param, s, expected) -> None:
    G = PGMCompiler(**param)
    for state in s:
        G.shutter(state)
    assert G._shutter_on is expected


@pytest.mark.parametrize('t', [[0.0], [None], [10.0], [0.25], [5.0, 0.23, 0.5], [-1.23, 1.23]])
def test_dwell_inputs(param, t) -> None:
    G = PGMCompiler(**param)
    for dt in t:
        G.dwell(pause=dt)

    # remove None from t and compute abs value
    t = list(filter(None, t))
    assert G._total_dwell_time == np.sum(np.abs(t))


@pytest.mark.parametrize(
    'h_pos, expectation',
    [
        ([None, None, None], pytest.raises(ValueError)),
        ([None, 1, 2], does_not_raise()),
        ([None, None, 0.9], does_not_raise()),
        ([None, None], pytest.raises(ValueError)),
        ([1, None], pytest.raises(ValueError)),
        ([None, -0.32], pytest.raises(ValueError)),
        ([0.45, 182], pytest.raises(ValueError)),
        ([12, 45, 0.098123], does_not_raise()),
    ],
)
def test_set_home_error(param, h_pos, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        assert G.set_home(h_pos) is None


@pytest.mark.parametrize(
    'h_pos',
    [
        [15.0, 34.18, 0.54],
        [-1.23, None, 0.8234],
        [None, None, 0.8234],
    ],
)
def test_set_home_values(param, h_pos) -> None:
    G = PGMCompiler(**param)
    G.set_home(h_pos)

    x, y, z = h_pos
    args = ''
    if x is not None:
        args += f'X{x:.6f} '
    if y is not None:
        args += f'Y{y:.6f} '
    if z is not None:
        args += f'Z{z:.6f}'
    assert G._instructions[-1] == f'G92 {args}\n'


@pytest.mark.parametrize(
    'speedp, pos, speed, expectation',
    [
        (12, [1, 2, 3], 14, does_not_raise()),
        (13, [1, 2, 3], None, does_not_raise()),
        (5, [None, 1, 2], 6.7, does_not_raise()),
        (5, [1, 2], None, pytest.raises(ValueError)),
        (5, [None], 12.43, pytest.raises(ValueError)),
        (4, [], 1.78, pytest.raises(ValueError)),
        (5, [1, None, None], 19, does_not_raise()),
        (7, [None, None, None], 13, does_not_raise()),
    ],
)
def test_move_to_errors(param, speedp, pos, speed, expectation) -> None:
    param['speed_pos'] = speedp
    G = PGMCompiler(**param)
    with expectation:
        assert G.move_to(pos, speed) is None


def test_move_to_close_shutter(param) -> None:
    G = PGMCompiler(**param)
    G.shutter('on')
    assert G._shutter_on is True
    G.move_to([0, 0, 0])
    assert G._shutter_on is False


@pytest.mark.parametrize(
    'speedp, pos, speed',
    [
        (12, [1, 2, 3], 14),
        (13, [0, 0, 0], None),
        (13, [0, -5, None], None),
        (5, [None, 1, 2], 6.7),
        (5, [1, None, None], 19),
        (7, [None, None, None], 13),
    ],
)
def test_move_to_values(param, speedp, pos, speed) -> None:
    param['speed_pos'] = speedp
    G = PGMCompiler(**param)
    G.move_to(pos, speed)

    speed_pos = speedp if speed is None else speed
    x, y, z = pos
    args = ''
    if x is not None:
        args += f'X{x:.6f} '
    if y is not None:
        args += f'Y{y:.6f} '
    if z is not None:
        args += f'Z{z:.6f} '
    args += f'F{speed_pos:.6f}'

    assert G._instructions[-2] == f'G4 P{G.long_pause} ; DWELL\n'
    if all(coord is None for coord in pos):
        assert G._instructions[-3] == f'{args}\n'
    else:
        assert G._instructions[-3] == f'G1 {args}\n'
    assert G._shutter_on is False


def test_homing(param) -> None:
    G = PGMCompiler(**param)
    G.go_origin()
    assert G._instructions[-4] == '\n; HOMING\n'
    assert G._instructions[-3] == f'G1 X0.000000 Y0.000000 Z0.000000 F{G.speed_pos:.6f}\n'


def test_go_init(param) -> None:
    G = PGMCompiler(**param)
    G.go_init()
    assert G._instructions[-3] == f'G1 X-2.000000 Y0.000000 Z0.000000 F{G.speed_pos:.6f}\n'


def test_axis_rotation_contex_manager(param) -> None:
    param['aerotech_angle'] = 2.0
    G = PGMCompiler(**param)
    with G.axis_rotation():
        assert G._instructions[-6] == '\n; ACTIVATE AXIS ROTATION\n'
        assert G._instructions[-5] == f'G1 X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{G.speed_pos:.6f}\n'
        assert G._instructions[-4] == 'G84 X Y\n'
        assert G._instructions[-2] == f'G84 X Y F{G.aerotech_angle}\n\n'

        # do operations in G-Code compiler
        G.move_to([1, 2, 3])
        G.go_origin()

    print(G._instructions)
    assert G._instructions[-4] == '\n; DEACTIVATE AXIS ROTATION\n'
    assert G._instructions[-3] == f'G1 X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{G.speed_pos:.6f}\n'
    assert G._instructions[-2] == 'G84 X Y\n'


@pytest.mark.parametrize('angle_p, angle', [(None, 13), (14, -3), (9, None), (-1, -1), (None, None)])
def test_enter_axis_rotation(param, angle_p, angle) -> None:
    param['aerotech_angle'] = angle_p
    G = PGMCompiler(**param)
    G._enter_axis_rotation(angle)

    a = G.aerotech_angle if angle is None else float(angle % 360)
    if a == 0.0:
        assert not G._instructions
    else:
        assert G._instructions[-6] == '\n; ACTIVATE AXIS ROTATION\n'
        assert G._instructions[-5] == f'G1 X{0.0:.6f} Y{0.0:.6f} Z{0.0:.6f} F{G.speed_pos:.6f}\n'
        assert G._instructions[-4] == 'G84 X Y\n'
        assert G._instructions[-2] == f'G84 X Y F{a}\n\n'


@pytest.mark.parametrize(
    'n, dvar, var, expectation',
    [
        (13, ['PIPPO'], 'PIPPO', does_not_raise()),
        (0, ['PIPPO'], 'PIPPO', pytest.raises(ValueError)),
        (-5, ['PIPPO'], 'PIPPO', pytest.raises(ValueError)),
        (-3.2, ['PIPPO'], 'PIPPO', pytest.raises(ValueError)),
        (3.2, ['PIPPO'], 'PIPPO', does_not_raise()),
        (6, ['pippo'], 'paperino', pytest.raises(ValueError)),
        (13, ['paperino'], 'PAPERINO', does_not_raise()),
        (13, ['paperino', 'pippo', 'pluto'], 'pluto', does_not_raise()),
        (6, ['pippo', 'pippi', 'pupo'], 'p', pytest.raises(ValueError)),
    ],
)
def test_for_loop_errors(param, n, dvar, var, expectation) -> None:
    G = PGMCompiler(**param)
    G.dvar(dvar)
    with expectation:
        assert G.for_loop(var, n).__enter__() is not None


@pytest.mark.parametrize('n, v', [(1, 'VAR'), (5, 'VAR'), (100.2, 'VAR'), (2, 'VAR')])
def test_for_loop(param, n, v) -> None:
    p = 1
    G = PGMCompiler(**param)
    G.dvar([v])
    with G.for_loop(v, n):
        assert G._instructions[-1] == f'FOR ${v} = 0 TO {int(n) - 1}\n'
        # do G-Code operations
        G.dwell(p)
    assert G._instructions[-1] == f'NEXT ${v}\n'
    assert G.dwell_time == int(n) * p


@pytest.mark.parametrize(
    'n, expectation',
    [
        (13, does_not_raise()),
        (0, pytest.raises(ValueError)),
        (-5, pytest.raises(ValueError)),
        (-3.2, pytest.raises(ValueError)),
        (3.2, does_not_raise()),
    ],
)
def test_repeat_errors(param, n, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        assert G.repeat(n).__enter__() is not None


@pytest.mark.parametrize('n', [1, 3, 6, 7, 99])
def test_repeat(param, n) -> None:
    p = 1
    G = PGMCompiler(**param)
    with G.repeat(n):
        assert G._instructions[-1] == f'REPEAT {int(n)}\n'
        # do G-Code operations
        G.dwell(p)
    assert G._instructions[-1] == 'ENDREPEAT\n'
    assert G.dwell_time == int(n) * p


def test_tic(param) -> None:
    G = PGMCompiler(**param)
    G.tic()
    assert G._instructions[-1] == 'MSGDISPLAY 1, "START #TS"\n\n'


def test_toc(param) -> None:
    G = PGMCompiler(**param)
    G.toc()
    assert G._instructions[-3] == 'MSGDISPLAY 1, "END   #TS"\n'
    assert G._instructions[-2] == 'MSGDISPLAY 1, "---------------------"\n'
    assert G._instructions[-1] == 'MSGDISPLAY 1, " "\n\n'


def test_instruction(param) -> None:
    G = PGMCompiler(**param)
    G.instruction('INSTRUCTION_1\n')
    assert G._instructions[-1] == 'INSTRUCTION_1\n'
    G.instruction('INSTRUCTION_2')
    assert G._instructions[-1] == 'INSTRUCTION_2\n'


@pytest.mark.parametrize(
    'fn, fp, ext, res',
    [
        ('test', None, None, 'test'),
        ('test', 'femto/test_dir', None, 'femto/test_dir/test'),
        ('test.pgm', None, 'pgm', 'test.pgm'),
        ('test.pgm', None, '.pgm', 'test.pgm'),
        ('test.pgm', None, '.PGM', 'test.pgm'),
        ('test.pgm', None, '.PgM', 'test.pgm'),
        ('test.pgm', 'femto/test_dir', '.pgm', 'femto/test_dir/test.pgm'),
    ],
)
def test_get_filepath_values(param, fn, fp, ext, res) -> None:
    G = PGMCompiler(**param)
    if fp is None:
        file = Path(fn)
    else:
        file = Path(fp) / fn
    assert G._get_filepath(fn, fp, ext) == file


@pytest.mark.parametrize(
    'fn, fp, ext, expectation',
    [
        ('test', None, None, does_not_raise()),
        ('test', 'femto/test_dir', None, does_not_raise()),
        ('test.pgm', None, 'pgm', does_not_raise()),
        ('test.mp3', None, '.pgm', pytest.raises(ValueError)),
        ('test.pgm', None, '.wav', pytest.raises(ValueError)),
        ('test.pgm', None, '.TXT', pytest.raises(ValueError)),
    ],
)
def test_get_filepath_extensions(param, fn, fp, ext, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        assert G._get_filepath(fn, fp, ext) is not None


@pytest.mark.parametrize('fn, i', [('test.pgm', 0), ('test.pgm', 1), ('test.pgm', -5), ('mzi.pgm', 3)])
def test_load_program(param, fn, i) -> None:
    G = PGMCompiler(**param)
    G.load_program(fn, i)

    f = Path(fn)
    assert G._instructions[-1] == f'PROGRAM {abs(i)} LOAD "{f}"\n'
    assert f.stem in G._loaded_files


def test_programstop(param) -> None:
    G = PGMCompiler(**param)
    G.programstop()
    assert G._instructions[-2] == 'PROGRAM 2 STOP\n'
    assert G._instructions[-1] == 'WAIT (TASKSTATUS(2, DATAITEM_TaskState) == TASKSTATE_Idle) -1\n'

    G = PGMCompiler(**param)
    G.programstop(task_id=3)
    assert G._instructions[-2] == 'PROGRAM 3 STOP\n'
    assert G._instructions[-1] == 'WAIT (TASKSTATUS(3, DATAITEM_TaskState) == TASKSTATE_Idle) -1\n'

def test_wait_default_time(param) -> None:
    G = PGMCompiler(**param)
    G.wait(condition='ciao')
    assert G._instructions[-1] == 'WAIT (ciao) -1\n'


@pytest.mark.parametrize('t', [0,1,2,3,4,69, 99, 420])
def test_wait_time(param, t) -> None:
    G = PGMCompiler(**param)
    G.wait(condition='ciao', time=t)
    if t == 0:
        assert G._instructions[-1] == 'WAIT (ciao) -1\n'
    else:
        assert G._instructions[-1] == f'WAIT (ciao) {t}\n'


@pytest.mark.parametrize(
    'lp, rp, expectation',
    [
        ('test.pgm', 'test.pgm', does_not_raise()),
        ('test.pgm', 'femto/test.pgm', does_not_raise()),
        ('test.pgm', 'test', pytest.raises(ValueError)),
        ('test.pgm', 'tttest.pgm', pytest.raises(FileNotFoundError)),
    ],
)
def test_remove_program_raise(param, lp, rp, expectation) -> None:
    G = PGMCompiler(**param)
    G.load_program(lp)
    with expectation:
        assert G.remove_program(rp) is None


@pytest.mark.parametrize(
    'lp, cp, expectation',
    [
        ('test.pgm', 'test.pgm', does_not_raise()),
        ('test.pgm', 'femto/test.pgm', does_not_raise()),
        ('test.pgm', 'test', pytest.raises(ValueError)),
        ('test.pgm', 'tttest.pgm', pytest.raises(FileNotFoundError)),
    ],
)
def test_farcall_raise(param, lp, cp, expectation) -> None:
    G = PGMCompiler(**param)
    G.load_program(lp)
    with expectation:
        G.farcall(cp) is None


def test_farcall_value(param) -> None:
    fn = 'test.pgm'
    G = PGMCompiler(**param)
    G.load_program(fn)
    G.farcall(fn)
    assert G._instructions[-1] == f'FARCALL "{fn}"\n'


@pytest.mark.parametrize(
    'lp, cp, expectation',
    [
        ('test.pgm', 'test.pgm', does_not_raise()),
        ('test.pgm', 'femto/test.pgm', does_not_raise()),
        ('test.pgm', 'test', pytest.raises(ValueError)),
        ('test.pgm', 'tttest.pgm', pytest.raises(FileNotFoundError)),
    ],
)
def test_bufferedcall_raise(param, lp, cp, expectation) -> None:
    G = PGMCompiler(**param)
    G.load_program(lp)
    with expectation:
        G.bufferedcall(cp) is None


def test_bufferedcall_value(param) -> None:
    fn = 'test.pgm'
    G = PGMCompiler(**param)
    G.load_program(fn)
    G.bufferedcall(fn)
    assert G._instructions[-1] == f'PROGRAM 2 BUFFEREDRUN "{fn}"\n'


@pytest.mark.parametrize(
    'fns, tid, id_exp, expectation',
    [
        (['f1.pgm', 'f2.pgm', 'f3.pgm'], [1, 2, 3], [1, 2, 3], does_not_raise()),
        (['f1.pgm', 'f2.pgm', 'f3.pgm'], [1, 2, 3, 4, 5], [1, 2, 3], does_not_raise()),
        (['f1.pgm', 'f2.pgm', 'f3.pgm'], [1], [1, 2, 2], does_not_raise()),
        (['f1.pgm', 'f2.pgm', 'f3.pgm'], [], [2, 2, 2], does_not_raise()),
        ([], [1, 2, 3], [], does_not_raise()),
    ],
)
def test_farcall_list(param, fns, tid, id_exp, expectation) -> None:
    G = PGMCompiler(**param)
    with expectation:
        G.farcall_list(fns, tid) is None

    T = PGMCompiler(**param)
    for fpath, t_id in zip(fns, id_exp):
        fpath = Path(fpath)
        T.load_program(str(fpath), t_id)
        T.farcall(fpath.name)
        T.dwell(T.short_pause)
        T.remove_program(fpath.name, t_id)
        T.dwell(T.short_pause)
        T.instruction('\n\n')

    assert G._instructions == T._instructions


@pytest.mark.parametrize(
    'x, y, fp_x, fp_y, exp_x, exp_y',
    [
        (
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
            False,
            False,
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
        ),
        (
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
            True,
            False,
            np.array([2, -5, -10, -14, -23, -35]),
            np.array([12, 12, 5, 6, 17, 15]),
        ),
        (
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
            False,
            True,
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([-12, -12, -5, -6, -17, -15]),
        ),
        (
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
            True,
            True,
            np.array([2, -5, -10, -14, -23, -35]),
            np.array([-12, -12, -5, -6, -17, -15]),
        ),
    ],
)
def test_flip_path(param, x, y, fp_x, fp_y, exp_x, exp_y) -> None:
    param['flip_x'] = fp_x
    param['flip_y'] = fp_y
    G = PGMCompiler(**param)
    xf, yf = G.flip(x, y)
    np.testing.assert_almost_equal(xf, exp_x)
    np.testing.assert_almost_equal(yf, exp_y)


@pytest.mark.parametrize(
    'x, y, z',
    [
        (
            np.array([-2, 5, 10, 14, 23, 35]),
            np.array([12, 12, 5, 6, 17, 15]),
            np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002]),
        ),
        (
            np.array([-2, 27]),
            np.array([10, 10]),
            np.array([-0.002, -0.002]),
        ),
        (
            np.array([-2, 27, 27, -2]),
            np.array([10, 10, 10, 1]),
            np.array([-0.002, -0.002, -0.002, -0.002]),
        ),
        (
            np.array([-2]),
            np.array([10]),
            np.array([-0.002]),
        ),
    ],
)
def test_compensate(param, x, y, z) -> None:
    from pathlib import Path

    def fun(h):
        x, y = h.T
        return (x**2 * y) * 1e-3

    funpath = Path.cwd() / 'fwarp.pickle'
    with open(funpath, 'wb') as f:
        dill.dump(fun, f)

    param['warp_flag'] = True
    param['flip_x'] = False

    G = PGMCompiler(**param)
    xc, yc, zc = G.compensate(x, y, z)
    zfun = z + np.array([fun(np.array([xp, yp]).T) for (xp, yp) in zip(x, y)])

    np.testing.assert_array_equal(xc, x)
    np.testing.assert_array_equal(yc, y)
    np.testing.assert_almost_equal(zc, zfun)

    funpath.unlink()


@pytest.mark.parametrize(
    'angle, res',
    [
        (
            np.pi / 4,
            np.array(
                [[1 / np.sqrt(2), -1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1.33 / 1.5]],
            ),
        ),
        (0, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.33 / 1.5]])),
        (np.pi / 3, np.array([[0.5, -0.5 * np.sqrt(3), 0], [0.5 * np.sqrt(3), 0.5, 0], [0, 0, 1.33 / 1.5]])),
    ],
)
def test_t_matrix_matrix(param, angle, res) -> None:
    M = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.33 / 1.5]]).T

    param['rotation_angle'] = math.degrees(angle)
    G = PGMCompiler(**param)
    np.testing.assert_almost_equal(G.t_matrix, M)


@pytest.mark.parametrize(
    'xflip, yflip, warpf, angle, xin, yin, zin, xo, yo, zo',
    [
        (
            True,
            False,
            True,
            0.0,
            np.array([-2, 10, 10, 28]),
            np.array([5, 5, 16, 16]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([2, -10, -10, -28]),
            np.array([5, 5, 16, 16]),
            np.array([-0.0022167, 0.0084233, 0.007448, 0.023408]),
        ),
        (
            False,
            True,
            True,
            1.0,
            np.array([-2, 10, 13, 34]),
            np.array([5, 5, 8, 16]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([-1.9124334, 10.085739, 13.1376393, 34.2740601]),
            np.array([-5.0341433, -4.8247144, -7.7719003, -15.4041813]),
            np.array([-0.0022167, 0.0084233, 0.0108173, 0.028728]),
        ),
    ],
)
def test_transform_points_(param, xflip, yflip, warpf, angle, xin, yin, zin, xo, yo, zo) -> None:
    def fun(h):
        a = np.array([[1e-3], [-1e-4]]).T
        return np.matmul(a, h.T).flatten()

    file = Path.cwd() / 'fwarp.pickle'
    if not file.is_file():
        with open(file, 'wb') as f:
            dill.dump(fun, f)

    param['flip_x'] = xflip
    param['flip_y'] = yflip
    param['warp_flag'] = warpf
    param['rotation_angle'] = angle  # in degree
    G = PGMCompiler(**param)
    x, y, z = G.transform_points(xin, yin, zin)

    np.testing.assert_almost_equal(x, xo)
    np.testing.assert_almost_equal(y, yo)
    np.testing.assert_almost_equal(z, zo)

    file.unlink()


@pytest.mark.parametrize(
    'x, y, z, f, expected',
    [
        (1, 2, 3, 4, 'X1.000000 Y2.000000 Z3.000000 F4.000000'),
        (None, 2, 3, 4, 'Y2.000000 Z3.000000 F4.000000'),
        (None, None, 3, 4, 'Z3.000000 F4.000000'),
        (None, None, None, 4, 'F4.000000'),
        (None, None, None, None, ''),
        (5.8543, 2.0000001, -3.9456127, 4, 'X5.854300 Y2.000000 Z-3.945613 F4.000000'),
        (None, 2, 3, None, 'Y2.000000 Z3.000000'),
    ],
)
def test_format_arguments(param, x, y, z, f, expected) -> None:
    G = PGMCompiler(**param)

    assert G._format_args(x, y, z, f) == expected


@pytest.mark.parametrize(
    'x, y, z, f, expectation',
    [
        (1, 2, 3, 4, does_not_raise()),
        (None, 2, 3, 4, does_not_raise()),
        (None, None, 3, 4, does_not_raise()),
        (None, None, None, 4, does_not_raise()),
        (None, None, None, None, does_not_raise()),
        (5.8543, 2.0000001, -3.9456127, 4, does_not_raise()),
        (None, 2, 3, None, does_not_raise()),
        (None, 2, 3, -1, pytest.raises(ValueError)),
        (None, 2, 3, 0.0000000001, pytest.raises(ValueError)),
        (None, 2, 3, 1e-9, pytest.raises(ValueError)),
    ],
)
def test_format_arguments_raise(param, x, y, z, f, expectation) -> None:
    G = PGMCompiler(**param)

    with expectation:
        assert G._format_args(x, y, z, f) is not None


@pytest.mark.parametrize(
    'pts, expected',
    [
        (
            [
                [
                    -2.0,
                    -2.0,
                    1.0,
                    4.0,
                    7.0,
                    7.0,
                    -2.0,
                ],
                [
                    0.040,
                    0.040,
                    0.040,
                    0.040,
                    3.040,
                    3.040,
                    0.040,
                ],
                [
                    0.0350,
                    0.0350,
                    0.0350,
                    3.0350,
                    6.0350,
                    6.0350,
                    0.0350,
                ],
                [
                    0.50,
                    0.50,
                    20.0,
                    20.0,
                    20.0,
                    20.0,
                    5.0,
                ],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ],
            deque(
                [
                    'G1 X1.998997 Y0.074899 Z0.031033 F0.500000\n',
                    '\n',
                    'G4 P0.025 ; DWELL\n',
                    'PSOCONTROL X ON\n',
                    'G4 P1.0 ; DWELL\n',
                    '\n',
                    'G1 X-1.000546 Y0.022542 F20.000000\n',
                    'G1 X-4.000089 Y-0.029816 Z2.691033\n',
                    'G1 X-7.051989 Y2.917370 Z5.351033\n',
                    '\n',
                    'G4 P0.025 ; DWELL\n',
                    'PSOCONTROL X OFF\n',
                    'G4 P1.0 ; DWELL\n',
                    '\n',
                    'G1 X1.998997 Y0.074899 Z0.031033 F5.000000\n',
                    'G4 P1.0 ; DWELL\n',
                    '\n',
                ]
            ),
        )
    ],
)
def test_write(param, pts, expected) -> None:
    G = PGMCompiler(**param)

    G.write(np.array(pts))
    assert G._instructions == expected


@pytest.mark.parametrize(
    'pts, exp',
    [
        (np.array([]), 0),
        (np.random.rand(67, 2).T, 0),
        (np.random.rand(69, 3).T, 0),
        (np.random.rand(67, 5).T, 69),  # +2 for final pause and '\n' instruction
        (np.random.rand(167, 5).T, 169),  # +2 for final pause and '\n' instruction
        (np.random.rand(10, 6).T, 0),
        (np.random.rand(10, 11).T, 0),
    ],
)
def test_write_raise(param, pts, exp) -> None:
    G = PGMCompiler(**param)
    G.write(pts)
    assert len(G._instructions) == exp


@pytest.mark.parametrize(
    'list_instr, expected',
    [
        (
            [
                'G01 F255\n',
                'G9 G1 X0.01 F20\n',
                'G9 G1 F20\n',
                'G01 F255\n',
                'G1 F12\n',
                'G01 F 0.88\n',
                'G09 G01 X1 F 12\n',
                'G09G01 F81\n',
            ],
            [
                'F255\n',
                'G9 G1 X0.01 F20\n',
                'F20\n',
                'F255\n',
                'F12\n',
                'F 0.88\n',
                'G09 G01 X1 F 12\n',
                'F81\n',
            ],
        )
    ],
)
def test_re_filtering(param, list_instr, expected) -> None:
    G = PGMCompiler(**param)
    G._instructions.extend(list_instr)
    G.close()

    file = Path(param['export_dir']) / param['filename']
    with open(file) as f:
        exp_instr = f.readlines()

    assert exp_instr == expected

    dire = Path(param['export_dir'])
    file.unlink()
    dire.rmdir()


def test_close_dir(param) -> None:
    exp_dir = './src/femto/test/'
    param['export_dir'] = exp_dir
    param['verbose'] = True

    # add some istructions
    G = PGMCompiler(**param)
    G.header()
    with G.repeat(19):
        G.dwell(3)
        G.move_to([0, 0, 0])

    # G is non-empty
    assert G._instructions != deque([])

    # close and write to file
    G.close()
    assert G._instructions == deque([])

    # assert the exp directory has been created
    dire = Path(exp_dir)
    assert dire.is_dir()
    # assert the file exists
    file = dire / param['filename']
    assert file.is_file()

    file.unlink()
    dire.rmdir()


def test_pgm_farcall_empty_external_files(param) -> None:
    dir = Path('./dir/')
    dir.mkdir(parents=True, exist_ok=True)
    farcall_file = dir / 'FARCALL.pgm'
    assert dir.is_dir()
    farcall(directory=dir, parameters=param)
    assert not farcall_file.is_file()
    dir.rmdir()


def test_pgm_farcall_external_files(param) -> None:
    dir = Path('./dir/')
    dir.mkdir(parents=True, exist_ok=True)
    farcall_file = dir / 'FARCALL.pgm'

    with PGMCompiler.from_dict(param, filename='test1', export_dir='dir') as G:
        G.write(np.random.rand(90, 3))
    with PGMCompiler.from_dict(param, filename='test2', export_dir='dir') as G:
        G.write(np.random.rand(90, 3))
    with PGMCompiler.from_dict(param, filename='test3', export_dir='dir') as G:
        G.write(np.random.rand(90, 3))

    assert dir.is_dir()
    farcall(directory=dir, parameters=param)
    assert farcall_file.is_file()
    Path('./dir/test1.pgm').unlink()
    Path('./dir/test2.pgm').unlink()
    Path('./dir/test3.pgm').unlink()
    Path('./dir/FARCALL.pgm').unlink()
    dir.rmdir()
