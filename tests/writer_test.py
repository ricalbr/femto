from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from pathlib import Path

import numpy as np
import pytest
from femto.helpers import dotdict
from femto.helpers import flatten
from femto.helpers import listcast
from femto.marker import Marker
from femto.trench import TrenchColumn
from femto.trench import UTrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import NasuWriter
from femto.writer import TrenchWriter
from femto.writer import UTrenchWriter
from femto.writer import WaveguideWriter
from femto.writer import Writer


@pytest.fixture
def gc_param() -> dict:
    p = dict(
        filename='testCell.pgm',
        n_glass=1.5,
        n_environment=1.33,
        laser='PHAROS',
        shift_origin=(0.5, 0.5),
        samplesize=(25, 1),
    )
    return p


@pytest.fixture
def list_wg() -> list[Waveguide]:
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [Waveguide(**PARAM_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.linear([5, 0, 0])
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.linear([5, 0, 0])
        wg.end()
    return coup


@pytest.fixture
def list_ng() -> list[NasuWaveguide]:
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [NasuWaveguide(**PARAM_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.linear([5, 0, 0])
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.sin_coupler((-1) ** i * wg.dy_bend)
        wg.linear([5, 0, 0])
        wg.end()
    return coup


@pytest.fixture
def list_mk() -> list[Marker]:
    PARAM_MK = dotdict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    markers = []
    for x, y in zip(range(4, 8), range(3, 7)):
        m = Marker(**PARAM_MK)
        m.cross([x, y])
        markers.append(m)
    return markers


@pytest.fixture
def list_tcol(list_wg) -> list[TrenchColumn]:
    PARAM_TC = dotdict(length=1.0, base_folder='', y_min=-0.1, y_max=19 * 0.08 + 0.1, u=[30.339, 32.825])
    x_c = [3, 7.5, 10.5]
    t_col = []
    for x in x_c:
        T = TrenchColumn(x_center=x, **PARAM_TC)
        T.dig_from_waveguide(flatten([list_wg]), remove=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        t_col.append(T)
    return t_col


@pytest.fixture
def list_utcol(list_wg) -> list[UTrenchColumn]:
    PARAM_TC = dotdict(length=1.0, base_folder='', n_pillars=1, y_min=-0.1, y_max=19 * 0.08 + 0.1, u=[30.339, 32.825])
    x_c = [3, 7.5, 10.5]
    ut_col = []
    for x in x_c:
        T = UTrenchColumn(x_center=x, **PARAM_TC)
        T.dig_from_waveguide(flatten([list_wg]), remove=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ut_col.append(T)
    return ut_col


def test_writer(gc_param) -> None:
    from abc import ABCMeta

    Writer.__abstractmethods__ = set()

    class Dummy(Writer):
        pass

    d = Dummy(**gc_param)
    append = d.append(None)
    extend = d.extend([None])
    plot2d = d.plot2d(None)
    plot3d = d.plot3d()
    pgm = d.pgm()
    assert isinstance(Writer, ABCMeta)
    assert append is None
    assert extend is None
    assert plot2d is None
    assert plot3d is None
    assert pgm is None


def test_trench_writer_init(gc_param, list_tcol) -> None:
    twr = TrenchWriter(list_tcol, **gc_param)
    dirname = 'TRENCH'
    expp = Path.cwd() / dirname

    tcs = []
    for col in list_tcol:
        for tr in col:
            tcs.append(tr)

    assert twr.obj_list == list_tcol
    assert twr.trenches == tcs
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp

    del twr

    dirname = 'test'
    twr = TrenchWriter([], dirname=dirname, **gc_param)
    expp = Path.cwd() / dirname

    assert twr.obj_list == []
    assert twr.trenches == []
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp


def test_trench_writer_append(gc_param, list_tcol) -> None:
    twr = TrenchWriter([], **gc_param)
    for col in list_tcol:
        twr.append(col)
    assert twr.obj_list == list_tcol
    assert twr.trenches == flatten([tr for col in listcast(list_tcol) for tr in col])


def test_trench_writer_append_raise(gc_param, list_tcol) -> None:
    twr = TrenchWriter([], **gc_param)
    with pytest.raises(TypeError):
        twr.append(list_tcol)


def test_trench_writer_extend(gc_param, list_tcol) -> None:
    twr = TrenchWriter([], **gc_param)
    twr.extend(list_tcol)
    assert twr.obj_list == list_tcol
    assert twr.trenches == flatten([tr for col in listcast(list_tcol) for tr in col])
    del twr

    twr = TrenchWriter([], **gc_param)
    new_list = [[[list_tcol]]]
    twr.extend(new_list)
    assert twr.obj_list == list_tcol
    assert twr.trenches == flatten([tr for col in listcast(list_tcol) for tr in col])
    del twr

    twr = TrenchWriter([], **gc_param)
    new_list = [[[list_tcol, list_tcol], list_tcol], list_tcol]
    twr.extend(new_list)
    assert twr.obj_list == flatten([list_tcol, list_tcol, list_tcol, list_tcol])
    assert twr.trenches == flatten([tr for col in flatten([list_tcol, list_tcol, list_tcol, list_tcol]) for tr in col])
    del twr


def test_trench_writer_extend_raise(gc_param, list_tcol) -> None:
    twr = TrenchWriter([], **gc_param)
    with pytest.raises(TypeError):
        l_t_col = (list_tcol, TrenchColumn(1, 2, 3))
        twr.extend(l_t_col)


def test_trench_writer_plot2d(gc_param, list_tcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    twr = TrenchWriter(list_tcol, **gc_param)
    assert twr.plot2d(fig=fig) is not None
    del twr

    twr = TrenchWriter(list_tcol, **gc_param)
    assert twr.plot2d() is not None
    del twr


def test_trench_writer_plot3d(gc_param, list_tcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()
    twr = TrenchWriter(list_tcol, **gc_param)
    assert twr.plot3d(fig=fig) is not None
    del twr, fig

    twr = TrenchWriter(list_tcol, **gc_param)
    assert twr.plot3d() is not None
    del twr


def test_trench_writer_pgm_empty(gc_param) -> None:
    twr = TrenchWriter(tc_list=[], **gc_param)
    assert twr.pgm() is None
    assert not twr._export_path.is_dir()


def test_trench_writer_pgm(gc_param, list_tcol) -> None:
    twr = TrenchWriter(tc_list=list_tcol, **gc_param)
    twr.pgm()

    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_tcol):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()
        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').unlink()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
            (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').unlink()
        (twr._export_path / f'trenchCol{i_col + 1:03}').rmdir()
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert (twr._export_path / 'MAIN.pgm').is_file()
    (twr._export_path / 'MAIN.pgm').unlink()
    twr._export_path.rmdir()


def test_trench_writer_export_array2d_raise(gc_param, list_tcol) -> None:
    twr = TrenchWriter(tc_list=list_tcol, **gc_param)
    with pytest.raises(ValueError):
        twr.export_array2d(filename=None, x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), speed=4)


def test_utrench_writer_init(gc_param, list_utcol) -> None:
    twr = UTrenchWriter(list_utcol, **gc_param)
    dirname = 'U-TRENCH'
    expp = Path.cwd() / dirname

    tcs = []
    bds = []
    for col in list_utcol:
        for tr in col:
            tcs.append(tr)
        bds.extend(col.trenchbed)

    assert twr.obj_list == list_utcol
    assert twr.trenches == tcs
    assert twr.beds == bds
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp

    del twr

    dirname = 'test'
    twr = UTrenchWriter([], dirname=dirname, **gc_param)
    expp = Path.cwd() / dirname

    assert twr.obj_list == []
    assert twr.trenches == []
    assert twr.beds == []
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp


def test_utrench_writer_append(gc_param, list_utcol) -> None:
    twr = UTrenchWriter([], **gc_param)
    for col in list_utcol:
        twr.append(col)
    assert twr.obj_list == list_utcol
    assert twr.trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr.beds == flatten([bd for col in listcast(list_utcol) for bd in col.trenchbed])


def test_utrench_writer_append_raise(gc_param, list_utcol) -> None:
    twr = UTrenchWriter([], **gc_param)
    with pytest.raises(TypeError):
        twr.append(list_utcol)


def test_utrench_writer_extend(gc_param, list_utcol) -> None:
    twr = UTrenchWriter([], **gc_param)
    twr.extend(list_utcol)
    assert twr.obj_list == list_utcol
    assert twr.trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr.beds == flatten([bd for col in listcast(list_utcol) for bd in col.trenchbed])
    del twr

    twr = UTrenchWriter([], **gc_param)
    new_list = [[[list_utcol]]]
    twr.extend(new_list)
    assert twr.obj_list == list_utcol
    assert twr.trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr.beds == flatten([bd for col in listcast(list_utcol) for bd in col.trenchbed])
    del twr

    twr = UTrenchWriter([], **gc_param)
    new_list = [[[list_utcol, list_utcol], list_utcol], list_utcol]
    twr.extend(new_list)
    assert twr.obj_list == flatten([list_utcol, list_utcol, list_utcol, list_utcol])
    assert twr.trenches == flatten(
        [tr for col in flatten([list_utcol, list_utcol, list_utcol, list_utcol]) for tr in col]
    )
    assert twr.beds == flatten(
        [bd for col in flatten([list_utcol, list_utcol, list_utcol, list_utcol]) for bd in col.trenchbed]
    )
    del twr


def test_utrench_writer_extend_raise(gc_param, list_utcol) -> None:
    twr = UTrenchWriter([], **gc_param)
    with pytest.raises(TypeError):
        l_ut_col = (list_utcol, UTrenchColumn(1, 2, 3))
        twr.extend(l_ut_col)


def test_utrench_writer_plot2d(gc_param, list_utcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    twr = UTrenchWriter(list_utcol, **gc_param)
    assert twr.plot2d(fig=fig) is not None
    del twr

    twr = UTrenchWriter(list_utcol, **gc_param)
    assert twr.plot2d() is not None
    del twr


def test_utrench_writer_plot3d(gc_param, list_utcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()
    twr = UTrenchWriter(list_utcol, **gc_param)
    assert twr.plot3d(fig=fig) is not None
    del twr, fig

    twr = UTrenchWriter(list_utcol, **gc_param)
    assert twr.plot3d() is not None
    del twr


def test_utrench_writer_pgm_empty(gc_param) -> None:
    twr = UTrenchWriter(utc_list=[], **gc_param)
    assert twr.pgm() is None
    assert not twr._export_path.is_dir()


def test_utrench_writer_pgm(gc_param, list_utcol) -> None:
    twr = UTrenchWriter(utc_list=list_utcol, **gc_param)
    twr.pgm()

    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_utcol):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()

        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').unlink()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
            (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').unlink()

        for i_bed, _ in enumerate(col.trenchbed):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench_BED_{i_bed + 1:03}.pgm').is_file()
            (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench_BED_{i_bed + 1:03}.pgm').unlink()

        (twr._export_path / f'trenchCol{i_col + 1:03}').rmdir()
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert (twr._export_path / 'MAIN.pgm').is_file()
    (twr._export_path / 'MAIN.pgm').unlink()
    twr._export_path.rmdir()


def test_utrench_writer_export_array2d_raise(gc_param, list_utcol) -> None:
    twr = UTrenchWriter(utc_list=list_utcol, **gc_param)
    with pytest.raises(ValueError):
        twr.export_array2d(filename=None, x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), speed=4)


def test_waveguide_writer_init(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(list_wg, **gc_param)
    expp = Path.cwd()

    assert wwr.obj_list == list_wg

    assert wwr._param == gc_param
    assert wwr._export_path == expp

    del wwr

    dirname = 'test'
    wwr = WaveguideWriter([], export_dir=dirname, **gc_param)
    expp = Path.cwd() / dirname

    assert wwr.obj_list == []

    assert wwr._param == dict(export_dir=dirname, **gc_param)
    assert wwr._export_path == expp


def test_waveguide_writer_append(gc_param, list_wg) -> None:
    wwr = WaveguideWriter([], **gc_param)
    for wg in list_wg:
        wwr.append(wg)
    assert wwr.obj_list == list_wg


def test_waveguide_writer_append_raise(gc_param, list_wg) -> None:
    wwr = WaveguideWriter([], **gc_param)
    with pytest.raises(TypeError):
        wwr.append(list_wg)


def test_waveguide_writer_extend(gc_param, list_wg) -> None:
    wwr = WaveguideWriter([], **gc_param)
    wwr.extend(list_wg)
    assert wwr.obj_list == list_wg
    del wwr

    wwr = WaveguideWriter([], **gc_param)
    new_list = [list_wg, list_wg, list_wg[0]]
    wwr.extend(new_list)
    assert wwr.obj_list == new_list


@pytest.mark.parametrize(
    'l_wg, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([Waveguide()], does_not_raise()),
        ([Waveguide(), [Waveguide(), Waveguide()]], does_not_raise()),
        (
            [[[Waveguide()]], [[Waveguide(), [Waveguide(), [Waveguide(), Waveguide()]]], Waveguide()]],
            pytest.raises(ValueError),
        ),
        ([TrenchColumn(1, 2, 3), [Waveguide(), Waveguide()]], pytest.raises(TypeError)),
    ],
)
def test_waveguide_writer_extend_raise(gc_param, l_wg, expectation) -> None:
    wwr = WaveguideWriter([], **gc_param)
    with expectation:
        wwr.extend(l_wg)


def test_waveguide_writer_plot2d(gc_param, list_wg) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = WaveguideWriter(list_wg, **gc_param)
    assert wwr.plot2d(fig=fig) is not None
    del wwr

    wwr = WaveguideWriter(list_wg, **gc_param)
    assert wwr.plot2d() is not None
    del wwr


def test_waveguide_writer_plot3d(gc_param, list_wg) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = WaveguideWriter(list_wg, **gc_param)
    assert wwr.plot3d(fig=fig) is not None
    del wwr

    wwr = WaveguideWriter(list_wg, **gc_param)
    assert wwr.plot3d() is not None
    del wwr


def test_waveguide_writer_pgm(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(list_wg, **gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_WG.pgm'

    assert wwr._export_path.is_dir()
    assert (wwr._export_path / fp).is_file()
    (wwr._export_path / fp).unlink()


def test_waveguide_writer_pgm_empty(gc_param) -> None:
    wwr = WaveguideWriter([], **gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_WG.pgm'

    assert wwr._export_path.is_dir()
    assert not (wwr._export_path / fp).is_file()


def test_nasu_writer_init(gc_param, list_ng) -> None:
    wwr = NasuWriter(list_ng, **gc_param)
    expp = Path.cwd()

    assert wwr.obj_list == list_ng

    assert wwr._param == gc_param
    assert wwr._export_path == expp

    del wwr

    dirname = 'test'
    wwr = NasuWriter([], export_dir=dirname, **gc_param)
    expp = Path.cwd() / dirname

    assert wwr.obj_list == []

    assert wwr._param == dict(export_dir=dirname, **gc_param)
    assert wwr._export_path == expp


def test_nasu_writer_append(gc_param, list_ng) -> None:
    wwr = NasuWriter([], **gc_param)
    for wg in list_ng:
        wwr.append(wg)
    assert wwr.obj_list == list_ng


def test_nasu_writer_append_raise(gc_param, list_ng) -> None:
    wwr = NasuWriter([], **gc_param)
    with pytest.raises(TypeError):
        wwr.append(list_ng)


def test_nasu_writer_extend(gc_param, list_ng) -> None:
    wwr = NasuWriter([], **gc_param)
    wwr.extend(list_ng)
    assert wwr.obj_list == list_ng
    del wwr

    wwr = NasuWriter([], **gc_param)
    new_list = [list_ng, list_ng, list_ng[0]]
    wwr.extend(new_list)
    assert wwr.obj_list == new_list


@pytest.mark.parametrize(
    'l_nw, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([NasuWaveguide()], does_not_raise()),
        (NasuWaveguide(), pytest.raises(TypeError)),
        ([NasuWaveguide(), [NasuWaveguide(), NasuWaveguide()]], does_not_raise()),
        ([[[NasuWaveguide()]], [[NasuWaveguide(), [NasuWaveguide(), [NasuWaveguide()]]]]], does_not_raise()),
        ([Waveguide(), [NasuWaveguide(), NasuWaveguide()]], pytest.raises(TypeError)),
    ],
)
def test_nasu_writer_extend_raise(gc_param, l_nw, expectation) -> None:
    wwr = NasuWriter([], **gc_param)
    with expectation:
        wwr.extend(l_nw)


def test_nasu_writer_plot2d(gc_param, list_ng) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = NasuWriter(list_ng, **gc_param)
    assert wwr.plot2d(fig=fig) is not None
    del wwr

    wwr = NasuWriter(list_ng, **gc_param)
    assert wwr.plot2d() is not None
    del wwr


def test_nasu_writer_plot3d(gc_param, list_ng) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = NasuWriter(list_ng, **gc_param)
    assert wwr.plot3d(fig=fig) is not None
    del wwr

    wwr = NasuWriter(list_ng, **gc_param)
    assert wwr.plot3d() is not None
    del wwr


def test_nasu_writer_pgm(gc_param, list_ng) -> None:
    wwr = NasuWriter(list_ng, **gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_NASU.pgm'

    assert wwr._export_path.is_dir()
    assert (wwr._export_path / fp).is_file()
    (wwr._export_path / fp).unlink()


def test_nasu_writer_pgm_empty(gc_param) -> None:
    wwr = NasuWriter([], **gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_NASU.pgm'

    assert wwr._export_path.is_dir()
    assert not (wwr._export_path / fp).is_file()


def test_marker_writer_init(gc_param, list_mk) -> None:
    mwr = MarkerWriter(list_mk, **gc_param)
    expp = Path.cwd()

    assert mwr.obj_list == list_mk

    assert mwr._param == gc_param
    assert mwr._export_path == expp

    del mwr

    mwr = MarkerWriter([[list_mk]], **gc_param)
    expp = Path.cwd()

    assert mwr.obj_list == list_mk

    assert mwr._param == gc_param
    assert mwr._export_path == expp

    del mwr

    dirname = 'test'
    mwr = MarkerWriter([], export_dir=dirname, **gc_param)
    expp = Path.cwd() / dirname

    assert mwr.obj_list == []

    assert mwr._param == dict(export_dir=dirname, **gc_param)
    assert mwr._export_path == expp


def test_marker_writer_append(gc_param, list_mk) -> None:
    mwr = MarkerWriter([], **gc_param)
    for mk in list_mk:
        mwr.append(mk)
    assert mwr.obj_list == list_mk


def test_marker_writer_append_raise(gc_param, list_mk) -> None:
    mwr = MarkerWriter([], **gc_param)
    with pytest.raises(TypeError):
        mwr.append(list_mk)


def test_marker_writer_extend(gc_param, list_mk) -> None:
    mwr = MarkerWriter([], **gc_param)
    mwr.extend(list_mk)
    assert mwr.obj_list == list_mk
    del mwr

    mwr = MarkerWriter([], **gc_param)
    new_list = [[list_mk, list_mk, list_mk[0]]]
    mwr.extend(new_list)
    assert mwr.obj_list == flatten(new_list)


@pytest.mark.parametrize(
    'l_mk, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([Waveguide()], pytest.raises(TypeError)),
        ([Marker(), [Marker(), Marker()]], does_not_raise()),
        ((Marker()), pytest.raises(TypeError)),
        ((Marker(), Marker()), pytest.raises(TypeError)),
        ([[[Marker()]], [[Marker(), [Marker(), [Marker(), Marker()]]], Marker()]], does_not_raise()),
    ],
)
def test_marker_writer_extend_raise(gc_param, l_mk, expectation) -> None:
    mwr = MarkerWriter([], **gc_param)
    with expectation:
        mwr.extend(l_mk)


def test_marker_writer_plot2d(gc_param, list_mk) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    mwr = MarkerWriter(list_mk, **gc_param)
    assert mwr.plot2d(fig=fig) is not None
    del mwr

    mwr = MarkerWriter(list_mk, **gc_param)
    assert mwr.plot2d() is not None
    del mwr


def test_marker_writer_plot3d(gc_param, list_mk) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    mwr = MarkerWriter(list_mk, **gc_param)
    assert mwr.plot3d(fig=fig) is not None
    del mwr

    mwr = MarkerWriter(list_mk, **gc_param)
    assert mwr.plot3d() is not None
    del mwr


def test_marker_writer_pgm(gc_param, list_mk) -> None:
    mwr = MarkerWriter(list_mk, **gc_param)
    mwr.pgm()

    fp = Path(mwr.filename).stem + '_MK.pgm'

    assert mwr._export_path.is_dir()
    assert (mwr._export_path / fp).is_file()
    (mwr._export_path / fp).unlink()


def test_marker_writer_pgm_empty(gc_param, list_mk) -> None:
    mwr = MarkerWriter([], **gc_param)
    mwr.pgm()

    fp = Path(mwr.filename).stem + '_MK.pgm'

    assert mwr._export_path.is_dir()
    assert not (mwr._export_path / fp).is_file()
