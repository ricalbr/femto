from __future__ import annotations

import pathlib
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from femto.curves import sin
from femto.helpers import delete_folder
from femto.helpers import flatten
from femto.helpers import listcast
from femto.marker import Marker
from femto.trench import TrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import NasuWriter
from femto.writer import TrenchWriter
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
    PARAM_WG = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [Waveguide(**PARAM_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.linear([5, 0, 0])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.linear([5, 0, 0])
        wg.end()
    return coup


@pytest.fixture
def list_ng() -> list[NasuWaveguide]:
    PARAM_WG = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [NasuWaveguide(**PARAM_WG) for _ in range(20)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch, 0.035])
        wg.linear([5, 0, 0])
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.linear([5, 0, 0])
        wg.end()
    return coup


@pytest.fixture
def list_mk() -> list[Marker]:
    PARAM_MK = dict(scan=1, speed=2, speed_pos=5, speed_closed=5, depth=0.000, lx=1, ly=1)
    markers = []
    for x, y in zip(range(4, 8), range(3, 7)):
        m = Marker(**PARAM_MK)
        m.cross([x, y])
        markers.append(m)
    return markers


@pytest.fixture
def list_tcol(list_wg) -> list[TrenchColumn]:
    PARAM_TC = dict(length=1.0, base_folder='', y_min=-0.1, y_max=19 * 0.08 + 0.1, u=[30.339, 32.825])
    x_c = [3, 7.5, 10.5]
    t_col = []
    for x in x_c:
        T = TrenchColumn(x_center=x, **PARAM_TC)
        T.dig_from_waveguide(flatten([list_wg]), remove=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        t_col.append(T)
    return t_col


@pytest.fixture
def list_utcol(list_wg) -> list[TrenchColumn]:
    PARAM_TC = dict(length=1.0, n_pillars=5, base_folder='', y_min=-0.1, y_max=19 * 0.08 + 0.1, u=[30.339, 32.825])
    x_c = [3, 7.5, 10.5]
    ut_col = []
    for x in x_c:
        T = TrenchColumn(x_center=x, **PARAM_TC)
        T.dig_from_waveguide(flatten([list_wg]), remove=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ut_col.append(T)
    return ut_col


def test_writer(gc_param) -> None:
    from abc import ABCMeta

    Writer.__abstractmethods__ = set()

    class Dummy(Writer):
        pass

    d = Dummy(**gc_param)
    append = d.add(None)
    extend = d.add([None])
    plot2d = d.plot2d(None)
    plot3d = d.plot3d()
    pgm = d.pgm()
    assert isinstance(Writer, ABCMeta)
    assert append is None
    assert extend is None
    assert plot2d is None
    assert plot3d is None
    assert pgm is None


@pytest.mark.parametrize(
    'attr, exp',
    [
        ('_param', does_not_raise()),
        ('_ram', pytest.raises(AttributeError)),
        ('dirname', does_not_raise()),
        ('dirnames', pytest.raises(AttributeError)),
        ('fab_time', pytest.raises(AttributeError)),
        ('_fab_time', pytest.raises(AttributeError)),
        ('_fabtime', does_not_raise()),
    ],
)
def test_trench_writer_slots(gc_param, attr, exp) -> None:
    twr = TrenchWriter(gc_param)
    with exp:
        setattr(twr, attr, None)


def test_trench_writer_init(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, list_tcol)
    dirname = 'TRENCH'
    expp = Path.cwd() / dirname

    tcs = []
    for col in list_tcol:
        for tr in col:
            tcs.append(tr)

    assert twr._obj_list == list_tcol
    assert twr._trenches == tcs
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp

    del twr

    dirname = 'test'
    twr = TrenchWriter(gc_param, objects=[], dirname=dirname)
    expp = Path.cwd() / dirname

    assert twr._obj_list == []
    assert twr._trenches == []
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp


def test_trench_writer_trench_list(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_tcol)
    t_list = [t for tcol in list_tcol for t in tcol]
    assert twr.trench_list == t_list


def test_trench_writer_utrench_list(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_utcol)
    t_list = [t for tcol in list_utcol for t in tcol]
    assert twr.trench_list == t_list


def test_trench_writer_bed_list(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_tcol)
    assert twr.beds_list == []


def test_trench_writer_utrench_bed_list(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_utcol)
    b_list = [b for tcol in list_utcol for b in tcol.bed_list]
    assert twr.beds_list == b_list


def test_trench_writer_append_behaviour(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    for col in list_tcol:
        twr.add(col)
    assert twr._obj_list == list_tcol
    assert twr._trenches == flatten([tr for col in listcast(list_tcol) for tr in col])


def test_trench_writer_fab_time(gc_param) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    twr._fabtime = 10
    assert twr.fab_time == 10


def test_trench_writer_extend_behaviour(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    twr.add(list_tcol)
    assert twr._obj_list == list_tcol
    assert twr._trenches == flatten([tr for col in listcast(list_tcol) for tr in col])
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    new_list = [[[list_tcol]]]
    twr.add(new_list)
    assert twr._obj_list == list_tcol
    assert twr._trenches == flatten([tr for col in listcast(list_tcol) for tr in col])
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    new_list = [[[list_tcol, list_tcol], list_tcol], list_tcol]
    twr.add(new_list)
    assert twr._obj_list == flatten([list_tcol, list_tcol, list_tcol, list_tcol])
    assert twr._trenches == flatten([tr for col in flatten([list_tcol, list_tcol, list_tcol, list_tcol]) for tr in col])
    del twr


def test_trench_writer_extend_raise(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    with pytest.raises(TypeError):
        l_t_col = (list_tcol, TrenchColumn(1, 2, 3))
        twr.add(l_t_col)


def test_trench_writer_add_raise_type(gc_param, list_wg) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    with pytest.raises(TypeError):
        twr.add(list_wg)


def test_trench_writer_plot2d(gc_param, list_tcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    twr = TrenchWriter(gc_param, objects=list_tcol)
    assert twr.plot2d(fig=fig) is not None
    del twr

    twr = TrenchWriter(gc_param, objects=list_tcol)
    assert twr.plot2d() is not None
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    assert twr.plot2d() is not None
    del twr


def test_trench_writer_plot3d(gc_param, list_tcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()
    twr = TrenchWriter(gc_param, objects=list_tcol)
    assert twr.plot3d(fig=fig) is not None
    del twr, fig

    twr = TrenchWriter(gc_param, objects=list_tcol)
    assert twr.plot3d() is not None
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    assert twr.plot3d() is not None
    del twr


def test_trench_writer_pgm_empty(gc_param) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    assert twr.pgm() is None
    assert not twr._export_path.is_dir()


def test_trench_writer_pgm(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_tcol)
    twr.pgm()

    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_tcol):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()
        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
        delete_folder(twr._export_path / f'trenchCol{i_col + 1:03}')
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert (twr._export_path / 'MAIN.pgm').is_file()
    (twr._export_path / 'MAIN.pgm').unlink()
    delete_folder(twr._export_path)


def test_trench_writer_pgm_custom_filename(gc_param, list_tcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_tcol)
    old_dir = twr._export_path
    twr.pgm(filename='trenchtest')

    assert twr._export_path == old_dir / 'TRENCHTEST'
    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_tcol):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()
        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
        delete_folder(twr._export_path / f'trenchCol{i_col + 1:03}')
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert (twr._export_path / 'MAIN.pgm').is_file()
    (twr._export_path / 'MAIN.pgm').unlink()
    delete_folder(twr._export_path)


def test_trench_writer_pgm_single_col(gc_param, list_tcol) -> None:
    list_cols = [list_tcol[0]]
    twr = TrenchWriter(gc_param, objects=list_cols)
    twr.pgm()

    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_cols):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()
        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
        delete_folder(twr._export_path / f'trenchCol{i_col + 1:03}')
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert not (twr._export_path / 'MAIN.pgm').is_file()
    delete_folder(twr._export_path)


def test_utrench_writer_init(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=list_utcol)
    dirname = 'TRENCH'
    expp = Path.cwd() / dirname

    tcs = []
    bds = []
    for col in list_utcol:
        for tr in col:
            tcs.append(tr)
        bds.extend(col._trenchbed)

    assert twr.objs == list_utcol
    assert twr._trenches == tcs
    assert twr._beds == bds
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp

    del twr

    dirname = 'test'
    twr = TrenchWriter(gc_param, objects=[], dirname=dirname)
    expp = Path.cwd() / dirname

    assert twr._obj_list == []
    assert twr._trenches == []
    assert twr._beds == []
    assert twr.dirname == dirname

    assert twr._param == gc_param
    assert twr._export_path == expp


def test_utrench_writer_append_behaviour(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    for col in list_utcol:
        twr.add(col)
    assert twr._obj_list == list_utcol
    assert twr._trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr._beds == flatten([bd for col in listcast(list_utcol) for bd in col._trenchbed])


def test_utrench_writer_extend_behaviour(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    twr.add(list_utcol)
    assert twr._obj_list == list_utcol
    assert twr._trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr._beds == flatten([bd for col in listcast(list_utcol) for bd in col._trenchbed])
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    new_list = [[[list_utcol]]]
    twr.add(new_list)
    assert twr._obj_list == list_utcol
    assert twr._trenches == flatten([tr for col in listcast(list_utcol) for tr in col])
    assert twr._beds == flatten([bd for col in listcast(list_utcol) for bd in col._trenchbed])
    del twr

    twr = TrenchWriter(gc_param, objects=[])
    new_list = [[[list_utcol, list_utcol], list_utcol], list_utcol]
    twr.add(new_list)
    assert twr._obj_list == flatten([list_utcol, list_utcol, list_utcol, list_utcol])
    assert twr._trenches == flatten(
        [tr for col in flatten([list_utcol, list_utcol, list_utcol, list_utcol]) for tr in col]
    )
    assert twr._beds == flatten(
        [bd for col in flatten([list_utcol, list_utcol, list_utcol, list_utcol]) for bd in col._trenchbed]
    )
    del twr


def test_utrench_writer_extend_raise(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    with pytest.raises(TypeError):
        l_ut_col = (list_utcol, TrenchColumn(1, 2, 3))
        twr.add(l_ut_col)


def test_utrench_writer_plot2d(gc_param, list_utcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    twr = TrenchWriter(gc_param, list_utcol)
    assert twr.plot2d(fig=fig) is not None
    del twr

    twr = TrenchWriter(gc_param, list_utcol)
    assert twr.plot2d() is not None
    del twr


def test_utrench_writer_plot3d(gc_param, list_utcol) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()
    twr = TrenchWriter(gc_param, list_utcol)
    assert twr.plot3d(fig=fig) is not None
    del twr, fig

    twr = TrenchWriter(gc_param, list_utcol)
    assert twr.plot3d() is not None
    del twr


def test_utrench_writer_pgm_empty(gc_param) -> None:
    twr = TrenchWriter(gc_param, objects=[])
    assert twr.pgm() is None
    assert not twr._export_path.is_dir()


def test_utrench_writer_pgm(gc_param, list_utcol) -> None:
    twr = TrenchWriter(gc_param, list_utcol)
    twr.pgm()

    assert twr._export_path.is_dir()
    for i_col, col in enumerate(list_utcol):
        assert (twr._export_path / f'trenchCol{i_col + 1:03}').is_dir()
        assert (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').is_file()

        for i_tr, tr in enumerate(col):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_WALL.pgm').is_file()
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench{i_tr + 1:03}_FLOOR.pgm').is_file()
        for i_bed, _ in enumerate(col._trenchbed):
            assert (twr._export_path / f'trenchCol{i_col + 1:03}' / f'trench_BED_{i_bed + 1:03}.pgm').is_file()
        delete_folder(twr._export_path / f'trenchCol{i_col + 1:03}')
        (twr._export_path / f'FARCALL{i_col + 1:03}.pgm').unlink()
    assert (twr._export_path / 'MAIN.pgm').is_file()
    (twr._export_path / 'MAIN.pgm').unlink()
    delete_folder(twr._export_path)


@pytest.mark.parametrize(
    'attr, exp',
    [
        ('_param', does_not_raise()),
        ('_ram', pytest.raises(AttributeError)),
        ('dirname', does_not_raise()),
        ('dirnames', pytest.raises(AttributeError)),
        ('fab_time', pytest.raises(AttributeError)),
        ('_fab_time', pytest.raises(AttributeError)),
        ('_fabtime', does_not_raise()),
    ],
)
def test_waveguide_writer_slots(gc_param, attr, exp) -> None:
    wwr = WaveguideWriter(gc_param)
    with exp:
        setattr(wwr, attr, None)


def test_waveguide_writer_init(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(gc_param, objects=list_wg)
    expp = Path.cwd()

    assert wwr._obj_list == list_wg

    assert wwr._param == gc_param
    assert wwr._export_path == expp

    del wwr

    dirname = 'test'
    wwr = WaveguideWriter(gc_param, objects=[], export_dir=dirname)
    expp = Path.cwd() / dirname

    assert wwr._obj_list == []

    assert wwr._param == dict(export_dir=dirname, **gc_param)
    assert wwr._export_path == expp


def test_waveguide_writer_fab_time(gc_param) -> None:
    wwr = WaveguideWriter(gc_param, objects=[])
    wwr._fabtime = 10
    assert wwr.fab_time == 10


def test_waveguide_writer_append_behaviour(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(gc_param)
    for wg in list_wg:
        wwr.add(wg)
    assert wwr._obj_list == list_wg


def test_waveguide_writer_extend_behaviour(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(gc_param)
    wwr.add(list_wg)
    assert wwr._obj_list == list_wg
    del wwr

    wwr = WaveguideWriter(gc_param)
    new_list = [*list_wg, *list_wg, list_wg[0]]
    wwr.add(new_list)
    assert wwr._obj_list == new_list


@pytest.mark.parametrize(
    'l_wg, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([Waveguide()], does_not_raise()),
        ([Waveguide(), [Waveguide(), Waveguide()]], does_not_raise()),
        ([[[Waveguide()]], [[Waveguide(), [Waveguide(), [Waveguide(), Waveguide()]]], Waveguide()]], does_not_raise()),
        ([TrenchColumn(x_center=1, y_min=2, y_max=3), [Waveguide(), Waveguide()]], pytest.raises(TypeError)),
    ],
)
def test_waveguide_writer_extend_raise(gc_param, l_wg, expectation) -> None:
    wwr = WaveguideWriter(gc_param)
    with expectation:
        wwr.add(l_wg)


def test_waveguide_writer_plot2d(gc_param, list_wg) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = WaveguideWriter(gc_param, objects=list_wg)
    assert wwr.plot2d(fig=fig) is not None
    del wwr

    wwr = WaveguideWriter(gc_param, objects=list_wg)
    assert wwr.plot2d() is not None
    del wwr


def test_waveguide_writer_plot3d(gc_param, list_wg) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = WaveguideWriter(gc_param, objects=list_wg)
    assert wwr.plot3d(fig=fig) is not None
    del wwr

    wwr = WaveguideWriter(gc_param, objects=list_wg)
    assert wwr.plot3d() is not None
    del wwr


def test_waveguide_writer_pgm(gc_param, list_wg) -> None:
    wwr = WaveguideWriter(gc_param, objects=list_wg)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_WG.pgm'

    assert wwr._export_path.is_dir()
    assert (wwr._export_path / fp).is_file()
    (wwr._export_path / fp).unlink()


def test_waveguide_writer_pgm_empty(gc_param) -> None:
    wwr = WaveguideWriter(gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_WG.pgm'

    assert wwr._export_path.is_dir()
    assert not (wwr._export_path / fp).is_file()


def test_export_default(gc_param, list_wg) -> None:
    import os

    wwr = WaveguideWriter(gc_param, list_wg)
    wwr.export()

    fn = pathlib.Path(wwr.CWD) / wwr.export_dir / 'EXPORT' / pathlib.Path(wwr.filename).stem
    assert fn.is_dir()
    _, _, files = next(os.walk(fn))
    assert len(files) == len(list_wg)
    delete_folder(fn.parent)


def test_export_custom_filename(gc_param, list_wg) -> None:
    import os

    filename = 'CUSTOM_FOLDER'
    wwr = WaveguideWriter(gc_param, list_wg)
    wwr.export(filename=filename)

    fn = pathlib.Path(wwr.CWD) / wwr.export_dir / 'EXPORT' / filename
    assert fn.is_dir()
    _, _, files = next(os.walk(fn))
    assert len(files) == len(list_wg)
    delete_folder(wwr.CWD / wwr.export_dir / 'EXPORT')


def test_export_custom_dir(gc_param, list_wg) -> None:
    import os

    dir = 'CUSTOM_FOLDER'
    wwr = WaveguideWriter(gc_param, list_wg)
    wwr.export(export_root=dir)

    fn = pathlib.Path(wwr.CWD) / wwr.export_dir / dir / pathlib.Path(wwr.filename).stem
    assert fn.is_dir()
    _, _, files = next(os.walk(fn))
    assert len(files) == len(list_wg)
    delete_folder(wwr.CWD / wwr.export_dir / dir)


def test_export_none_dirs(gc_param, list_wg) -> None:
    import os

    dir = None
    gc_param['export_dir'] = None
    wwr = WaveguideWriter(gc_param, list_wg)
    wwr.export(export_root=dir)

    fn = pathlib.Path(wwr.CWD) / pathlib.Path(wwr.filename).stem
    assert fn.is_dir()
    _, _, files = next(os.walk(fn))
    assert len(files) == len(list_wg)
    for f in files:
        pathlib.Path(fn / f).unlink()
    fn.rmdir()


@pytest.mark.parametrize(
    'attr, exp',
    [
        ('_param', does_not_raise()),
        ('_ram', pytest.raises(AttributeError)),
        ('dirname', does_not_raise()),
        ('dirnames', pytest.raises(AttributeError)),
        ('fab_time', pytest.raises(AttributeError)),
        ('_fab_time', pytest.raises(AttributeError)),
        ('_fabtime', does_not_raise()),
    ],
)
def test_nasu_waveguide_writer_slots(gc_param, attr, exp) -> None:
    wwr = NasuWriter(gc_param)
    with exp:
        setattr(wwr, attr, None)


def test_nasu_writer_init(gc_param, list_ng) -> None:
    wwr = NasuWriter(gc_param, objects=list_ng)
    expp = Path.cwd()

    assert wwr._obj_list == list_ng

    assert wwr._param == gc_param
    assert wwr._export_path == expp

    del wwr

    dirname = 'test'
    wwr = NasuWriter(gc_param, export_dir=dirname)
    expp = Path.cwd() / dirname

    assert wwr._obj_list == []

    assert wwr._param == dict(export_dir=dirname, **gc_param)
    assert wwr._export_path == expp


def test_nasu_waveguide_writer_fab_time(gc_param) -> None:
    wwr = NasuWriter(gc_param, objects=[])
    wwr._fabtime = 10
    assert wwr.fab_time == 10


def test_nasu_writer_append_behaviour(gc_param, list_ng) -> None:
    wwr = NasuWriter(gc_param)
    for wg in list_ng:
        wwr.add(wg)
    assert wwr._obj_list == list_ng


def test_nasu_writer_extend_behaviour(gc_param, list_ng) -> None:
    wwr = NasuWriter(gc_param)
    wwr.add(list_ng)
    assert wwr._obj_list == list_ng
    del wwr

    wwr = NasuWriter(gc_param)
    new_list = [*list_ng, *list_ng, list_ng[0]]
    wwr.add(new_list)
    assert wwr._obj_list == new_list


def test_nasu_writer_objects(gc_param, list_ng) -> None:
    wwr = NasuWriter(gc_param)
    wwr.add(list_ng)
    assert wwr.objs == list_ng


@pytest.mark.parametrize(
    'l_nw, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([NasuWaveguide()], does_not_raise()),
        (NasuWaveguide(), does_not_raise()),
        ([NasuWaveguide(), [NasuWaveguide(), NasuWaveguide()]], does_not_raise()),
        ([[[NasuWaveguide()]], [[NasuWaveguide(), [NasuWaveguide(), [NasuWaveguide()]]]]], does_not_raise()),
        ([Waveguide(), [NasuWaveguide(), NasuWaveguide()]], pytest.raises(TypeError)),
    ],
)
def test_nasu_writer_extend_raise(gc_param, l_nw, expectation) -> None:
    wwr = NasuWriter(gc_param)
    with expectation:
        wwr.add(l_nw)


def test_nasu_writer_plot2d(gc_param, list_ng) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = NasuWriter(gc_param, objects=list_ng)
    assert wwr.plot2d(fig=fig) is not None
    del wwr

    wwr = NasuWriter(gc_param, objects=list_ng)
    assert wwr.plot2d() is not None
    del wwr


def test_nasu_writer_plot3d(gc_param, list_ng) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    wwr = NasuWriter(gc_param, objects=list_ng)
    assert wwr.plot3d(fig=fig) is not None
    del wwr

    wwr = NasuWriter(gc_param, objects=list_ng)
    assert wwr.plot3d() is not None
    del wwr


def test_nasu_writer_pgm(gc_param, list_ng) -> None:
    wwr = NasuWriter(gc_param, objects=list_ng)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_NASU.pgm'

    assert wwr._export_path.is_dir()
    assert (wwr._export_path / fp).is_file()
    (wwr._export_path / fp).unlink()


def test_nasu_writer_pgm_empty(gc_param) -> None:
    wwr = NasuWriter(gc_param)
    wwr.pgm()

    fp = Path(wwr.filename).stem + '_NASU.pgm'

    assert wwr._export_path.is_dir()
    assert not (wwr._export_path / fp).is_file()


@pytest.mark.parametrize(
    'attr, exp',
    [
        ('_param', does_not_raise()),
        ('_ram', pytest.raises(AttributeError)),
        ('dirname', does_not_raise()),
        ('dirnames', pytest.raises(AttributeError)),
        ('fab_time', pytest.raises(AttributeError)),
        ('_fab_time', pytest.raises(AttributeError)),
        ('_fabtime', does_not_raise()),
    ],
)
def test_marker_waveguide_writer_slots(gc_param, attr, exp) -> None:
    mwr = MarkerWriter(gc_param)
    with exp:
        setattr(mwr, attr, None)


def test_marker_waveguide_writer_fab_time(gc_param) -> None:
    wwr = MarkerWriter(gc_param, objects=[])
    wwr._fabtime = 10
    assert wwr.fab_time == 10


def test_marker_writer_init(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param, objects=list_mk)
    expp = Path.cwd()

    assert mwr._obj_list == list_mk

    assert mwr._param == gc_param
    assert mwr._export_path == expp

    del mwr

    mwr = MarkerWriter(gc_param, objects=[[list_mk]])
    expp = Path.cwd()

    assert mwr._obj_list == list_mk

    assert mwr._param == gc_param
    assert mwr._export_path == expp

    del mwr

    dirname = 'test'
    print()
    print(gc_param)
    mwr = MarkerWriter(gc_param, export_dir=dirname)
    expp = Path.cwd() / dirname

    assert mwr._obj_list == []

    print(gc_param)
    assert mwr._param == dict(export_dir=dirname, **gc_param)
    assert mwr._export_path == expp


def test_marker_writer_append_behaviour(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param)
    for mk in list_mk:
        mwr.add(mk)
    assert mwr._obj_list == list_mk


def test_marker_writer_extend_behaviour(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param)
    mwr.add(list_mk)
    assert mwr._obj_list == list_mk
    del mwr

    mwr = MarkerWriter(gc_param)
    new_list = [[list_mk, list_mk, list_mk[0]]]
    mwr.add(new_list)
    assert mwr._obj_list == flatten(new_list)


def test_marker_writer_objects(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param)
    mwr.add(list_mk)
    assert mwr.objs == list_mk


@pytest.mark.parametrize(
    'l_mk, expectation',
    [
        (None, pytest.raises(TypeError)),
        ([Waveguide()], pytest.raises(TypeError)),
        ([Marker(), [Marker(), Marker()]], does_not_raise()),
        ((Marker()), does_not_raise()),
        ((Marker(), Marker()), does_not_raise()),
        ([[[Marker()]], [[Marker(), [Marker(), [Marker(), Marker()]]], Marker()]], does_not_raise()),
    ],
)
def test_marker_writer_extend_raise(gc_param, l_mk, expectation) -> None:
    mwr = MarkerWriter(gc_param)
    with expectation:
        mwr.add(l_mk)


def test_marker_writer_plot2d(gc_param, list_mk) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    mwr = MarkerWriter(gc_param, objects=list_mk)
    assert mwr.plot2d(fig=fig) is not None
    del mwr

    mwr = MarkerWriter(gc_param, objects=list_mk)
    assert mwr.plot2d() is not None
    del mwr


def test_marker_writer_plot3d(gc_param, list_mk) -> None:
    from plotly import graph_objs as go

    fig = go.Figure()

    mwr = MarkerWriter(gc_param, objects=list_mk)
    assert mwr.plot3d(fig=fig) is not None
    del mwr

    mwr = MarkerWriter(gc_param, objects=list_mk)
    assert mwr.plot3d() is not None
    del mwr


def test_marker_writer_pgm(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param, objects=list_mk)
    mwr.pgm()

    fp = Path(mwr.filename).stem + '_MK.pgm'

    assert mwr._export_path.is_dir()
    assert (mwr._export_path / fp).is_file()
    (mwr._export_path / fp).unlink()


def test_marker_writer_pgm_empty(gc_param, list_mk) -> None:
    mwr = MarkerWriter(gc_param)
    mwr.pgm()

    fp = Path(mwr.filename).stem + '_MK.pgm'

    assert mwr._export_path.is_dir()
    assert not (mwr._export_path / fp).is_file()
