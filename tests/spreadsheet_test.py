from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from itertools import product
from pathlib import Path

import numpy as np
import openpyxl
import pytest
from femto import __file__ as fpath
from femto.device import Device
from femto.helpers import dotdict
from femto.marker import Marker
from femto.spreadsheet import Spreadsheet
from femto.waveguide import Waveguide

src_path = Path(fpath).parent
dot_path = Path('.').cwd()


@pytest.fixture
def all_cols():
    all_cols = np.genfromtxt(
        src_path / 'utils/spreadsheet_columns.txt',
        delimiter=', ',
        dtype=[
            ('tagname', 'U20'),
            ('fullname', 'U20'),
            ('unit', 'U20'),
            ('width', int),
            ('format', 'U20'),
        ],
    )
    return all_cols


@pytest.fixture
def saints() -> list:
    with open(src_path / 'utils/saints_data.txt') as f:
        lines = f.readlines()
    return lines


@pytest.fixture
def list_wg() -> list[Waveguide]:
    PARAM_WG = dotdict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))

    coup = [Waveguide(**PARAM_WG) for _ in range(5)]
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
def ss_param() -> dotdict:
    return dotdict(
        book_name='custom_book_name.xlsx',
        sheet_name='custom_sheet_name',
        columns_names='name power speed scan depth int_dist yin yout obs',
    )


@pytest.fixture
def wg_param() -> dotdict:
    return dotdict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )


@pytest.fixture
def gc_param() -> dotdict:
    return dotdict(
        filename='test_program.pgm',
        laser='PHAROS',
        n_glass=1.4625,
        n_environment=1.000,
        samplesize=(25, 25),
    )


@pytest.fixture
def device() -> Device:
    powers = np.linspace(600, 800, 5)
    speeds = [20, 30, 40]
    scans = [3, 5, 7]

    dev = Device(filename='test_program.pgm', laser='PHAROS')

    wg_param = dotdict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )

    for i_guide, (p, v, ns) in enumerate(product(powers, speeds, scans)):

        start_pt = [-2, 2 + i_guide * 0.08, wg_param.depth]
        wg = Waveguide(**wg_param, speed=v, scan=ns)
        wg.power = p  # Can NOT be added inside of the arguments of Waveguide
        wg.start(start_pt)
        wg.linear([27, 0, 0])
        wg.end()

        dev.append(wg)

    return dev


@pytest.fixture
def empty_device() -> Device:
    dev = Device(filename='test_program.pgm', laser='PHAROS')
    return dev


def device_redd_cols(redd_cols, non_redd_cols, gc_param):

    # redd_cols: the reddundant columns, so all guides have this same attribute

    dev = Device(**gc_param)
    ints = iter(range(2, 500))

    i_global = 0

    for i_guide in range(10):

        non_rep_value = next(ints)

        start_pt = [-2, 2 + i_guide * 0.08, 0]
        wg = Waveguide()

        for attribute in redd_cols:
            setattr(wg, attribute, 50)  # set them all to the same value

        for attribute in non_redd_cols:
            setattr(wg, attribute, non_rep_value)

        wg.start(start_pt)
        wg.linear([27, 0, 0])
        wg.end()

        dev.append(wg)

    return dev


def test_spsh_defaults(device):
    spsh = Spreadsheet(device)
    spsh.close()
    assert spsh.columns_names == 'name power speed scan radius int_dist depth yin yout obs'
    assert spsh.wb.default_format_properties['font_name'] == 'DejaVu Sans Mono'
    assert spsh.wb.default_format_properties['font_size'] == 11
    assert spsh.wb.filename == 'my_fabrication.xlsx'
    assert spsh.ws.name == 'Fabrication'
    assert spsh.suppr_redd_cols
    assert not spsh.static_preamble
    Path(dot_path / 'my_fabrication.xlsx').unlink()


def test_spsh_initialization_without_device():
    with pytest.raises(TypeError):
        spsh = Spreadsheet()
        spsh.close()


# @pytest.mark.parametrize('fmt_str', ['0000.000', '000000', '00000.', '0', ''])
@pytest.mark.parametrize('fmt_str', ['0000.000'])
def test_create_numerical_format(fmt_str, device, ss_param):
    spsh = Spreadsheet(device, **ss_param)
    spsh._create_numerical_format(fmt_str)
    assert spsh.formats[fmt_str].num_format == fmt_str
    spsh.close()
    (dot_path / 'custom_book_name.xlsx').unlink()


def test_extra_preamble_info(all_cols, device, ss_param):

    extra_preamble_info = {
        'wl': '1030 nm',
        'laboratory': 'CAPABLE',
        'reprate': '1 MHz',
        'material': 'EAGLE XG',
        'facet': 'bottom',
        'thickness': '900 um',
        'preset': '4',
        'attenuator': '33 %',
        'objective': '1000x',
        'sample_name': 'My sample 01',
    }

    ss_param.extra_preamble_info = extra_preamble_info
    ss_param.book_name = 'extra_preamble_info.xlsx'

    device.xlsx(**ss_param)

    wb = openpyxl.load_workbook(dot_path / 'extra_preamble_info.xlsx')
    worksheet = wb['custom_sheet_name']

    for k, v in extra_preamble_info.items():
        found_extra_par = False
        for row in range(1, 50):
            if worksheet.cell(row=row, column=2).value == k.capitalize().replace('_', ' '):
                found_extra_par = True
                assert worksheet.cell(row=row, column=3).value == v
                break
        assert found_extra_par

    (dot_path / 'extra_preamble_info.xlsx').unlink()


def test_add_custom_column(device, ss_param):

    rep_rates = [20, 100, 1000]
    powers = np.linspace(100, 500, 5)
    scan_offsets = np.linspace(0.000100, 0.000500, 5)

    dev = Device(filename='test_program.pgm', laser='PHAROS')

    wg_param = dotdict(
        speed_closed=40,
        radius=40,
        speed=2,
        scan=10,
        depth=-0.100,
        samplesize=(25, 25),
    )

    for i_guide, (rr, p, so) in enumerate(product(rep_rates, powers, scan_offsets)):

        start_pt = [-2, 2 + i_guide * 0.08, wg_param.depth]

        wg = Waveguide(**wg_param)
        wg.power = p  # Can NOT be added inside of the arguments of Waveguide
        wg.reprate = rr
        wg.sco = so

        wg.start(start_pt)
        wg.linear([27, 0, 0])
        wg.end()

        dev.append(wg)

    ss_param.book_name = 'test_new_column.xlsx'
    ss_param.columns_names = 'name power reprate sco yin yout obs'
    ss_param.new_columns = [
        ('sco', 'Scan offset', 'um', 7, '0.0000'),
        ('reprate', 'Rep. rate', 'kHz', 7, '0'),
    ]
    dev.xlsx(**ss_param)

    wb = openpyxl.load_workbook(dot_path / 'test_new_column.xlsx')
    worksheet = wb['custom_sheet_name']

    assert worksheet.cell(row=8, column=9).value == 'Scan offset / um'
    assert worksheet.cell(row=8, column=8).value == 'Rep. rate / kHz'

    for row in range(1, 20):
        assert worksheet.cell(row=row, column=2).value != 'Rep. rate / kHz'
        assert worksheet.cell(row=row, column=2).value != 'Rep. rate / MHz'

    (dot_path / 'test_new_column.xlsx').unlink()


@pytest.mark.parametrize('day', [0, 20, 35, 64, 72, 128, 148, 182, 234, 300, 310])
def test_write_saints_list(device, ss_param, saints, day):
    spsh = Spreadsheet(device, **ss_param)
    spsh._write_saints_list()
    spsh.close()

    wb = openpyxl.load_workbook(dot_path / 'custom_book_name.xlsx')
    worksheet = wb['custom_sheet_name']

    assert worksheet.cell(row=day + 1, column=157).value == saints[day].strip()

    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize('saints', [True, False])
def test_with_saints(saints, device, ss_param):
    ss_param.saints = saints
    spsh = Spreadsheet(device, **ss_param)
    spsh.write_structures()
    spsh.close()

    assert spsh.saints is saints


def test_write_header(device, ss_param):
    spsh = Spreadsheet(device, **ss_param)
    spsh._build_struct_list()
    spsh._write_header()
    spsh.close()

    wb = openpyxl.load_workbook(dot_path / 'custom_book_name.xlsx')
    worksheet = wb['custom_sheet_name']

    assert worksheet.cell(row=2, column=4).value == 'Description'
    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize(
    'cols',
    [
        'power speed scan radius int_dist, ',
        'power, speed scan radius int_dist',
        'power scan, speed radius int_dist',
        'radius, power speed scan',
    ],
)
def test_redd_cols(cols, ss_param, gc_param):

    non_redd_cols, redd_cols = cols.split(', ')
    non_redd_cols = non_redd_cols.split()
    redd_cols = redd_cols.split()

    d = device_redd_cols(redd_cols, non_redd_cols, gc_param)

    ss_param.columns_names = cols.replace(',', '').strip()
    ss_param.suppr_redd_cols = True
    ss_param.device = d

    spsh = Spreadsheet(**ss_param)
    spsh._build_struct_list()
    spsh.close()

    columns_tnames = list(spsh.columns_data['tagname'])
    columns_tnames.remove('name')

    assert all([tn in non_redd_cols for tn in columns_tnames])
    assert all([tn not in redd_cols for tn in columns_tnames])

    (dot_path / 'custom_book_name.xlsx').unlink()


def test_write_structures(device, wg_param, gc_param, ss_param):
    with Spreadsheet(device=device, **ss_param) as spsh:
        spsh.write_structures(verbose=True)

    assert (dot_path / 'custom_book_name.xlsx').is_file()
    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize(
    'powers, speeds, scans',
    [
        ([200.0], [20.0], [3]),
        ([200.0], [20.0, 30.0], [3]),
        ([200.0], [20.0, 30.0], [3, 5]),
        (np.linspace(200, 300, 5), [20.0], [3]),
        (np.linspace(200, 300, 5), [20.0, 30.0], [3]),
        (np.linspace(200, 300, 5), [20.0, 30.0], [3, 5]),
    ],
)
def test_static_preamble(powers, speeds, scans, gc_param, wg_param, ss_param):

    device = Device(**gc_param)

    for i_guide, (p, v, ns) in enumerate(product(powers, speeds, scans)):

        start_pt = [-2, 2 + i_guide * 0.08, wg_param.depth]
        wg = Waveguide(**wg_param, speed=v, scan=ns)
        wg.power = p
        wg.start(start_pt)
        wg.linear([wg.samplesize[0] + 2, 0, 0])
        wg.end()

        device.append(wg)

    obj_list = device.writers[Waveguide].obj_list
    spsh = Spreadsheet(device=device, book_name=ss_param.book_name)
    spsh._build_struct_list(obj_list, columns_names=ss_param.columns_names, static_preamble=True)
    spsh.close()

    for k, v in {'power': powers, 'speed': speeds, 'scan': scans}.items():
        exp = f'{v[0]}' if (len(v) == 1) else 'variable'
        assert spsh.preamble[k].v == exp

    (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize(
    'init_dev, bsl_dev',
    list(product(*2 * [[True, False]])),
)
def test_device_init(device, ss_param, init_dev, bsl_dev):

    ss_pars = ss_param
    bsl_pars = {}

    if init_dev:
        ss_pars['device'] = device

    if bsl_dev:
        bsl_pars['structures'] = device.writers[Waveguide].obj_list

    exp = does_not_raise() if init_dev else pytest.raises(TypeError)
    with exp:
        spsh = Spreadsheet(**ss_pars)
        spsh._build_struct_list()
        spsh.close()

    if (dot_path / 'custom_book_name.xlsx').exists():
        (dot_path / 'custom_book_name.xlsx').unlink()


@pytest.mark.parametrize('verbose', [True, False])
def test_build_structure_list(empty_device, list_wg, list_mk, ss_param, verbose):
    empty_device.extend(list_wg)

    obj_list = empty_device.writers[Waveguide].obj_list

    spsh = Spreadsheet(device=empty_device, **ss_param)
    # Use the defaults for suppressing reddundant olumns and static preamble
    spsh._build_struct_list(obj_list, ss_param.columns_names, verbose=verbose)
    spsh.close()

    assert (dot_path / 'custom_book_name.xlsx').is_file()
    assert len(spsh.struct_data) == len(obj_list)

    (dot_path / 'custom_book_name.xlsx').unlink()
