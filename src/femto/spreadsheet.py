from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import cast
from typing import Union

import femto.device
import nptyping as nptyp
import numpy as np
from femto import __file__ as fpath
from femto.helpers import flatten
from femto.marker import Marker
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import WaveguideWriter
from xlsxwriter import Workbook


def generate_all_cols_data() -> (
    nptyp.NDArray[
        Any,
        nptyp.Structure[tagname:Str, fullname:Str, unit:Str, width:Int, format:Str],
    ]
):
    """
    Create the available columns array from a file.

    Gathers all data from the ``utils/spreadsheet_columns.txt`` file and
    creates a structured array with the information for all possible columns.
    The user can only select columns to add to the spreadsheet throught their
    tagname, which must be in the first column of the txt document.
    """
    all_cols = np.genfromtxt(
        Path(fpath).parent / 'utils' / 'spreadsheet_columns.txt',
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


class NestedDict:
    """Class to handle a nested dictionary."""

    def __init__(self, d):
        self.dict = d

    def __getitem__(self, k):
        """Support indexing, return the first occurence."""
        path = NestedDict.get_path(k, self.dict)

        if path:
            ret = self.dict
            for step in path[0].split():
                ret = ret[step]
            return ret
        else:
            return None

    @staticmethod
    def get_path(key, d, path=None, prev_k=None):
        """Get paths to all occurencies of the key in the dictionary."""
        if prev_k is None:
            prev_k = []

        if path is None:
            path = []

        nothing = True
        if hasattr(d, 'items'):
            for k, v in d.items():

                if k == key:
                    prev_k.append(k)
                    path.append(' '.join(prev_k))
                    nothing = False

                if isinstance(v, dict):
                    prev_k.append(k)
                    path = NestedDict.get_path(key, v, path=path, prev_k=prev_k)

            if nothing and prev_k:
                prev_k.pop(-1)

        return path

    def pop(self, k):
        """Pop element with a given key from nested structure."""
        paths = NestedDict.get_path(k, self.dict)
        elem = self.dict
        for p in paths:
            steps = p.split()
            final = steps[-1]
            for step in steps[:-1]:
                elem = elem[step]
            elem.pop(final)


@dataclass
class Parameter:
    """Class that handles preamble parameters."""

    n: str  # Full name
    v: str = ''  # Value
    loc: tuple[int, int] = (0, 0)  # Location (1-indexing)
    sz: tuple[int, int] = (0, 0)  # Size (for merged cells)
    fmt: str = 'parval'  # Format

    def __post_init__(self):
        """Set row and column, from the location with Excel 1-indexing."""
        self.row = self.loc[0] + 1
        self.col = self.loc[1] + 1

    def _set_loc(self, loc: tuple[int, int]):
        """Set location of a cell parameter."""
        self.loc = loc
        self.row = self.loc[0] + 1
        self.col = self.loc[1] + 1


@dataclass
class Spreadsheet:
    """Class representing the spreadsheet with all entities to fabricate."""

    device: femto.device.Device | None = None
    columns_names: str = ''
    book_name: str | Path = 'my_fabrication.xlsx'
    sheet_name: str = 'Fabrication'
    font_name: str = 'DejaVu Sans Mono'
    font_size: int = 11
    suppr_redd_cols: bool = True
    static_preamble: bool = False
    saints: bool = False
    new_columns: list | None = None
    extra_preamble_info: dict | None = None

    def __post_init__(
        self,
    ) -> None:
        """Intitialization of the Spreadsheet object.

        Opens a new workbook with a default spreadsheet named ``Fabrication``.
        Creates the basic formats that will be used, determines the columns
        that really need to be used, and writed the map of the fabrication.

        Parameters
        ----------
        femto_cell: femto.device.Device
            Femto Device object, which can contain all sorts of structures.

        columns_names: str
            The columns to be written, separated by a single whitespace.
            The user must provide a string with tagnames separated by a
            whitespace and the tagnames must be contained in the first column
            of the text file ``columns.txt`` located in the utils folder.

        book_name: str
            Name of the Excel file, without the extension, which will be
            added automatically.
            Defaults to ``my_fabrication``.

        sheet_name: str
            Name of the Excel spreadsheet.
            Defaults to ``Fabrication``.

        suppr_redd_cols: bool
            If True, it will suppress all redundant columns, meaning that it
            will not include them in the final spreadsheet, even if they are in
            the sel_cols string. Redundant columns are columns that contain the
            same value for all of the lines (structures) in the file.
            Defaults to True.

        static_preamble: bool
            If True, the preamble contains always the same information. For
            instance, if power changes during fabrication, the preamble should
            not contain this information, and a dedicated column would appear.
            However, with static_preamble, a preamble row appears with the in
            formation ``variable``.
            Defaults to False.

        Returns
        -------
        None

        """
        if self.device is None:
            raise TypeError('Device must be given when initializing Spreadsheet.')

        if not self.columns_names:
            scn = 'name power speed scan radius int_dist depth yin yout obs'
            self.columns_names = scn
            self.suppr_redd_cols = True
            print(
                'Columns_names not given in spreadsheet initialization.'
                f' Will proceed with standard columns names \'{scn}\' '
                'and activate the suppr_redd_cols flag to deal with '
                'reddundant columns.'
            )

        if 'name' not in self.columns_names:
            self.columns_names = f'name {self.columns_names}'

        if isinstance(self.book_name, Path):
            spsh_dir = self.book_name.parent
            spsh_dir.mkdir(parents=True, exist_ok=True)

        if self.extra_preamble_info is None:
            self.extra_preamble_info = {}

        ac = generate_all_cols_data()
        if self.new_columns:

            for elem in self.new_columns:
                tns = ac['tagname']
                if elem[0] in tns:
                    # tagname already present, replace
                    ind = [i for i, tn in enumerate(tns) if tn != elem[0]]
                    ac = ac[ind]

                ac = np.append(ac, np.array([elem], dtype=ac.dtype))

        self.all_cols = ac

        defaults = {'font_name': self.font_name, 'font_size': self.font_size}
        self.wb = Workbook(self.book_name, options={'default_format_properties': defaults})
        self.ws = self.wb.add_worksheet(self.sheet_name)

        self.wb.set_calc_mode('auto')

        # Create all the Parameters contained in the general preamble_info
        # Add them to a dictionary with the key equal to their tagname

        preamble_info: dict = {
            'General': 'laboratory temperature humidity date preghiera start end sample_name',
            'Substrate': 'material facet thickness',
            'Laser': 'laser_name wl duration reprate attenuator preset',
            'Irradiation': 'objective power speed scan depth',
        }

        pr = {}
        for k, v in preamble_info.items():
            subcat_data = {}
            for t in v.split(' '):
                p = Parameter(t.replace('_', ' '))
                subcat_data[t] = p
            pr[k] = subcat_data

        preamble = NestedDict(pr)

        self.description = self.extra_preamble_info.pop('description', '')

        for key, value in self.extra_preamble_info.items():
            preamble[key].v = value

        preamble['laser_name'].v = self.device._param['laser']

        # Preghiera has specific value, not ""
        # FORMULAS ALWAYS WITH , NOT ;
        formula = (
            '=IF(ISBLANK(INDIRECT(ADDRESS(INDEX(ROW(B:B),MATCH("Date",'
            'B:B,0)),3))),"",OFFSET(FA2,DATE(2020,MONTH(INDIRECT('
            'ADDRESS(INDEX(ROW(B:B),MATCH("Date",B:B,0)),3))),'
            'DAY(INDIRECT(ADDRESS(INDEX(ROW(B:B),MATCH("Date",B:B,0)),'
            '3))))-DATE(2020,1,0)-1,0))'
        )
        preamble['preghiera'].v = formula
        preamble['preghiera'].n = '=IF(ISBLANK(INDIRECT(ADDRESS(INDEX(' 'ROW(B:B),MATCH("Date",B:B,0)),3))),' '"",FA1)'
        preamble['preghiera'].sz = np.array([2, 0])

        dt = time.gmtime(self.device.fabrication_time)
        formula = (
            '=INDIRECT(ADDRESS(INDEX(ROW(B:B),MATCH("Start",B:B,0)),3))'
            f' + TIME({dt.tm_hour},{dt.tm_min},{dt.tm_sec})'
        )

        preamble['end'].v = formula
        preamble['end'].fmt = 'time'
        preamble['start'].fmt = 'time'
        preamble['date'].fmt = 'date'

        self.preamble = preamble
        self._create_formats()

    def __enter__(self) -> Spreadsheet:
        """Context manager entry.

        Can be use like:
        ::
            with Spreadsheet(**SS_PARAMETERS) as ss:
                ...
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def close(self):
        """Close the workbook."""
        self.wb.close()

    def write_structures(
        self,
        verbose: bool = True,
        saints: bool | None = None,
    ) -> None:
        """
        Write the structures to the spreadsheet.

        Builds the structures list, containing all the required information
        about the structures to fabricate. Then, prepares the columns in the
        spreadsheet, setting their width according to the ``columns.txt`` file.
        Finally, fills the spreadsheet with the contents of the structures
        list, one structure per line, and creates the preamble, which is the
        general information to the left of individual structure information.

        Returns
        -------
        None.

        """
        if saints is None:
            saints = self.saints

        obj_list = self._get_structure_list()

        if not saints:
            self.preamble.pop('preghiera')
        if saints:
            self._write_saints_list()

        self._build_struct_list(obj_list)
        self._prepare_columns()
        self._fill_spreadsheet()
        self._write_header()
        self._write_preamble()

    # Private interface
    def _write_saints_list(self, column: int = 156) -> None:
        """Write a list with all the daily saints to which pray.

        In a distant column, write in cronological order the saint celebrated
        in each day of the year. This data is available under the file
        ``saints_data.txt`` in the utils folder. When writing the preamble,
        upon adding the fabrication date, an empty box is filled with the words
        ``Oggi dobbiamo pregare...`` meaning ``Today we must pray...``, in
        order to protect our fabrication from bugs and general bad luck.

        Returns
        -------
        None.

        """
        with open(Path(fpath).parent / 'utils' / 'saints_data.txt') as f:
            for i in range(367):
                s = f.readline().strip('\n')
                # print(f'writing day {i}\t{s}')
                self.ws.write(i, column, s)

    def _dtype(self, tagname):
        """Return the data type corresponding to a give column tagname.

        The data type is determined in the ``columns.txt`` file, under the
        column named ``format``. The dtypes are assigned according to the
        following possibilities for that field:

        - ``text`` or ``title`` -> dtype: 20-character str
        - sequence of zeros with a dot somewhere -> dtype: float
        - sequence of zeros with no dot -> dtype: int

        Parameters
        ----------
        tagname: str
            The tagname of the column type. Must be contained in the
            ``columns.txt`` file under the ``utils`` folder.
            Not case sensitive.

        Returns
        -------
        dt: type
            Type of the data.

        """
        ac = self.all_cols
        ind = np.where(ac['tagname'] == tagname.lower())[0][0]

        if ac['format'][ind] in 'text title':
            dt = 'U20'
        elif '.' in ac['format'][ind]:
            dt = np.float64
        else:
            dt = np.int64

        return dt

    def _get_structure_list(self, str_list: list[Waveguide | Marker] | None = None) -> list[Waveguide | Marker]:

        assert isinstance(self.device, femto.device.Device)

        if str_list is None:
            d = self.device
            wgwr = cast(WaveguideWriter, d.writers[Waveguide])
            mkwr = cast(MarkerWriter, d.writers[Marker])
            wgstrucs = flatten(wgwr.obj_list)
            mkstrucs = flatten(mkwr.obj_list)
        else:
            wgstrucs = [s for s in flatten(str_list) if isinstance(s, Waveguide)]
            mkstrucs = [s for s in flatten(str_list) if isinstance(s, Marker)]

        wgstrucs.sort(key=lambda wg: wg.path3d[1][0])
        structures = wgstrucs + mkstrucs

        return structures

    def _build_struct_list(
        self,
        structures: list[Waveguide | Marker] | None = None,
        columns_names: str | None = None,
        suppr_redd_cols: bool | None = None,
        static_preamble: bool | None = None,
        verbose: bool = False,
    ):
        """
        Build a table with all of the structures.

        The table has as lines the several structures, and for each of them,
        all of the fields given as columns_names as column data. Determines the
        columns that will be effectively added to the sheet, according to the
        specified suppr_redd_cols and static_preamble.

        Parameters
        ----------
        structures: list
            Contains the waveguides and the markers to be added to the table.

        columns_names: str
            The relevant table columns, separated by a single whitespace.
            The user must provide a string with tagnames separated by a
            whitespace and the tagnames must be contained in the first column
            of the text file ``columns.txt`` located in the utils folder.

        suppr_redd_cols: bool
            If True, it will suppress all redundant columns, meaning that it
            will not include them in the final spreadsheet, even if they are in
            the sel_cols string. Redundant columns are columns that contain the
            same value for all of the lines (structures) in the file.
            Defaults to the given instation value (otherwise True).

        static_preamble: bool
            If True, the preamble contains always the same information. For
            instance, if power changes during fabrication, the preamble should
            not contain this information, and a dedicated column would appear.
            However, with static_preamble, a preamble row appears with the in
            formation ``variable``.
            Defaults to the given instation value (otherwise False).

        verbose: bool
            If True, prints the columns, selected by the user, that will be
            excluded from the spreadsheet because they are reddundant (in the
            case that suppr_redd_cols is set to True).

        """
        coords = lambda x: {
            Waveguide: {'yin': x.path3d[1][0], 'yout': x.path3d[1][-1]},
            Marker: {
                'yin': (max(x.path3d[0][:]) + min(x.path3d[0][:])) / 2,
                'yout': (max(x.path3d[1][:]) + min(x.path3d[1][:])) / 2,
            },
        }

        structures = self._get_structure_list(structures)
        n_structures = len(structures)

        if columns_names is None:
            columns_names = self.columns_names

        if suppr_redd_cols is None:
            suppr_redd_cols = self.suppr_redd_cols

        if static_preamble is None:
            static_preamble = self.static_preamble

        sel_cols = columns_names.split(' ')

        ac = self.all_cols

        inds = [np.where(ac['tagname'] == sc)[0][0] for sc in sel_cols]

        cols_data = ac[inds]
        tagnames = cols_data['tagname']
        dtype = [(t, self._dtype(t)) for t in tagnames]

        table_lines = np.zeros(n_structures, dtype=dtype)

        for i, ent in enumerate(structures):

            sline = []

            for t in tagnames:

                if t in 'yin yout':
                    item = coords(ent)[type(ent)][t]
                else:
                    item = getattr(ent, t, None)

                if item is None:
                    item = 1.1e5 if self._dtype(t) in [np.float64, np.int64] else ''

                sline.append(item)

            table_lines[i] = tuple(sline)

        keep = []

        ignored_fields = []

        for i, t in enumerate(tagnames):

            if t.lower() == 'name':
                # the name of the structure is always present
                keep.append(i)
                continue

            if table_lines.dtype.fields[t][0].char in 'ld' and np.all(table_lines[t] > 1e5):
                ignored_fields.append(t)
                continue

            if np.all(table_lines[t] == table_lines[t][0]) and suppr_redd_cols and table_lines[t][0] != '':
                # eliminate reddundancies if explicitly requested
                ignored_fields.append(t)

                if self.preamble[t]:
                    # is it something that might go on the preamble?
                    # If yes, put it there
                    self.preamble[t].v = f'{table_lines[t][0]}'

                continue

            elif static_preamble and self.preamble[t]:
                # keep it in the preamble with the indication variable
                self.preamble[t].v = 'variable'

            elif self.preamble[t]:
                # it has a dedicated column, so need not be in the preamble
                self.preamble.pop(t)

            keep.append(i)

        if ignored_fields and verbose:
            fields_left_out = ', '.join(ignored_fields)
            print(
                f'For all entities, the fields {fields_left_out} were not '
                'defined, so they will not be shown as table columns.'
            )

        self.keep = keep
        self.struct_data = table_lines[tagnames[keep]]
        self.columns_data = cols_data[keep]
        self.cols_data = cols_data

    def _prepare_columns(self, columns=None) -> None:
        """
        Prepare the columns that will be present in the spreadsheet.

        Create the data format and set the correct width.
        """
        start = 5

        if columns is None:
            columns = self.cols_data[self.keep]
            self.columns_data = columns

        for i, col in enumerate(columns):

            fmt = col['format']
            w = col['width']

            self.ws.set_column(start + i, start + i, w)
            if fmt not in self.formats.keys():
                self._create_numerical_format(fmt)

    def _create_formats(self) -> None:
        """Prepare the basic formats that will be used.

        These are the following:
            - title: used for all section titles
            - parameter name: used for the name of variables in the preamble
            - parameter value: used for the preamble variables themselves
            - time: general time format HH:MM:SS for the fabrication time
            - date: general date format DD/MM/YYYY for the fabrication date

        They are inserted into a dictionary, becoming available through the
        respective abbreviated key.
        """
        wb = self.wb

        al = {'align': 'center', 'valign': 'vcenter', 'border': 1}
        titt = dict(**{'bold': True, 'text_wrap': True}, **al)

        tit_specs = {'font_color': 'white', 'bg_color': '#0D47A1'}
        title_fmt = wb.add_format(dict(**tit_specs, **titt))
        parname_fmt = wb.add_format(dict(**{'bg_color': '#BBDEFB'}, **titt))

        parval_fmt = wb.add_format(dict(**{'text_wrap': True}, **al))
        time_fmt = wb.add_format(dict(**{'num_format': 'HH:MM:SS'}, **al))
        date_fmt = wb.add_format(dict(**{'num_format': 'DD/MM/YYYY'}, **al))
        text_fmt = wb.add_format({'align': 'center', 'valign': 'vcenter'})

        self.formats = {
            'title': title_fmt,
            'parname': parname_fmt,
            'parval': parval_fmt,
            'text': text_fmt,
            'date': date_fmt,
            'time': time_fmt,
        }

    def _create_numerical_format(self, fmt_string) -> None:
        wb = self.wb
        self.formats[fmt_string] = wb.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': fmt_string})

    def _fill_spreadsheet(self):

        self.struct_data = self.struct_data[self.columns_data['tagname']]

        self.ws.set_row(7, 50)
        self.ws.set_row(2, 50)

        cols = self.columns_data
        titles = [f'{f} / {u}' if u != '' else f'{f}' for f, u in zip(cols['fullname'], cols['unit'])]
        self._add_line((7, 5), titles, fmt='title')

        for i, sdata in enumerate(self.struct_data):

            sdata = [
                (
                    s
                    if (isinstance(s, (np.int64, np.float64)) and s < 1e5)
                    or (not isinstance(s, (np.int64, np.float64)))
                    else ''
                )
                for s in sdata
            ]

            self._add_line(
                (i + 9, 5),
                sdata,
                fmt=[self.formats[f] for f in self.columns_data['format']],
            )

    def _write_header(self) -> None:

        ws = self.ws

        ws.set_column(1, 1, 15)
        ws.set_column(2, 2, 15)

        nc_f = len(self.columns_data)

        # Add the femto logo at top left of spreadsheet
        path_logo = Path(fpath).parent / 'utils' / 'logo_excel.jpg'
        ws.insert_image('B2', path_logo, {'x_scale': 0.525, 'y_scale': 0.525})

        # Write the header and leave space for fabrication description
        ws.merge_range(1, 3, 1, nc_f + 4, 'Description', self.formats['title'])
        ws.merge_range(2, 3, 2, nc_f + 4, self.description, self.formats['parval'])

    def _write_preamble(self) -> None:

        ws = self.ws

        row = 8
        for sg, parameters in self.preamble.dict.items():
            ws.merge_range(row, 1, row, 2, sg, self.formats['title'])

            row += 1

            for tname, p in parameters.items():
                p._set_loc((row, 1))
                row += p.sz[0] + 1

            row += 2

        for sg, parameters in self.preamble.dict.items():
            for tname, p in parameters.items():

                if np.any(p.sz):
                    ws.merge_range(*p.loc, *(p.loc + p.sz), '', self.formats['parname'])

                    ws.merge_range(
                        *(p.loc + np.array([0, 1])),
                        *(p.loc + np.array([0, 1]) + p.sz),
                        '',
                        self.formats[p.fmt],
                    )

                self._add_line(p.loc, [p.n.capitalize(), p.v], fmt=['parname', p.fmt])

    def _add_line(
        self,
        start_cell: tuple[int, int],
        data: list[str],
        fmt: list[str] | None = None,
    ) -> None:
        """Add a line to the spreadsheet.

        Takes a start cell and writes a sequence of data in that and the
        following cells in the same line. Also accepts a fmt kwarg that tells
        the format of the data.

        Parameters
        ----------
        start_cell: tuple or list
            Tuple with the row and column of the starting cell, 1-indexing.

        data: list
            List of strings with the several data to write, in the order which
            they should appear, column-wise.

        fmt: list
            List of the formats to be used. Must be present in self.formats, so
            make sure to create the format prior to using it. Defaults to the
            default text properties of the spreadsheet, given at the moment of
            instantiation.

        Returns
        -------
        None.
        """
        if not isinstance(data, list):
            data = list(data)

        if fmt is None:
            fmt = len(data) * [self.wb.default_format_properties]

        elif not isinstance(fmt, list):
            fmt = len(data) * [fmt]

        fmt = [self.formats[key] if isinstance(key, str) else key for key in fmt]

        row, col = start_cell

        for i, (d, f) in enumerate(zip(data, fmt)):
            if isinstance(d, str) and d.startswith('='):
                self.ws.write_formula(row, col + i, d, f)
            else:
                self.ws.write(row, col + i, d, f)


def main() -> None:

    import numpy as np
    from itertools import product
    from femto.device import Device
    from femto.helpers import dotdict
    from femto.waveguide import Waveguide

    LS = -2
    LE = 27

    # GEOMETRICAL DATA
    PARS_WG = dotdict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )

    # G-CODE DATA
    PARS_GC = dotdict(
        filename='ottimizazzione.pgm',
        laser='CARBIDE',
        n_glass=1.4625,
        n_environment=1.000,
        samplesize=PARS_WG.samplesize,
    )

    # SPREADSHEET PARAMETERS
    PARS_SS = dotdict(
        book_name='Fabbrication_GHZ_Jack.xlsx',
        columns_names='name power speed depth int_dist yin yout obs',
    )

    powers = np.linspace(600, 800, 5)
    speeds = [20, 30, 40]
    scans = [3, 5, 7]

    all_fabb = Device(**PARS_GC)

    for i_guide, (p, v, ns) in enumerate(product(powers, speeds, scans)):

        start_pt = [LS, 2 + i_guide * 0.08, PARS_WG.depth]
        wg = Waveguide(**PARS_WG, speed=v, scan=ns)
        # wg.power = p  # Can NOT be added inside of the arguments of Waveguide
        wg.start(start_pt)
        wg.linear([LE - LS, 0, 0])
        wg.end()

        all_fabb.append(wg)

    all_fabb.xlsx(**PARS_SS, saints=True)


if __name__ == '__main__':
    main()
