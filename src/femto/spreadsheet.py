from __future__ import annotations

import dataclasses
import itertools
import pathlib
from types import TracebackType
from typing import Any
from typing import NamedTuple

import attrs
import numpy as np
import numpy.typing as npt
import xlsxwriter
from femto import logger
from femto.helpers import flatten
from femto.helpers import listcast
from femto.marker import Marker
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide

# Define array type
nparray = npt.NDArray[np.float64]


class ColumnData(NamedTuple):
    """Class that handles column data."""

    tagname: str  #: Tag of the data represented in the column.
    name: str  #: Name of the column.
    unit: str  #: Unit of measurement for the data in the column.
    width: str  #: With of the column cells.
    format: str  #: Formatting information for the data in the column.


@dataclasses.dataclass
class PreambleParameter:
    """Class that handles preamble parameters."""

    name: str  #: Full name.
    value: str = ''  #: Value.
    location: tuple[int, int] = (0, 0)  #: Location (1-indexing).
    format: str = 'parval'  #: Format.
    row: int | None = None  #: Row of Spreadsheet document.
    col: int | None = None  #: Column of Spreadsheet document.

    def __post_init__(self) -> None:
        """Set row and column, from the location with Excel 1-indexing."""
        self.row = self.location[0]
        self.col = self.location[1]

    def set_location(self, loc: tuple[int, int]) -> None:
        """Set location of a cell parameter."""
        self.location = loc
        self.row = loc[0]
        self.col = loc[1]


@attrs.define(kw_only=True)
class Spreadsheet:
    """Class representing the spreadsheet with all entities to fabricate."""

    columns_names: list[str] = attrs.field(factory=list)  #: List of column names for the Excel file.
    description: str = ''  #: Brief description of the fabrication for the Excel file header.
    book_name: str | pathlib.Path = 'FABRICATION.xlsx'  #: Name of the Excel file. Defaults to ``FABRICATION.xlsx``.
    sheet_name: str = 'Fabrication'  #: Name of the worksheet. Defaults to ``Fabrication``.
    font_name: str = 'DejaVu Sans Mono'  #: Font-family used in the document. Defaults to DejaVu Sans Mono.
    font_size: int = 11  #: Font-size used in the document. Defaults to 11.
    redundant_cols: bool = False  #: Flag, remove redundant columns when filling the Excel file. Defaults to True.
    new_columns: list[Any] = attrs.field(factory=list)  #: New columns for the Excel file. As default value it is empty.
    metadata: dict[str, Any] = attrs.field(factory=dict)  #: Extra preamble information. As default it is empty.
    export_dir: str = ''  #: Directory of the Excel file.

    _workbook: xlsxwriter.Workbook = None
    _worksheet: xlsxwriter.Workbook.worksheets = None
    _all_cols: list[ColumnData] = attrs.field(alias='_all_cols', factory=list)
    _preamble_data: dict[str, dict[str, Any]] = attrs.field(alias='_preamble_data', factory=dict)
    _formats: dict[str, dict[str, Any]] = attrs.field(alias='_formats', factory=dict)

    def __attrs_post_init__(self) -> None:
        # Fetch default column names
        if not self.columns_names:
            default_cols = ['name', 'power', 'speed', 'scan', 'radius', 'int_dist', 'depth', 'yin', 'yout', 'obs']
            self.columns_names = default_cols
            self.redundant_cols = False
            logger.debug(
                'Columns_names not given in Spreadsheet initialization. Will proceed with standard columns names '
                f'"{default_cols}" and activate the redundant_cols flag to deal with reddundant columns.'
            )

        # Prepend 'name' column as default one
        if 'name' not in self.columns_names:
            self.columns_names = ['name'] + self.columns_names

        # Create the workbook and worksheet
        self._workbook = xlsxwriter.Workbook(
            pathlib.Path(self.export_dir) / self.book_name,
            options={'default_format_properties': {'font_name': self.font_name, 'font_size': self.font_size}},
        )
        self._worksheet = self._workbook.add_worksheet(self.sheet_name)
        self._workbook.set_calc_mode('auto')

        # Create all the Parameters contained in the preamble_info
        # Add them to a dictionary with the key equal to their tagname
        preamble_info: dict[str, list[str]] = {
            'General': ['laboratory', 'temperature', 'humidity', 'date', 'start', 'filename'],
            'Substrate': ['material', 'facet', 'thickness'],
            'Laser Parameters': ['laser', 'wavelength', 'duration', 'reprate', 'attenuator', 'preset'],
            'Irradiation': ['objective', 'power', 'speed', 'scan', 'depth'],
        }

        preamble_data: dict[str, dict[str, PreambleParameter]] = {}
        for k, section in preamble_info.items():
            preamble_data[k] = {
                field: PreambleParameter(name=field, value=self.metadata.get(field) or '') for field in section
            }

        preamble_data['General']['start'].format = 'time'
        preamble_data['General']['date'].format = 'date'

        self._preamble_data = preamble_data
        self._all_cols = self.generate_all_cols_data()
        self._formats = self._default_formats()

    def __enter__(self) -> Spreadsheet:
        """Context manager entry.

        Can be use like:
        ::
            with Spreadsheet(**SS_PARAMETERS) as ss:
                ...
        """
        self.header()
        self.fabbrication_info()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def header(self, desc_size: int = 8) -> None:
        """Header.

        Write the header info to xlsx file.

        Parameters
        ----------
        desc_size: int, optional
            Number of cells to merge for the decription field, the default value is 8.

        Returns
        -------
        None.
        """

        # Set columns properties
        self._worksheet.set_column(first_col=1, last_col=1, width=15)
        self._worksheet.set_column(first_col=2, last_col=2, width=25)
        self._worksheet.set_column(first_col=3, last_col=3, width=5)

        # Add the femto logo at top left of spreadsheet
        path_logo = pathlib.Path(__file__).parent / 'utils' / 'logo_excel.png'
        self._worksheet.insert_image('B2', path_logo, options={'x_scale': 0.33, 'y_scale': 0.33})

        # Write the header and leave space for fabrication description
        self._worksheet.set_row(row=2, height=50)
        self._worksheet.merge_range(
            first_row=1,
            first_col=5,
            last_row=1,
            last_col=desc_size + 5,
            data='Description',
            cell_format=self._workbook.add_format(self._formats['title']),
        )
        self._worksheet.merge_range(
            first_row=2,
            first_col=5,
            last_row=2,
            last_col=desc_size + 5,
            data=self.description,
            cell_format=self._workbook.add_format(self._formats['parval']),
        )

    def fabbrication_info(self, row: int = 8) -> None:
        """Fabrication information.

        Write the fabrication info to xlsx file.

        Parameters
        ----------
        row: int, optional
            First row of the preamble information table, the values are 0-indexed. The default value is 8.

        Returns
        -------
        None.
        """

        for pre_title, parameters in self._preamble_data.items():
            self._worksheet.merge_range(
                first_row=row,
                first_col=1,
                last_row=row,
                last_col=2,
                data=pre_title,
                cell_format=self._workbook.add_format(self._formats['title']),
            )
            row += 1

            for tname, p in parameters.items():
                p.set_location((row, 1))
                self.add_line(row=p.row, col=p.col, data=[p.name.capitalize(), p.value], fmt=['parname', p.format])
                row += 1
            row += 2

    def close(self) -> None:
        """Close the workbook."""
        self._workbook.close()

    def write(self, obj_list: list[Waveguide | NasuWaveguide], start: int = 5) -> None:
        """Write the structures to the spreadsheet.

        Builds the structures list, containing all the required information about the structures to fabricate. Then,
        prepares the columns in the spreadsheet, setting their width according to the ``columns.txt`` file. Finally,
        fills the spreadsheet with the contents of the structures list, one structure per line, and creates the
        preamble, which is the general information to the left of individual structure information.

        Returns
        -------
        None.

        """

        if not all([isinstance(obj, (Waveguide, NasuWaveguide)) for obj in obj_list]):
            logger.error(
                'Objects for Spreasheet files must be of type Waveguide or NasuWaveguide.'
                f'Given {[type(obj) for obj in obj_list]}.'
            )
            raise ValueError(
                'Objects for Spreasheet files must be of type Waveguide, NasuWaveguide.'
                f'Given {[type(obj) for obj in obj_list]}.'
            )
        cols_info, numerical_data = self._extract_data(obj_list)

        if not cols_info or numerical_data.size == 0:
            logger.debug('No column names or numerical data.')
            return

        # Set the correct width and create the data format.
        for i, col in enumerate(cols_info):
            self._worksheet.set_column(first_col=start + i, last_col=start + i, width=int(col.width))
            fmt = col.format
            if fmt not in self._formats.keys():
                self._formats[str(fmt)] = {'align': 'center', 'valign': 'vcenter', 'num_format': str(fmt)}

        # Fill SpreadSheet
        # Titles
        titles: list[str | int | float] = [
            f'{col.name}\n[{col.unit}]' if col.unit != '' else f'{col.name}' for col in cols_info
        ]
        self._worksheet.set_row(row=7, height=50)
        self.add_line(row=7, col=5, data=titles, fmt=len(titles) * ['title'])

        # Data
        for i, sdata in enumerate(numerical_data):
            sdata = [
                (
                    s
                    if (isinstance(s, (np.int64, np.float64)) and s < 1e5)
                    or (not isinstance(s, (np.int64, np.float64)))
                    else ''
                )
                for s in sdata
            ]
            self.add_line(row=i + 8, col=5, data=sdata, fmt=[col.format for col in cols_info])

        # Add autofilters
        self._worksheet.autofilter(
            first_row=7,
            first_col=5,
            last_row=7 + len(numerical_data) - 1,
            last_col=5 + len(titles) - 1,
        )

    def generate_all_cols_data(self) -> list[ColumnData]:
        """Create the available columns array from a file.

        Fetches all data from the ``utils/spreadsheet_columns.txt`` file and creates a list with the information for
        all possible columns. The user can only select columns to add to the spreadsheet throught their tagname,
        which must be in the first column of the txt document.
        """

        default_cols = []
        with open(pathlib.Path(__file__).parent / 'utils' / 'spreadsheet_columns.txt') as f:
            next(f)
            for line in f:
                tag, name, unit, width, fmt = line.strip().split(', ')
                default_cols.append(ColumnData(tagname=tag, name=name, unit=unit, width=width, format=fmt))

        if self.new_columns:
            new_cols = []
            for elem in self.new_columns:
                try:
                    tag, name, unit, width, fmt = elem
                    new_cols.append(ColumnData(tagname=tag, name=name, unit=unit, width=width, format=fmt))
                except ValueError:
                    logger.error('Wrong format. Columns elements must be a tuple: (tag, name, unit, width, fmt).')
                    raise ValueError('Wrong format. Columns elements must be a tuple: (tag, name, unit, width, fmt).')

            # Merge default columns and new columns keeping the values of the latter ones.
            def_dict = dict(zip([col.tagname for col in default_cols], default_cols))
            new_dict = dict(zip([col.tagname for col in new_cols], new_cols))
            def_dict.update(new_dict)
            # Keep 'Observation' column as last one
            obs_entry = def_dict.pop('obs')
            def_dict['obs'] = obs_entry

            # Update the columns names and return
            self.columns_names = list(def_dict.keys())
            return list(def_dict.values())
        return default_cols

    def add_line(self, row: int, col: int, data: list[str | int | float], fmt: list[str] | None = None) -> None:
        """Add a line to the spreadsheet.

        Takes a start cell and writes a sequence of data in that and the following cells in the same line. Also
        accepts a fmt kwarg that tells the cell_fmt of the data.

        Parameters
        ----------
        row: int
            Row of the starting cell, 1-indexing.
        col: int
            Column of the starting cell, 1-indexing.
        data: str
            List of strings with the several data to write, in the order which they should appear, column-wise.
        fmt: str, optional
            list of the formatting options to be used. They must be present in self._formats, so make sure to create the
            format prior to using it. defaults to the default text properties of the spreadsheet, given at the moment of
            instantiation.

        Returns
        -------
        None.
        """

        data = listcast(data)
        if fmt is None:
            cell_fmt = [self._workbook.default_format_properties]
        else:
            cell_fmt = []
            for key in listcast(fmt):
                try:
                    cell_fmt.append(self._formats[str(key).lower()])
                except KeyError:
                    logger.error(f'Found unknown key for formatting options. Given key {key}.')
                    raise KeyError(f'Found unknown key for formatting options. Given key {key}.')

        if len(data) < len(cell_fmt):
            logger.error('The number of formatting options is bigger than the number of data to write in xlsx file.')
            raise ValueError(
                'The number of formatting options is bigger than the number of data to write in xlsx file.'
            )

        for i, (data_val, format_opts) in enumerate(
            itertools.zip_longest(data, cell_fmt, fillvalue=self._workbook.default_format_properties)
        ):
            f = self._workbook.add_format(format_opts)
            if isinstance(data_val, str) and data_val.startswith('='):
                self._worksheet.write_formula(row, col + i, data_val, f)
            else:
                self._worksheet.write(row, col + i, data_val, f)

    def _extract_data(
        self,
        structures: list[Waveguide | NasuWaveguide] | None = None,
    ) -> tuple[list[ColumnData], npt.NDArray[Any]]:
        """Build a table with all of the structures.

        The table has as lines the several structures, and for each of them, all of the fields given as columns_names
        as column data. Determines the columns that will be effectively added to the sheet, according to the specified
        redundant_cols and static_preamble.

        Parameters
        ----------
        structures: list
            Contains the waveguides and the markers to be added to the table.

        Returns
        -------
        tuple[list[ColumnData], np.ndarray]
            Returns a tuple with the informations of the columns to export to the spreadsheet file and a numpy array
            with all the numerical data relative to the columns.
        """

        def coords(x: Waveguide | Marker) -> dict[type[Waveguide] | type[Marker], dict[str, float]]:
            """Input/output y-coordinates of Waveguide and Marker objects."""
            return {
                Waveguide: {
                    'yin': x.path3d[1][0],
                    'yout': x.path3d[1][-1],
                },
                Marker: {
                    'yin': (max(x.path3d[0][:]) + min(x.path3d[0][:])) / 2,
                    'yout': (max(x.path3d[1][:]) + min(x.path3d[1][:])) / 2,
                },
            }

        structures = flatten(listcast(structures))
        if not structures:
            # Base case, if the structures list is empty, return no columns info and no numerical data
            return [], np.array([])

        # Select with tagname
        column_name_info = [col for col in self._all_cols if col.tagname in self.columns_names]
        dtype = [(t, self._dtype(t)) for t in self.columns_names]

        # Create data table
        table_lines = np.zeros_like(structures, dtype=dtype)

        # Extract all data from structures (either attributes of metadata)
        for i, ent in enumerate(structures):
            data_line: list[float | str | None] = []
            for tag, typ in dtype:
                if tag in ['yin', 'yout']:
                    data_line.append(coords(ent)[type(ent)][tag])
                else:
                    fallback: float | str = 1.1e5 if typ in [np.float64, np.int64] else ''
                    item: float | str = getattr(ent, tag, None) or ent.metadata.get(tag) or fallback
                    data_line.append(item)
            table_lines[i] = tuple(data_line)

        # Select
        keep = []
        ignored_fields = []
        for i, (t, typ) in enumerate(dtype):
            # Keep string-type columns (names, obs,...)
            # Ignore redundant columns (same value on all the rows) if explicitly requested
            if typ in [np.int64, np.float64]:
                if not self.redundant_cols and len(set(table_lines[t])) == 1 and table_lines.shape[0] != 1:
                    ignored_fields.append(t)
                    continue
                # Ignore columns with all the values greater than 1e5
                elif np.all(table_lines[t] >= 1e5):
                    ignored_fields.append(t)
                    continue
            keep.append(t)

        if ignored_fields:
            fields_left_out = ', '.join(ignored_fields)
            logger.debug(
                f'For all entities, the fields {fields_left_out} were not defined, they will not be shown as columns.'
            )
        info = [col for col in column_name_info if col.tagname in keep]
        return info, table_lines[keep]

    @staticmethod
    def _default_formats() -> dict[str, dict[str, Any]]:
        """Prepare the basic formats that will be used.

        These are the following:
            - title: used for all section titles
            - parameter name: used for the name of variables in the preamble
            - parameter value: used for the preamble variables themselves
            - time: general time format HH:MM:SS for the fabrication time
            - date: general date format DD/MM/YYYY for the fabrication date

        They are inserted into a dictionary, becoming available through the respective abbreviated key.
        """

        al: dict[str, Any] = {'align': 'center', 'valign': 'vcenter', 'border': 1}
        titt: dict[str, Any] = {**{'bold': True, 'text_wrap': True}, **al}
        tit_specs: dict[str, Any] = {'font_color': 'white', 'bg_color': '#6C5B7B'}

        return {
            'title': {**tit_specs, **titt},
            'parname': {**{'bg_color': '#D5CABD'}, **titt},
            'parval': {**{'text_wrap': True}, **al},
            'text': {'align': 'center', 'valign': 'vcenter'},
            'date': {**{'num_format': 'DD/MM/YYYY'}, **al},
            'time': {**{'num_format': 'HH:MM:SS'}, **al},
        }

    def _dtype(self, tag: str) -> str | type:
        """Return the data type corresponding to a give column tagname.

        The data type is determined in the ``columns.txt`` file, under the column named ``format``. The dtypes are
        assigned according to the following possibilities for that field:

        - ``text`` or ``title`` -> dtype: 20-character str
        - sequence of zeros with a dot somewhere -> dtype: float
        - sequence of zeros with no dot -> dtype: int

        Parameters
        ----------
        tag: str
            The tagname of the column type. Must be contained in the ``columns.txt`` file under the ``utils`` folder.
            Not case sensitive.

        Returns
        -------
        type
            Type of the data.

        """
        tag_columns = dict(zip([col.tagname for col in self._all_cols], self._all_cols))

        if tag_columns[tag].format in ['text', 'title']:
            return 'U20'
        elif '.' in tag_columns[tag].format:
            return np.float64
        else:
            return np.int64


def main() -> None:
    """The main function of the script."""
    from itertools import product

    from addict import Dict as ddict

    from femto.waveguide import Waveguide

    l_start = -2
    l_end = 27

    # GEOMETRICAL DATA
    pars_wg = ddict(
        speed_closed=40,
        radius=40,
        depth=-0.860,
        pitch=0.080,
        samplesize=(25, 25),
    )

    # SPREADSHEET PARAMETERS
    pars_ss = ddict(
        book_name='Fabbrication.xlsx',
        columns_names=['name', 'power', 'speed', 'scan', 'depth', 'int_dist', 'yin', 'yout', 'obs'],
        redundant_cols=False,
    )

    powers = np.linspace(600, 800, 5)
    speeds = [20, 30, 40]
    scans = [3, 5, 7]

    all_fabb: list[Waveguide] = []

    for i_guide, (p, v, ns) in enumerate(product(powers, speeds, scans)):
        start_pt = [l_start, 2 + i_guide * 0.08, pars_wg.depth]
        wg = Waveguide(**pars_wg, speed=v, scan=ns, metadata={'power': p})
        wg.start(start_pt)
        wg.linear([l_end - l_start, 0, 0])
        wg.end()

        all_fabb.append(wg)

    with Spreadsheet(**pars_ss) as S:
        S.write(all_fabb)


if __name__ == '__main__':
    main()
