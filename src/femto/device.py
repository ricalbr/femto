from __future__ import annotations

import collections
import copy
import pathlib
from typing import Any
from typing import get_args
from typing import Union

import attrs
import dill
import plotly.graph_objects as go
from femto import logger
from femto.curves import sin
from femto.helpers import flatten
from femto.helpers import walklevel
from femto.laserpath import LaserPath
from femto.marker import Marker
from femto.spreadsheet import Spreadsheet
from femto.trench import Trench
from femto.trench import TrenchColumn
from femto.waveguide import NasuWaveguide
from femto.waveguide import Waveguide
from femto.writer import MarkerWriter
from femto.writer import NasuWriter
from femto.writer import plot2d_base_layer
from femto.writer import plot3d_base_layer
from femto.writer import TrenchWriter
from femto.writer import WaveguideWriter

# List of femto objects
types = dict(
    WG=Waveguide,
    NWG=NasuWaveguide,
    MK=Marker,
    LP=LaserPath,
    TR=Trench,
    TC=TrenchColumn,
)
femto_objects = Union[Waveguide, NasuWaveguide, Marker, Trench, TrenchColumn]

writers = {
    TrenchColumn: TrenchWriter,
    Marker: MarkerWriter,
    Waveguide: WaveguideWriter,
    NasuWaveguide: NasuWriter,
}


@attrs.define(kw_only=True, repr=False)
class Cell:
    """Cell object.

    Class representing a Cell objects, a container for femto objects within a Device object.
    """

    name: str = 'base'  #: Name of the Cell objects. The default value is ``base``.
    description: str | None = None  #: Description of the content of the Cell.

    _objs: dict[type[femto_objects], list[femto_objects]] = attrs.field(
        alias='_objs',
        factory=dict,
    )  #: Collection of femto objects in the current Cell.

    def __attrs_post_init__(self) -> None:
        self._objs = {
            TrenchColumn: [],
            Marker: [],
            Waveguide: [],
            NasuWaveguide: [],
        }
        self.name = self.name.lower()
        if ' ' in self.name:
            self.name = self.name.strip().replace(' ', '-')
        logger.info(f'Init Cell {self.name.upper()}.')

    def __repr__(self) -> str:
        return f'Cell {self.name}'

    @property
    def objects(self) -> dict[type[Any], list[Any]]:
        """Objects.

        Returns
        -------
        dict[type[femto_objects], list[femto_objects]]
            Returns the dictionary of femto objects contained in the current Cell. The keys of the dictionary are the
            type of the objects and the values are the actual instances of objects.
        """
        return self._objs

    def parse_objects(self, unparsed_objects: Any | list[Any]) -> None:
        """Parse objects.

        The function takes a list of objects and parse all of them based on their types.
        If the type of the object matches one of the types of the ``Writer`` registered in the ``Cell`` class,
        the object is added to the ``Writer._obj_list``. If not, a ``TypeError`` is raised.

        Parameters
        ----------
        unparsed_objects : list
            List of object that can be stored in a ``Cell``.

        Returns
        -------
        None.
        """

        # split the unparsed_object list based on the type of each element
        unparsed_objects = flatten([unparsed_objects])
        d = collections.defaultdict(list)
        for o in unparsed_objects:
            d[type(o)].append(o)

        # add each element to the type-matching keys in self._objs dictionary
        for k, e in d.items():
            try:
                logger.debug(f'Assign {e} to {self._objs[k]}.')
                self._objs[k].extend(e)
            except KeyError as err:
                logger.error(f'Found unexpected type {err.args}.')
                raise TypeError(f'Found unexpected type {err.args}.')

    def add(self, objs: femto_objects | tuple[femto_objects] | list[femto_objects]) -> None:
        """Add.

        The method allows to add to the current Cell either femto objects or lists containing femto objects.

        Parameters
        ----------
        objs: femto_objects | tuple[femto_objects] | list[femto_objects]
            Objects, tuple of objects or list of objects that will be added to the current Cell.

        Returns
        -------
        None.
        """
        if all(isinstance(obj, get_args(femto_objects)) for obj in flatten([objs])):
            self.parse_objects(objs)
        else:
            logger.error(
                'Given objects of the wrong type. Cell objects just accept laserpath- or trench-derived objects.'
            )
            raise ValueError(
                'Given objects of the wrong type. Cell objects just accept laserpath- or trench-derived objects.'
            )


class Device:
    def __init__(self, **param: Any | None) -> None:
        self.cells_collection: dict[str, Cell] = collections.defaultdict(Cell)
        self.fig: go.Figure | None = None
        self.fabrication_time: float = 0.0

        self._param: dict[str, Any] = copy.deepcopy(param)
        self._print_angle_warning: bool = True
        self._print_base_cell_warning: bool = True
        logger.info(f'Instantiate device {self._param["filename"].rsplit(".", 1)[0].upper()}.')

    @classmethod
    def from_dict(cls, param: dict[str, Any], **kwargs: Any | None) -> Device:
        """Create an instance of the class from a dictionary.

        It takes a class and a dictionary, and returns an instance of the class with the dictionary's keys as the
        instance's attributes.

        Parameters
        ----------
        param: dict
           Dictionary mapping values to class attributes.
        kwargs: optional
           Series of keyword arguments that will be used to update the param file before the instantiation of the
           class.

        Returns
        -------
        Instance of class.
        """

        # Update parameters with kwargs
        p = copy.deepcopy(param)
        if kwargs:
            p.update(kwargs)

        logger.debug(f'Create {cls.__name__} object from dictionary.')
        return cls(**p)

    @property
    def keys(self) -> list[str]:
        """Cells keys.

        Returns
        -------
        list[str]
            Return the list of cell's keys added to current device.
        """
        return list(self.cells_collection.keys())

    def add(self, objs: Cell | list[Cell] | femto_objects | list[femto_objects]) -> None:
        """Add.

        The method allows to add to the current Cell either femto objects, cells or lists of both.
        If a cell is given, it is added to the ``self.cells_collection`` dictionary, if femto objects (or lists of femto
        objects) are given these structures will be first added to a common cell (named ``base``) and then the
        ``base`` cell is added to the cell dictionary.
        If a ``base`` cell is already present, the objects will be simply added to it.

        Parameters
        ----------
        objs: femto_objects | Cell | list[femto_objects] | list[Cell]
            Objects, tuple of objects or list of objects that will be added to the current Cell.

        Returns
        -------
        None.
        """
        objs = flatten([objs])
        for elem in objs:
            if isinstance(elem, Cell):
                self.add_cell(elem)
            elif isinstance(elem, get_args(femto_objects)):
                if self._print_base_cell_warning:
                    logger.warning('femto objects added straight to a Device will be added to a common layer: BASE.')
                    self._print_base_cell_warning = False
                self.add_to_cell(key='base', obj=elem)
            else:
                logger.error(
                    'Objects can only be Cells or other femto objects (Waveguide, Markers, etc.). '
                    f'Given {type(elem)}.'
                )
                raise TypeError(
                    'Objects can only be Cells or other femto objects (Waveguide, Markers, etc.). '
                    f'Given {type(elem)}.'
                )

    def add_cell(self, cell: Cell) -> None:
        """Add cell.

        The method adds a cell to the current Device. First it checks a Cell with the same name is not present in the
        cell list, and eventually it adds it to the Device. If already present an error is thrown.

        Parameters
        ----------
        cell: Cell
            Cell to add to the current Device.

        Returns
        -------
        None.

        Raises
        ------
        ValueError for cells with the same name.
        """
        if cell.name.lower() in self.cells_collection:
            logger.error(f'Cell ID "{cell.name}" already present in layer  dict, give another value.')
            raise KeyError(f'Cell ID "{cell.name}" already present in layer  dict, give another value.')
        self.cells_collection[cell.name] = cell

    def remove_cell(self, cell: Cell) -> None:
        """Remove cell.

        The method removes a cell to the current Device. First it checks a Cell with the same name is present in the
        cell list, and eventually it removes it from the Device.

        Parameters
        ----------
        cell: Cell
            Cell to add to the current Device.

        Returns
        -------
        None.
        """
        if cell.name.lower() not in self.cells_collection:
            logger.error(f'Cell ID "{cell.name}" not present in layer  dict, give another value.')
            raise KeyError(f'Cell ID "{cell.name}" not present in layer  dict, give another value.')
        del self.cells_collection[cell.name]

    def add_to_cell(self, key: str, obj: femto_objects | list[femto_objects]) -> None:
        """Adds a femto object to a the cell.

        Parameters
        ----------
        key: str
            ID of the cell.
        obj: femto object
            Object (or list of objects) to add to cell.

        Returns
        -------
        None.
        """
        key = key.lower()
        if key not in self.cells_collection.keys():
            self.cells_collection[key] = Cell(name=key)
        self.cells_collection[key].add(obj)

    def plot2d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        """Plot 2D.

        2D plot of all the objects stored in the ``Device`` class.
        The plot is made cell-by-cell.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.
        show_shutter_close: bool, optional
            Boolean flag to automatically show the parts written with the shutter closed. The default value is True.

        Returns
        -------
        None.
        """
        logger.info('Plotting 2D objects...')
        self.fig = go.Figure()
        for cell in self.cells_collection.values():
            logger.debug(f'2D plot of cell {cell.name.upper()}.')
            wrs = writers
            for typ, list_objs in cell.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                wr.plot2d(fig=self.fig, show_shutter_close=show_shutter_close)
        x0, y0, x1, y1 = writers[Waveguide](self._param)._get_glass_borders()
        plot2d_base_layer(fig=self.fig, x0=x0, y0=y0, x1=x1, y1=y1)
        if show:
            logger.debug('Show 2D plot.')
            self.fig.show()

        if save:
            self.save()

    def plot3d(self, show: bool = True, save: bool = False, show_shutter_close: bool = True) -> None:
        """Plot 3D.

        3D plot of all the objects stored in the ``Device`` class.
        The plot is made cell-by-cell.

        Parameters
        ----------
        show : bool, optional
            Boolean flag to automatically show the plot. The default value is True.
        save : bool, optional
            Boolean flag to automatically save the plot. The default value is False.
        show_shutter_close: bool, optional
            Boolean flag to automatically show the parts written with the shutter closed. The default value is True.

        Returns
        -------
        None.
        """
        logger.info('Plotting 3D objects...')
        self.fig = go.Figure()
        for cell in self.cells_collection.values():
            logger.debug(f'3D plot of cell {cell.name.upper()}.')
            wrs = writers
            for typ, list_objs in cell.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                wr.plot3d(fig=self.fig, show_shutter_close=show_shutter_close)
        plot3d_base_layer(self.fig)
        if show:
            logger.debug('Show 3D plot.')
            self.fig.show()
        if save:
            self.save()

    def save(self, filename: str = 'scheme.html', opt: dict[str, Any] | None = None) -> None:
        """Save figure.

        Save the plot as a file.

        Parameters
        ----------
        filename: str, optional
            Filename of the output image file. The default name is "scheme.html".
        opt : dict, optional
            Dictionary with exporting options specifications.

        Returns
        -------
        None

        See Also
        --------
        go.Figure.write_image : Method to export a plotly image.
        """

        if opt is None:
            opt = dict()
        default_opt = {'width': 1980, 'height': 1080, 'scale': 2, 'engine': 'kaleido'}
        opt = {**default_opt, **opt}

        if self.fig is None:
            return None

        fn = pathlib.Path(filename)
        if fn.suffix.lower() in ['.html', '']:
            self.fig.write_html(str(fn.with_suffix('.html')))
        else:
            self.fig.write_image(str(fn), **opt)
        logger.info(f'Plot saved to "{fn}".')

    def pgm(self, verbose: bool = False) -> None:
        """Produce PGM file.

        Create a PGM file for all the objects stored in ``Device``.
        The files are stored in folders cell-wise. The name of each folder is the same of the attribute of the
        ``Cell`` object.

        Parameters
        ----------
        verbose : bool, optional
            Boolean flag to print informations during the export operation.

        Returns
        -------
        None.
        """
        for cell in self.cells_collection.values():
            logger.info(f'Generate G-Code of cell: {cell.name.upper()}.')
            wrs = writers
            for typ, list_objs in cell.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                if cell.name.lower() == 'base' and len(self.cells_collection) == 1:
                    # objects are only stored in BASE Cell, they are exported with self.filename.
                    wr.pgm(filename=None, verbose=verbose)
                else:
                    # othterwise save the objects stored in Cell with the Cell's name.
                    wr.pgm(filename=cell.name.upper(), verbose=verbose)
                self.fabrication_time += wr.fab_time

    def export(self, export_dir: str = 'EXPORT') -> None:
        """Export objects as pickle files.

        Export all the objects stored in ``Device`` class as a `pickle` file.
        The files are stored in folders cell-wise. The name of each folder is the same of the attribute of the
        ``Cell`` object.

        Parameters
        ----------
        export_dir: str, optional
            Name of the directory inside which export objects.

        Returns
        -------
        None.
        """
        logger.info('Exporting layer objects...')
        for cell in self.cells_collection.values():
            wrs = writers
            for typ, list_objs in cell.objects.items():
                wr = wrs[typ](self._param, objects=list_objs)
                wr.export(filename=cell.name.upper(), export_root=export_dir)
        logger.info('Export completed.')

    def xlsx(self, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Generate the spreadsheet.

        Add all waveguides and markers of the ``Device`` to the spreadsheet.

        Parameters
        ----------
        metadata: dict[str, Any]
            Dictionary containing all of the metadata for the Spreadsheet file (e.g. fabrication info, etc.)

        Returns
        -------
        None.
        """

        # Case in which metadata is given as keyword argument, use it for the Spreadsheet generation
        if not metadata:
            meta = {
                'laser_name': self._param.get('laser') or '',
                'sample_name': pathlib.Path(self._param.get('filename') or '').stem,
            }
            metadata = {**self._param, **meta}

        # Fetch all objects from writers
        objs: list[Waveguide | NasuWaveguide] = []
        for layer in self.cells_collection.values():
            objs.extend(layer.objects[Waveguide])
            objs.extend(layer.objects[NasuWaveguide])

        # Generate Spreadsheet
        logger.info('Generating spreadsheet...')
        try:
            export_dir = self._param['export_dir']
        except KeyError:
            export_dir = ''
        with Spreadsheet(**kwargs, metadata=metadata, export_dir=export_dir) as S:
            S.write(objs)
        logger.info('Excel file created.')

    @classmethod
    def load_objects(
        cls, folder: str | pathlib.Path, param: dict[str, Any], level: int = 1, verbose: bool = False
    ) -> Device:
        """Load objects.

        The load_objects method loads the objects from a folder and create a ``Device`` containing those objects.
        The files are loaded and added to the Device within a Cell named after the directory of containing the objects.

        Parameters
        ----------
        folder: str | pathlib.Path
            Specify the folder where the objects are stored.
        param: dict
            Pass a dictionary of parameters to the load_objects function.
        level: int, optional
            Depth level of the directory/file tree for loading files. The default value is 1. ``level=0`` does not
            return anythin, ``level=-1`` traverse all of the subdirectories inside the ``folder``.
        verbose: bool, optional
            Flag for printing the progress of the loading process. The default values is False.

        Returns
        -------
        Device
            Instance of Device class.
        """

        dev = cls(**param)

        logger.info('Loading objects...')
        for root, _, files in walklevel(folder, level):
            if not files and pathlib.Path(root) != pathlib.Path(folder):
                logger.warning(f'No file is present in the given directory {root}.')
            for file in files:
                if verbose and file:
                    logger.info(f'Loading {file} object...')
                filename = pathlib.Path(root) / file
                with open(filename, 'rb') as f:
                    dev.add_to_cell(str(pathlib.Path(root).name), dill.load(f))

        logger.info('Loading complete.')
        return dev


def main() -> None:
    """The main function of the script."""

    from femto.marker import Marker
    from femto.trench import TrenchColumn
    from femto.waveguide import Waveguide

    # Parameters
    param_wg: dict[str, Any] = dict(speed=20, radius=25, pitch=0.080, int_dist=0.007, samplesize=(25, 3))
    param_tc: dict[str, Any] = dict(length=1.0, base_folder='', y_min=0.9, y_max=4 * 0.080 + 1.1, u=[30.0, 32.0])
    param_gc: dict[str, Any] = dict(
        filename='testCell.pgm', laser='PHAROS', new_origin=(0.5, 0.5), samplesize=(25, 3), aerotech_angle=-1.023
    )

    test = Device(**param_gc)
    waveguides = Cell(name='wgs')

    # Waveguides
    x_center = 0
    coup = [Waveguide(**param_wg) for _ in range(5)]
    for i, wg in enumerate(coup):
        wg.start([-2, i * wg.pitch + 1, 0.035])
        wg.linear([9, None, None], mode='INC')
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        x_center = wg.x[-1]
        wg.coupler(dy=(-1) ** i * wg.dy_bend, dz=0, fx=sin)
        wg.end()
        waveguides.add(wg)

    mk = Marker(**param_wg)
    mk.text('dev_01', origin=[0.5, 0.5], height=0.25)

    # Trench
    trenches = Cell(name='trenches')
    tcol = TrenchColumn(x_center=x_center, **param_tc)
    tcol.dig_from_waveguide(coup, remove=[0, 1])
    trenches.add(tcol)

    test.add(waveguides)
    test.add(mk)
    test.add(tcol)
    test.add_cell(trenches)

    test.plot2d(show=False)
    test.save('scheme.html')
    test.pgm()
    test.xlsx()
    # test.export()

    # test2 = Device.load_objects('.\\EXPORT', param_gc)
    # test2.plot2d()
    # test2.save()

    # Export
    data_xlsx = dict(
        laboratory='CAPABLE',
        samplename=pathlib.Path(param_gc['filename']).stem,
        material='Cornings Eagle-XG',
        facet='Bottom',
        thickness='1 mm',
        lasername=param_gc['laser'],
        wavelength='1030 nm',
        duration='180 fs',
        reprate='1 MHz',
        attenuator='15 %',
        preset='1',
        objective='20X WI',
    )
    test.xlsx(metadata=data_xlsx)


if __name__ == '__main__':
    main()
