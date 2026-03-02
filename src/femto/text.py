from __future__ import annotations

from typing import Any
from typing import Callable

import attr
import attrs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import shapely.affinity
import shapely.ops
from femto import logger
from femto.helpers import normalize_phase
from femto.utils import _fonts
from femto.utils._fonts import Font
from shapely import geometry

# Define custom types
nparray = npt.NDArray[np.float64]
AlignmentFunc = Callable[[nparray], float]


@attrs.define(kw_only=True)
class Alignment:
    """
    Alignment helper class.

    Given a bounding box in the form of ``((min_x, min_y), (max_x, max_y))``, this class
    returns special points of this box::

        (max_y)    top +------+------+
                       |      |      |
                       |      |      |
               center  +------+------+
                       |      |      |
                       |      |      |
        (min_y) bottom +------+------+
                  (min_x)         (max_x)
                    Left   Center   Right

    Alignment options are given as ``-`` separated tuple, allowing for combinations of
    ``left``, ``center``, ``right`` with ``bottom``, ``center``, ``top``.
    """

    def _alignment_validator(self, attribute, value: str) -> None:
        """Validator for the alignment attribute."""

        options = [option.strip() for option in value.split('-')]
        if len(options) != 2:
            logger.error('Alignment option string must be two options separated by a dash.')
            raise AttributeError('Alignment option string must be two options separated by a dash.')
        if options[0] not in self._ALIGNMENT['x']:
            logger.error(f'x-axis alignment option must be one of {self._ALIGNMENT["x"].keys()}')
            raise AttributeError(f'x-axis alignment option must be one of {self._ALIGNMENT["x"].keys()}')
        if options[1] not in self._ALIGNMENT['y']:
            logger.error(f'y-axis alignment option must be one of {self._ALIGNMENT["y"].keys()}')
            raise AttributeError(f'y-axis alignment option must be one of {self._ALIGNMENT["y"].keys()}')

    alignment: str = attrs.field(
        default='left-bottom',
        validator=[
            attrs.validators.instance_of(str),
            _alignment_validator,
        ],
    )  # : Alignment option for the text, as "x-y" string (e.g., "left-bottom").

    _ALIGNMENT: dict[str, dict[str, AlignmentFunc]] = {
        'x': {
            'left': lambda coord: float(coord[0, 0]),
            'center': lambda coord: float(np.mean(coord[:, 0])),
            'right': lambda coord: float(coord[-1, 0]),
        },
        'y': {
            'bottom': lambda coord: float(coord[0, 1]),
            'center': lambda coord: float(np.mean(coord[:, 1])),
            'top': lambda coord: float(coord[-1, 1]),
        },
    }  # : Dictionary mapping axis ("x" or "y") and alignment keywords to functions computing coordinate offsets.

    @property
    def alignment_functions(self) -> tuple[Callable[[nparray], float], Callable[[nparray], float]]:
        """Returns a 2-tuple of functions, calculating the offset coordinates for a given bounding box.

        Returns
        -------
        tuple(str, str)
            Tuple of functions.
        """
        options = self.alignment.split('-')
        return self._ALIGNMENT['x'][options[0]], self._ALIGNMENT['y'][options[1]]

    def calculate_offset(self, bbox: nparray) -> nparray:
        """
        Calculate the offset coordinates corresponding to the current text alignment for a given bounding box.

        Parameters
        ----------
        bbox : nparray
            Bounding box as a NumPy array of shape (2, 2):
            [[min_x, max_x],
             [min_y, max_y]].

        Returns
        -------
        nparray
            Array of shape (2,) containing the (x, y) offset coordinates.
        """

        bbox = np.asarray(bbox)
        alignment_fun = self.alignment_functions
        offset = np.array([-alignment_fun[i](bbox) for i in [0, 1]])
        return offset


def polygon_order(polygon: geometry.Polygon) -> float:
    """
    Compute a sorting key for a polygon based on its minimum x-coordinate.

    This function returns the minimum x-value of the polygon's bounding box.
    It can be used to sort a collection of polygons from left to right.
    In case the input is invalid or not a proper polygon, a default high value
    (999) is returned to place it at the end of a sorted list.

    Parameters
    ----------
    polygon : geometry.Polygon
        A Shapely Polygon object to compute the sorting key from.

    Returns
    -------
    float
        The minimum x-coordinate of the polygon's bounding box, or 999 if invalid.
    """
    try:
        val = polygon.bounds[0]
        return val
    except TypeError:
        return 999


@attrs.define(kw_only=True, repr=False, init=False)
class Text:
    text: str = ''  #: Text content.
    alignment_position: str = 'left-bottom'  #: Alignment option for the text.
    origin: list[float] | nparray = [1.0, 1.0]  #: Origin point (x, y) for placement of the text.
    height: float = 1.0  #: Height of the text, [mm].
    angle: float = 0.0  #: Rotation angle of the text, [deg].
    font: str = 'stencil'  #: Name of the font.
    line_spacing: float = 1.5  #: Vertical spacing between lines.
    true_bbox_alignment: bool = False  #: If True, use the true bounding box for alignment calculations.

    _alignment: Alignment = attrs.field(factory=Alignment)  #: Internal Alignment object used to compute offsets.
    _bbox: nparray | None = None  #: Bounding box of the text.
    _shapely_object: geometry.MultiPolygon | None = None  #: Shape of the text.
    _letters: list[geometry.Polygon] = attr.field(
        factory=list
    )  #: List of Polygon objects representing individual letters.
    _id: str = attrs.field(alias='_id', default='TX')  #: Internal text ID.

    def __init__(self, **kwargs: Any) -> None:
        filtered: dict[str, Any] = {
            att.name: kwargs[att.name]
            for att in self.__attrs_attrs__  # type: ignore[attr-defined]
            if att.name in kwargs
        }
        self.__attrs_init__(**filtered)  # type: ignore[attr-defined]

    def __attrs_post_init__(self) -> None:
        if np.array(self.origin).shape != (2,):
            logger.error('Origin not valid. Need an iterable of two elements.')
            raise AttributeError('Origin not valid. Need an iterable of two elements.')
        self.origin = np.array(self.origin)

        if self.height < 0:
            logger.error(f'Height must be a positive number. Given {self.height}.')
            raise AttributeError(f'Height must be a positive number. Given {self.height}.')

        if self.font not in _fonts.FONTS:
            logger.error(f'Font is {self.font} unknown, must be one of {_fonts.FONTS.keys()}')
            raise AttributeError(f'Font is {self.font} unknown, must be one of {_fonts.FONTS.keys()}')

        self.angle = normalize_phase(self.angle)
        self.line_spacing = self.height * self.line_spacing
        self._alignment = Alignment(alignment=self.alignment_position)
        self._bbox = None
        self._shapely_object = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    def _invalidate(self) -> None:
        self._bbox = None
        self._shapely_object = None

    @property
    def alignment(self) -> str:
        """
        Get the current alignment position of the text.

        Returns
        -------
        str
            The alignment string in "x-y" format (e.g., "left-bottom").
        """
        return self._alignment.alignment

    @alignment.setter
    def alignment(self, alignment_pos: str) -> None:
        """
        Set a new alignment position for the text.

        This will invalidate the cached geometry and create a new Alignment object.

        Parameters
        ----------
        alignment_pos : str
            The new alignment string in "x-y" format (e.g., "center-top").
        """
        self._invalidate()
        self._alignment = Alignment(alignment=alignment_pos)

    @property
    def bounding_box(self) -> nparray:
        """
        Get the bounding box of the text.

        The bounding box is automatically computed from the Shapely geometry if not yet available.

        Returns
        -------
        nparray
            NumPy array of shape (2,2) representing [[min_x, max_x], [min_y, max_y]].

        Raises
        ------
        RuntimeError
            If the bounding box cannot be computed.
        """
        if self._bbox is None:
            self.get_shapely_object()
            if self._bbox is None:
                raise RuntimeError('Bounding box not initialized.')
        return self._bbox

    @property
    def letters(self) -> list[geometry.Polygon]:
        """
        Get the list of individual letters as Shapely polygons.

        The letters are extracted from the Shapely MultiPolygon representation
        of the text. If not yet available, the Shapely object will be generated.

        Returns
        -------
        list[geometry.Polygon]
            List of polygons corresponding to each letter.
        """
        if not self._letters:
            self.get_shapely_object()
        return self._letters

    def get_shapely_object(self) -> geometry.MultiPolygon:
        """
        Generate and return a MultiPolygon representing the text geometry.

        The method renders the text based on the selected font, height, line spacing,
        alignment, rotation, and origin. Each character is converted into one or more
        Polygon objects and then merged. Single-letter text is converted into a
        MultiPolygon to ensure consistent access to the `.geoms` property.

        The method also updates internal cached attributes:
            - `_shapely_object` : the rendered MultiPolygon
            - `_bbox`           : the bounding box of the rendered text as a 2x2 NumPy array
            - `_letters`        : list of Polygons representing individual letters

        Returns
        -------
        geometry.MultiPolygon
            A Shapely MultiPolygon representing the entire text.

        Notes
        -----
        - If `self.text` is empty, an empty MultiPolygon is returned.
        - Alignment, translation, and rotation are applied according to the
          `self.alignment`, `self.angle`, `self.origin`, and `self.true_bbox_alignment` settings.
        - Handles special characters like newlines to position text lines correctly.

        Raises
        ------
        AssertionError
            If a character in the text is not supported by the chosen font.
        """

        if not self.text:
            self._bbox = None
            return geometry.MultiPolygon()

        if self._shapely_object:
            return self._shapely_object

        polygons = list()

        special_handling_chars = '\n'
        font: Font = _fonts.FONTS[self.font]

        # Check the text
        for char in self.text:
            assert char not in special_handling_chars or char not in font, (
                f'Character "{char}" is not supported by ' f'font "{self.font}"'
            )

        max_x = 0.0
        cursor_x, cursor_y = 0.0, 0.0
        for i, char in enumerate(self.text):
            if char == '\n':
                cursor_x, cursor_y = 0.0, cursor_y - self.line_spacing
                continue

            char_font = font[char]
            cursor_x += char_font['width'] / 2 * self.height

            for line in char_font['lines']:
                points = np.array(line).T * self.height + (cursor_x, cursor_y)
                polygons.append(geometry.Polygon(points))

            # Add kerning (space between letters)
            if i < len(self.text) - 1 and self.text[i + 1] not in special_handling_chars:
                kerning = char_font['kerning'][self.text[i + 1]]
                cursor_x += (char_font['width'] / 2 + kerning) * self.height

            max_x = max(max_x, cursor_x + char_font['width'] / 2 * self.height)

        merged_polygon = shapely.ops.unary_union(polygons)

        # Handle the alignment, translation and rotation
        if not self.true_bbox_alignment:
            bbox = np.array([[0, max_x], [cursor_y, self.height]]).T
        else:
            bbox = np.array(merged_polygon.bounds).reshape(2, 2)

        offset = self._alignment.calculate_offset(bbox)

        if not np.isclose(normalize_phase(self.angle), 0):
            aligned_text = shapely.affinity.translate(merged_polygon, *offset)
            rotated_text = shapely.affinity.rotate(aligned_text, self.angle, origin=(0.0, 0.0), use_radians=True)
            final_text = shapely.affinity.translate(rotated_text, *self.origin)
        else:
            final_text = shapely.affinity.translate(merged_polygon, *(offset + self.origin))

        # Convert single letter text to Multipolygon to read the letters using the .geoms property
        if isinstance(final_text, geometry.Polygon):
            final_text = geometry.MultiPolygon([final_text])  # wrap single Polygon
        if not isinstance(final_text, geometry.MultiPolygon):
            raise TypeError(f"Unexpected geometry type: {type(final_text)}")

        self._bbox = np.array(final_text.bounds).reshape(2, 2)
        self._shapely_object = final_text
        if isinstance(final_text, geometry.Polygon):
            final_text = geometry.MultiPolygon([final_text])

        letters = sorted(list(final_text.geoms), key=lambda polygon: polygon.bounds[0])
        self._letters.extend(letters)
        return final_text


def _example():
    text = Text(origin=[0, 0], height=0.5, text='c', alignment='left-bottom')

    _, ax = plt.subplots()
    print(list(text.get_shapely_object().geoms))
    for polygon in text.get_shapely_object().geoms:
        # polygon = polygon.buffer(-text.height * 0.025, single_sided=True)
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='r', edgecolor='black', linewidth=1)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    _example()
