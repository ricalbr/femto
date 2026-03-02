from __future__ import annotations

import math
from typing import Any

import attrs
import numpy as np
from femto import logger
from femto.helpers import split_mask
from femto.laserpath import LaserPath
from PIL import Image


@attrs.define(kw_only=True, repr=False, init=False)
class RasterImage(LaserPath):
    """Class representing a laser path in the xy-plane of a b/w rastered image."""

    px_to_mm: float = 0.010  #: Pixel to millimeter scale convertion.
    img_size: tuple[int, int] = (0, 0)  #: Number of pixels in x and y direction of the image.

    _id: str = attrs.field(alias='_id', default='RI')  #: RasterImage ID.

    def __init__(self, **kwargs: Any) -> None:
        filtered: dict[str, Any] = {
            att.name: kwargs[att.name]
            for att in self.__attrs_attrs__  # type: ignore[attr-defined]
            if att.name in kwargs
        }
        self.__attrs_init__(**filtered)  # type: ignore[attr-defined]

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if math.isnan(self.z_init):
            self.z_init = 0.0

    @property
    def path_size(self) -> list[float]:
        """Path size.

        Returns
        -------
        list(float)
            (`x`, `y`) size of the laser path [mm].
        """
        if not all(self.img_size):  # check if img_size is non-zero
            logger.error('No image size given, unable to compute laserpath dimension.')
            raise ValueError('No image size given, unable to compute laserpath dimension.')
        else:
            logger.debug('Return list of path_size.')
            return [self.px_to_mm * elem for elem in self.img_size]

    # Methods
    def image_to_path(self, img: Image.Image) -> None:
        """Convert image to path.

        The function takes an image, converts it to a boolean matrix, and create a laser path with ablation lines
        representing  only the black pixels of the image.

        Parameters
        ----------
        img : Image.Image
            Image to convert to laserpath.

        Returns
        -------
        None.
        """
        # displaying image information
        self.img_size = img.size  # update of img_size property
        logger.info('Image opened. Displaying information..')
        logger.info(f'Extension:\t{img.format}\nImage size:\t{img.size}\nColor mode:\t{img.mode}\n')
        logger.info(f'Laser path dimension {self.path_size[0]:.3f} by {self.path_size[1]:.3f} mm^2\n')

        if img.mode != '1':
            img = img.convert('1')
        img_matrix = np.asarray(img, dtype=bool)

        x_scan = np.linspace(0, self.img_size[0] * self.px_to_mm, num=self.img_size[0], endpoint=True)
        y_scan = np.linspace(0, self.img_size[1] * self.px_to_mm, num=self.img_size[1], endpoint=True)
        z_val = self.z_init or 0.0

        # loop through all the rows of the image
        logger.debug('Convert image to path...')
        for row, y_val in zip(img_matrix, y_scan):
            x_open_shutter = split_mask(x_scan, ~row)

            # continue if no points with open shutter are found
            if not x_open_shutter:
                continue

            # add a path for each sub-split with open shutter
            for x_split in x_open_shutter:
                x_row = np.array([x_split[0], x_split[0], x_split[-1], x_split[-1], x_split[0]], dtype=np.float64)
                y_row = y_val * np.ones_like(x_row, dtype=np.float64)
                z_row = z_val * np.ones_like(x_row, dtype=np.float64)
                f_row = np.array(
                    [self.speed_closed, self.speed, self.speed, self.speed_closed, self.speed_closed], dtype=np.float64
                )
                s_row = np.array([0, 1, 1, 0, 0], dtype=int)

                self.add_path(x_row, y_row, z_row, f_row, s_row)
        logger.debug('Image converted.')


def main() -> None:
    """The main function of the script."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    param_rimg = dict(px_to_mm=0.04, speed=1)
    logo_path = Path('./utils/logo.png')

    im = Image.open(logo_path)
    im.thumbnail((512, 512))
    # im.thumbnail((512, 512), Image.ANTIALIAS)

    r_img = RasterImage(**param_rimg)
    r_img.image_to_path(im)

    # Plot
    _, ax = plt.subplots()
    x, y, *_, s = r_img.points
    for x_seg, y_seg in zip(split_mask(x, s.astype(bool)), split_mask(y, s.astype(bool))):
        ax.plot(x_seg, y_seg, '-k', linewidth=2.5)
    for x_seg, y_seg in zip(split_mask(x, ~s.astype(bool)), split_mask(y, ~s.astype(bool))):
        ax.plot(x_seg, y_seg, ':b', linewidth=0.5)
    ax.set_xlim((0, 0.04 * im.size[0]))
    ax.set_ylim((0, 0.04 * im.size[1]))
    plt.show()

    print(f'Expected writing time {r_img.fabrication_time:.3f} seconds')
    print(f'Laser path length {r_img.length:.3f} mm')


if __name__ == '__main__':
    main()
