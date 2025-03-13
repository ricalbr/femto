from __future__ import annotations

import dataclasses

import numpy as np
from femto.helpers import dotdict
from femto.helpers import split_mask
from femto.laserpath import LaserPath
from PIL import Image


@dataclasses.dataclass(repr=False)
class RasterImage(LaserPath):
    """Class representing a laser path in the xy-plane of a b/w rastered image."""

    px_to_mm: float = 0.01  # pixel to millimeter scale conversion
    img_size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.z_init is None:
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
            raise ValueError('No image size given, unable to compute laserpath dimension.')
        else:
            return [self.px_to_mm * elem for elem in self.img_size]

    # Methods
    def image_to_path(self, img: Image.Image) -> None:
        """Convert image to path.

        The function takes an image, converts it to a boolean matrix, and create a laser path with ablation lines
        representing  only the black pixels of the image.

        :param img: Image.Image - the image to be converted to a path

        Parameters
        ----------
        img : Image.Image
            Image to convert to laserpath.

        Returns
        -------
        None
        """
        # displaying image information
        self.img_size = img.size  # update of img_size property
        print('Image opened. Displaying information..')
        print(f'Extension:\t{img.format}\nImage size:\t{img.size}\nColor mode:\t{img.mode}\n', '-' * 40)
        print(f'Laser path dimension {self.path_size[0]:.3f} by {self.path_size[1]:.3f} mm^2\n', '-' * 40)

        if img.mode != '1':
            img = img.convert('1')
        img_matrix = np.asarray(img, dtype=bool)

        x_scan = np.linspace(0, self.img_size[0] * self.px_to_mm, num=self.img_size[0], endpoint=True)
        y_scan = np.linspace(0, self.img_size[1] * self.px_to_mm, num=self.img_size[1], endpoint=True)
        z_val = self.z_init or 0.0

        # loop through all the rows of the image
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


def main() -> None:
    import matplotlib.pyplot as plt

    PARAM_RIMG = dotdict(px_to_mm=0.04, speed=1)

    im = Image.open(r'.\\utils\\logo.png')
    im.thumbnail((512, 512), Image.ANTIALIAS)

    r_img = RasterImage(**PARAM_RIMG)
    r_img.image_to_path(im)

    # Plot
    fig, ax = plt.subplots()
    x, y, *_, s = r_img.points
    for x_seg, y_seg in zip(split_mask(x, s.astype(bool)), split_mask(y, s.astype(bool))):
        ax.plot(x_seg, y_seg, '-k', linewidth=2.5)
    for x_seg, y_seg in zip(split_mask(x, ~s.astype(bool)), split_mask(y, ~s.astype(bool))):
        ax.plot(x_seg, y_seg, ':b', linewidth=0.5)
    ax.set_xlim([0, 0.04 * im.size[0]])
    ax.set_ylim([0, 0.04 * im.size[1]])
    plt.show()

    print(f'Expected writing time {r_img.fabrication_time:.3f} seconds')
    print(f'Laser path length {r_img.length:.3f} mm')


if __name__ == '__main__':
    main()
