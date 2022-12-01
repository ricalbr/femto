from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from femto.helpers import Dotdict
from femto.laserpath import LaserPath
from femto.utils.GCODE_plot_colored import GCODE_plot_colored
from PIL import Image


@dataclass(repr=False)
class RasterImage(LaserPath):
    """
    Class representing an X raster laser path in the Xy plane obtained from a balck and white image.
    """

    px_to_mm: float = 0.01  # pixel to millimeter scale when converting image to laser path
    img_size: tuple[int, int] = (0, 0)

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return f'{self.__class__.__name__}@{id(self) & 0xFFFFFF:x}'

    @property
    def path_size(self):
        if not all(self.img_size):  # check if img_size is None
            raise ValueError('No image size given, unable to compute laserpath dimension')
        else:
            return tuple(self.px_to_mm * elem for elem in self.img_size)

    # Methods
    def convert_image_to_path(self, img, display_flag=False):
        # displaing image information
        print('Image opened. Displaying information')
        print(img.format)
        print(img.size)
        print(img.mode)
        print('----------')

        self.img_size = img.size  # update of img_size property

        print(f'Laser path dimension {self.path_size[0]:.3f} by {self.path_size[1]:.3f} mm^2')
        print('----------')

        if img.mode != '1':
            print(
                'The program takes as input black and white images. Conversion of input image to BW with arbitrary '
                'threshold at half of scale'
            )
            img_BW = img.convert('1', dither=None)
            if display_flag:
                img_BW.show()

        data = np.asarray(img_BW)
        GCODE_array = np.array([0, 0, 0, 2, 0, 0.1, 0, 0])  # initialization  of the GCODE array

        for ii in range(data.shape[0]):
            pixel_line = data[ii, :]
            pixel_line = np.append(pixel_line, False)  # appending a final balck pixel, to ensure that the shutter is
            # closed at the end
            pixel_line_shifted = np.zeros(pixel_line.size)
            pixel_line_shifted[1:] = pixel_line[0:-1]
            shutter_switch_array = pixel_line - pixel_line_shifted

            new_GCODE_line = np.array(
                [-1 * self.px_to_mm, (ii - 1) * self.px_to_mm, 0, self.speed_closed, 0, 0.5, 0, 0]
            )
            # first move with closed shutter
            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
            new_GCODE_line = np.array(
                [-1 * self.px_to_mm, ii * self.px_to_mm, 0, self.speed_pos, 0, 0.5, 0, 0]
            )  # first move with closed shutter
            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])

            shutter_state = 0
            speed = self.speed * 2
            indeces_shutter_closure = np.where(shutter_switch_array == -1)[0]
            if indeces_shutter_closure.size != 0:
                if sum(abs(shutter_switch_array)) != 0:
                    for jj in range(min(indeces_shutter_closure[-1], pixel_line.size) + 1):
                        if shutter_switch_array[jj] == 1:
                            new_GCODE_line = np.array(
                                [
                                    jj * self.px_to_mm,
                                    ii * self.px_to_mm,
                                    0,
                                    speed,
                                    shutter_state,
                                    0.1,
                                    0,
                                    0,
                                ]
                            )
                            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
                            shutter_state = 1
                            speed = self.speed
                        elif shutter_switch_array[jj] == -1:
                            new_GCODE_line = np.array(
                                [
                                    jj * self.px_to_mm,
                                    ii * self.px_to_mm,
                                    0,
                                    speed,
                                    shutter_state,
                                    0.1,
                                    0,
                                    0,
                                ]
                            )
                            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
                            shutter_state = 0
                            speed = self.speed * 2
        self.add_path(
            GCODE_array[:, 0],
            GCODE_array[:, 1],
            GCODE_array[:, 2],
            GCODE_array[:, 3],
            GCODE_array[:, 4],
        )
        return GCODE_array


def _example():
    from PIL import ImageDraw, ImageFont

    img = Image.new('L', (512, 256), color=255)
    font = ImageFont.truetype('arial.ttf', 40)
    d = ImageDraw.Draw(img)
    d.text((150, 100), 'Hello World', font=font, fill=0)

    # img.show()
    R_IMG_PARAMETERS = Dotdict(
        px_to_mm=0.04,
        speed=1,
    )

    r_img = RasterImage(**R_IMG_PARAMETERS)
    GCODE_array = r_img.convert_image_to_path(img)

    fig_colored = GCODE_plot_colored(GCODE_array)
    fig_colored.show()

    print(f'Expected writing time {r_img.fabrication_time:.3f} seconds')
    print(f'Laser path length {r_img.length:.3f} mm')


if __name__ == '__main__':
    _example()