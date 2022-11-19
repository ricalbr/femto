from dataclasses import dataclass

import numpy as np
from dacite import from_dict
from PIL import Image

from femto.helpers import dotdict
from femto.LaserPath import LaserPath
from femto.Parameters import RasterImageParameters
from femto.utils.GCODE_plot_colored import GCODE_plot_colored


@dataclass(kw_only=True)
class _RasterImage(LaserPath, RasterImageParameters):
    """
    Class representing an X raster laser path in the Xy plane obtained from a balck and white image.
    """

    def __post_init__(self):
        super().__post_init__()

    def __repr__(self):
        return "{cname}@{id:x}".format(cname=self.__class__.__name__, id=id(self) & 0xFFFFFF)

    # Methods
    def convert_image_to_path(self, img, display_flag=False):
        # displaing image information
        print("Image opened. Displaying information")
        print(img.format)
        print(img.size)
        print(img.mode)
        print("----------")

        self.img_size = img.size  # update of img_size property

        print("Laser path dimension {:.3f} by {:.3f} mm^2".format(self.path_size[0], self.path_size[1]))
        print("----------")

        if img.mode != "1":
            print(
                    "The program takes as input black and white images. Conversion of input image to BW with arbitrary "
                    "threshold at half of scale")
            img_BW = img.convert("1", dither=None)
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
                    [-1 * self.px_to_mm, (ii - 1) * self.px_to_mm, 0, self.speed_closed, 0, 0.5, 0, 0])  #
            # first move with closed shutter
            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
            new_GCODE_line = np.array([-1 * self.px_to_mm, ii * self.px_to_mm, 0, self.speed_pos, 0, 0.5, 0,
                                       0])  # first move with closed shutter
            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])

            shutter_state = 0
            speed = self.speed * 2
            indeces_shutter_closure = np.where(shutter_switch_array == -1)[0]
            if indeces_shutter_closure.size != 0:
                if sum(abs(shutter_switch_array)) != 0:
                    for jj in range(min(indeces_shutter_closure[-1], pixel_line.size) + 1):
                        if shutter_switch_array[jj] == 1:
                            new_GCODE_line = np.array(
                                    [jj * self.px_to_mm, ii * self.px_to_mm, 0, speed, shutter_state, 0.1, 0, 0])
                            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
                            shutter_state = 1
                            speed = self.speed
                        elif shutter_switch_array[jj] == -1:
                            new_GCODE_line = np.array(
                                    [jj * self.px_to_mm, ii * self.px_to_mm, 0, speed, shutter_state, 0.1, 0, 0])
                            GCODE_array = np.vstack([GCODE_array, new_GCODE_line])
                            shutter_state = 0
                            speed = self.speed * 2
        self.add_path(GCODE_array[:, 0], GCODE_array[:, 1], GCODE_array[:, 2], GCODE_array[:, 3], GCODE_array[:, 4])
        return GCODE_array


def RasterImage(param):
    return from_dict(data_class=_RasterImage, data=param)


def _example():
    from PIL import ImageDraw, ImageFont

    img = Image.new('L', (512, 256), color=255)
    font = ImageFont.truetype("arial.ttf", 40)
    d = ImageDraw.Draw(img)
    d.text((150, 100), "Hello World", font=font, fill=0)

    # img.show()
    R_IMG_PARAMETERS = dotdict(
            px_to_mm=0.04,
            speed=1,
    )

    r_img = RasterImage(R_IMG_PARAMETERS)
    GCODE_array = r_img.convert_image_to_path(img)

    fig_colored = GCODE_plot_colored(GCODE_array)
    fig_colored.show()

    print("Expected writing time {:.3f} seconds".format(r_img.fabrication_time))
    print("Laser path length {:.3f} mm".format(r_img.length))


if __name__ == '__main__':
    _example()
