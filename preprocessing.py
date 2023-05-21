from skimage import color, exposure


def increase_saturation(image, factor=1.2):
    image = color.rgb2hsv(image)
    image[:, :, 1] = image[:, :, 1] * factor
    image = color.hsv2rgb(image)
    image = image * 255
    return image.astype("uint8")

def gama_correction(image, gamma=0.8):
    image = exposure.adjust_gamma(image, gamma=gamma)
    return image