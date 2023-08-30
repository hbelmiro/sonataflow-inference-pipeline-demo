from PIL import Image


def overlay_images(base_image_path, overlay_image_path, position):
    # Open the base and overlay images
    base_img = Image.open(base_image_path)
    overlay_img = Image.open(overlay_image_path)

    # Convert the overlay image to RGBA mode
    overlay_img = overlay_img.convert('RGBA')

    # Overlay the image over the base image
    base_img.paste(overlay_img, position, overlay_img)

    file_name = 'overlayed_image.jpg'

    base_img.save(file_name)

    return file_name


# # Example usage:
# base_image_path = 'test_image.jpg'
# overlay_image_path = 'test_image_house.jpg'
# position = (200, 100)
#
# result_image = overlay_images(base_image_path, overlay_image_path, position)
#
# # Save the resulting image
# result_image.save('overlayed_image.jpg')
