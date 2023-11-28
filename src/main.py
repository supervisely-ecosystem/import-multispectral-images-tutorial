import cv2
import os
import tifffile
import supervisely as sly
import nrrd
from dotenv import load_dotenv

team_id = 448
workspace_id = 690
# Enter your team and workspace ids here.
# Learn how to copy ids: https://developer.supervisely.com/getting-started/environment-variables

load_dotenv(os.path.expanduser("~/supervisely.env"))
# Learn more about the supervisely.env file:
# https://developer.supervisely.com/getting-started/basics-of-authentication#use-.env-file-recommended

api = sly.Api.from_env()

# Create new project and dataset.
project = api.project.create(workspace_id, "Multispectral images", change_name_if_conflict=True)
dataset = api.dataset.create(project.id, "ds0")

# Set multispectral settings for project to enable images grouping and synchnonized view.
api.project.set_multispectral_settings(project.id)

# Example 1: Working with RGB png image.
image_name = "demo1.png"
image = cv2.imread(f"demo_data/{image_name}")

# Extract channels as 2d numpy arrays: channels = [a, b, c]
channels = [image[:, :, i] for i in range(image.shape[2])]

image_infos = api.image.upload_multispectral(dataset.id, image_name, channels)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")


# Example 2: Working with multi-channel tif image.
image_name = "demo2.tif"
image = tifffile.imread(f"demo_data/{image_name}")

# Extract channels as 2d numpy arrays: channels = [a, b, c, d, e, f]
channels = [image[:, :, i] for i in range(image.shape[2])]

image_infos = api.image.upload_multispectral(dataset.id, image_name, channels)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")

# # Example 3: Working with nrrd image.
# # Support images with high color depth (bigger than 8 bit per pixel, for example 16, 32, etc...)
image_name = "demo3.nrrd"
image, header = nrrd.read(f"demo_data/{image_name}")

# Extract channels as 2d numpy arrays: channels = [a, b, c, d, e, f]
channels = [image[:, :, i] for i in range(image.shape[2])]

image_infos = api.image.upload_multispectral(dataset.id, image_name, channels)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")


# Example 4: Uploading a pair of images (RGB and thermal) without splitting them into channels.
image_name = "demo4.png"
images = ["demo_data/demo4-rgb.png", "demo_data/demo4-thermal.png"]

image_infos = api.image.upload_multispectral(dataset.id, image_name, rgb_images=images)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")

# Example 5: Upload RGB image, it's channels and a depth image.
image_name = "demo5.png"
images = ["demo_data/demo5-rgb.png", "demo_data/demo5-depths.png"]

image = cv2.imread(images[0])

# Extract channels as 2d numpy arrays: channels = [a, b, c]
channels = [image[:, :, i] for i in range(image.shape[2])]

image_infos = api.image.upload_multispectral(dataset.id, image_name, channels, images)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")

# Example 6: Uploading grayscale image and UV image as a pair.
image_name = "demo6.png"
images = ["demo_data/demo6-grayscale.png", "demo_data/demo6-uv.png"]

image_infos = api.image.upload_multispectral(dataset.id, image_name, rgb_images=images)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")

# Example 7: Uploading RGB image, thermal image and thermal image channels.
image_name = "demo7.png"
images = ["demo_data/demo7-rgb.png", "demo_data/demo7-thermal.png"]

image = cv2.imread(images[1])

# Extract channels as 2d numpy arrays: channels = [a, b, c]
channels = [image[:, :, i] for i in range(image.shape[2])]

image_infos = api.image.upload_multispectral(dataset.id, image_name, channels, images)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")

# Example 8: Uploading RGB image, and two MRI images.
image_name = "demo8.png"
images = ["demo_data/demo8-rgb.png", "demo_data/demo8-mri1.png", "demo_data/demo8-mri2.png"]

image_infos = api.image.upload_multispectral(dataset.id, image_name, rgb_images=images)
print(f"Successfully uploaded {len(image_infos)} images (channels) from {image_name}")
