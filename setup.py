from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="tbsim",
    packages=[package for package in find_packages() if package.startswith("tbsim")],
    install_requires=[
        "l5kit==1.5.0",
        "numpy==1.23.4",  # need to manually update numpy version to (1.21.4) due to conflict with l5kit's requirement
        "pytorch-lightning==1.8.3.post0",
        "wandb",
        "torch==1.11",
        "torchvision==0.12.0",
        "pyemd",
        "h5py",
        "imageio-ffmpeg",
        "casadi",
        "protobuf==3.20.1", # new version might cause error
        "einops==0.6.0",
        "torchtext",  # weird pytorch-lightning dependency bug
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Traffic Behavior Simulation",
    author="NVIDIA AV Research",
    author_email="danfeix@nvidia.com",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
