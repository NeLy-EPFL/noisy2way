from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="noisy2way",
    version="0.1",
    packages=["noisy2way"],
    entry_points={"console_scripts": ["noisy2way = noisy2way.cli:cli"]},
    author="Florian Aymanns",
    author_email="florian.ayamnns@epfl.ch",
    description="Post hoc two way alignment for noisy images from resonant galvo imaging.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/noisy2way",
    install_requires=["numpy", "docopt", "sphinx", "pytest"],
)
