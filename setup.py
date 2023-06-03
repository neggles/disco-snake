from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="Shimeji",
    description="Shimeji is a framework to create chatbots.",
    version="0.2.1",
    license="GPLv2",
    author="Hitomi-Team, neggles",
    url="https://github.com/neggles/Shimeji",
    packages=["shimeji"],
    install_requires=required,
)
