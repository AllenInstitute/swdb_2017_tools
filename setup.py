from setuptools import setup, find_packages
import swdb2017

# Add setuptools boilerplate
setup(
    name='swdb2017',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=swdb2017.__version__,

    description='student contributed tools from swdb2017',
    long_description='Student tools from Summer Workshop on the Dynamic Brain 2017',

    # The project's main homepage.
    url='https://github.com/AllenInstitute/swdb_2017_tools',

    # Author details
    author='Allen Institute',
    author_email='',

    # Choose your license
    license='MIT',
    packages=find_packages()
)
