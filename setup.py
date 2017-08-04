import setuptools

setuptools.setup(
    name="swdb",
    version="0.1.0",
    url="https://github.com/AllenInstitute/swdb",

    author="Allen Institute for Brain Science",
    author_email="justink@alleninstitute.org",

    description="A collaborative Python package built by participants of the Summer Workshop on the Dynamic Brain",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=['allensdk'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
