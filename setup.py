import os
from setuptools import setup

AUTHORS = ('Janis Heuel'),

setup(
    name='cwt_denoiser',
    version='0.0.1',
    author=AUTHORS,
    author_email='janis.heuel@ruhr-uni-bochum.de',
    description=('Denoising Autoencoder to remove noise from seismic  data'),
    license='',
    url='',
    packages=[],
    install_requires=[],
    python_requires='>=3',
    long_description='',
    keywords=[],
    classifiers=[
        'Development Status :',
        'License :: ',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic ::',
        'Intended Audience :: Science/Research'
    ],
)


# Clone package PyCWT from GitHub
os.system("git clone https://github.com/regeirk/pycwt.git")

# Run Setup file of PyCWT
os.chdir('pycwt/')
os.system('python setup.py install')
# Create an empty file __init__.py in the pycwt directory, otherwise the package will not be recognised correctly.
f_init = open("__init__.py", "w")
f_init.close()

# XXX Create test!
# XXX Create train.sh for user and conda environment
