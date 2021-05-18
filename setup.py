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
