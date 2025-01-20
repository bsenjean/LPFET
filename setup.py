# Setuptools for saoovqe package
from setuptools import setup, find_packages

# Readme file as long_description:
long_description = ('===========================================\n' +
                    'Local Potential Functional Embedding Theory\n' +
                    '===========================================\n')

setup(
        name='lpfet',
        version='0.1',
        url='https://gitlab.com/bsenjean/LPFET',
        description='Local Potential Functional Embedding Theory',
        long_description=long_description,
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        include_package_data=True,

)

