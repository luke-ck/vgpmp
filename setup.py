from setuptools import setup, find_packages

requirements = (
    'gpflow>=2.2.1',
    'gpflow-sampling>=0.2',
    'numpy',
    'scipy',
    'tqdm',
)

setup(
    name='gpflow_vgpmp',
    version='0.0.1',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements
)
