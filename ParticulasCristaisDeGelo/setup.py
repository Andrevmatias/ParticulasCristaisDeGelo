from setuptools import setup, find_packages

setup(name='trainer',
      version='0.2',
      packages=find_packages(),
      install_requires=[
          'keras',
          'h5py',
          'sklearn',
          'matplotlib'
      ],
zip_safe=False)