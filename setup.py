# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyresynth',
    version='0.0.1',
    description='Package for sound analysis',
    long_description=readme,
    author='Artur Wilniewczyc',
    author_email='artur.wilniewczyc@gmail.com',
    url='https://github.com/vilchy/pyresynth',
    license=license,
    install_requires=[
        'matplotlib >= 3.7.1',
        'numpy >= 1.24.3',
        'scipy >= 1.10.1',
        'sounddevice >= 0.4.6',
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)
