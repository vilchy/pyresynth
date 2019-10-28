# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyresynth',
    version='0.0.1',
    description='Package for sound resynthesis',
    long_description=readme,
    author='Artur Wilniewczyc',
    author_email='artur.wilniewczyc@gmail.com',
    url='https://github.com/vilchy/pyresynth',
    license=license,
    install_requires=[
        'matplotlib >= 2.2.4',
        'numpy >= 1.16.4',
        'scipy >= 1.2.0',
        'sounddevice >= 0.3.14',
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)


