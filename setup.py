from setuptools import find_packages, setup

install_requires=[
    'torch',
    'opencv-python>=3.4.3.18',
    'numpy',
    'pillow'
    ]

setup(
    name='cvt',
    version='0.0.1',
    packages=find_packages(),
    author='jie.wang',
    author_email='jie.wang@ruc.edu.cn',
    description='Image Transformation Package',
    install_requires=install_requires
)
