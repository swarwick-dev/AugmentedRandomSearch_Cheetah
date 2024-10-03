from setuptools import setup, find_packages

setup(
    name='ai_training',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium',
        'torch',
        'tensorboard',
        'pyside6',
        'imageio',
        'moviepy',
        'mujuco'
    ],
    entry_points={
        'console_scripts': [
            'ars_train=main:main',
        ],
    },
)
