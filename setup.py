from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'A user-friendly Python library for building and training neural networks.'

setup(
    name='neuralib',
    version=VERSION,
    author='LuckMeelo',
    author_email='charmeel.vodouhe@epitech.eu',
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LuckMeelo/neuralib',
    packages=find_packages(),
    install_requires=[
        'numpy',
        # Add other dependencies required
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta' or 'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
