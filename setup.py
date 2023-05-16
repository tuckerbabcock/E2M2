from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('e2m2/__init__.py').read(),
)[0]

setup(name='E2M2',
      version=__version__,
      author='Tucker Babcock',
      author_email='tuckerbabcock1@gmail.com',
      url='https://github.com/tuckerbabcock/E2M2',
      license='Mozilla Public License 2.0',
      packages=[
          'e2m2',
      ],
      python_requires=">=3.8",
      install_requires=[
          'numpy>=1.21.4',
          'openmdao>=3.26.0',
      ],
      classifiers=[
        "Programming Language :: Python"
      ]
)
