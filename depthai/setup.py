from setuptools import setup, find_packages

setup(
    name='blazeposedepthai',
    packages=find_packages(),
    package_data={
        'blazeposedepthai': ['models/*']
    },
    include_package_data=True,
)