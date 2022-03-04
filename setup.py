from setuptools import find_packages, setup

setup(
    name="eq",
    version="0.0.1",
    description="Earthquake forecasting with neural temporal point processes",
    packages=find_packages("."),
    zip_safe=False,
)
