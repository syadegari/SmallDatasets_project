from setuptools import setup, find_packages

setup(
    name="course_small_datasets",
    version="0.1",
    packages=find_packages(where="course_small_datasets")  # This tells setuptools to find all packages under 'src'.
)

