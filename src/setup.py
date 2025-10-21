from setuptools import setup, find_packages

setup(
    name="ds_planner",
    version="0.1.0",
    description="A Python library for distribution substation planning",
    author="Han Li",
    author_email="hanli@lbl.gov",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "darts"
    ],
    include_package_data=True,
    python_requires=">=3.10",
    options={"build": {"egg_base": "src"}},  # Fix egg_base error
)
