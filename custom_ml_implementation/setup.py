import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="custom_ml_implementation", # Replace with your own username
    version="0.0.1",
    author="Paul Vazquez",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=['numpy','pandas', 'matplotlib'],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)