import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-tools", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="Python tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikolayilyin/python_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)