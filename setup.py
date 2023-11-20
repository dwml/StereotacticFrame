from setuptools import setup, find_packages

setup(
    name="StereotacticFrame",
    version="0.1",
    description="""A python package to detect a stereotactic frame, used for stereotactic surgery""",
    url="https://github.com/SFNNijmegen/StereotacticFrame",
    author="Dirk W.M. Loeffen",
    author_email="dirk.loeffen@radboudumc.nl",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Issues": "https://github.com/SFNNijmegen/StereotacticFrame/issues"
    },
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["itk", "numpy", "SimpleITK"],
    extras_require={
        "dev": [
            "flake8",
            "pytest",
            "pytest-cov",
        ]
    },
    python_requires=">=3.9",
)
