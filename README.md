# StereotacticFrame

A python package to detect a stereotactic frame in MR and CT images

# Installation

Use pip to install the package:

```pip install FrameRegistration```


# Install from source

Use git to clone the repository:

```git clone github.com/dwml/FrameRegistration.git```

Use cd to get into the directory:

```cd FrameRegistration```

Use poetry to install the package. For installation instructions for poetry see their documentation [here](https://python-poetry.org/docs/). After poetry is installed, in the FrameRegistration directory, run:

```poetry install```

# Usage

The package has a command line interface that can be run using

```calculate_frame_transform image_path modality transform_path```

The *image_path* should be the path to the input image, the *modality* should be one of **MR** or **CT** and *transform_path* is an optional path of the output transform.

# Warning

> [!CAUTION]
> In no circumstances can one use this package for clinical purposes. It is not thoroughly tested on a variety of images and should therefore only be used for research purposes.