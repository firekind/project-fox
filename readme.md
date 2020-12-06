<div align="center">
    <h1>
        Project Fox <br/>
        <img src="res/fox.png">
    </h1>
</div>

<p align="center">
    <a href="https://firekind.github.io/project-fox">
        <img src="https://img.shields.io/website-up-down-green-red/http/firekind.github.io/project-fox"/>
    </a>
    <a href="https://github.com/firekind/project-fox/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/firekind/project-fox" />
    </a>
    <img src="https://img.shields.io/badge/maintainer-firekind-red"/>
    <img src="https://img.shields.io/github/languages/top/firekind/project-fox" />
    <img src="https://img.shields.io/github/languages/count/firekind/project-fox">
</p>

A object detection, depth estimation and plane surface detection model, trained on a custom dataset containing images of hardhats, vests, masks and boots. Look at the documentation for more details.

## Setup

clone this repo using

```
$ git clone https://github.com/firekind/project-fox --recurse-submodules
```

and refer to the jupyter notebooks in the root directory of the repo to find out how to train. Make sure the required packages are installed from the `environment.yml` file.

## Dataset setup

The directory structure of the dataset is like this:

```
data
├── images/
├── midas
│   ├── custom.data
│   └── depth
├── planercnn
│   ├── camera.txt
│   ├── custom.data
│   ├── masks
│   └── parameters
├── yolo
│   ├── custom.data
│   ├── custom.names
│   ├── images.shapes
│   ├── labels
│   ├── labels.npy
│   ├── test.shapes
│   ├── test.txt
│   ├── train.shapes
│   └── train.txt

```

the `images` folder contains all the input images, `midas/depth` folder contains the depth image ground truths for the MiDAS portion of the model. The file `midas/custom.data` is something like this:

```
depth=./depth
images=../images
```

The `planercnn/masks` folder contain the plane segmentation images and the `planercnn/parameters` folder contain the plane parameters files (refer to [this notebook](./test/planercnn-dataset-generation.ipynb) file for more details). `camera.txt` is the same as the one used in the inference of vanilla PlaneRCNN. the `planercnn/custom.data` file is something like this:

```
masks=./masks
parameters=./parameters
images=../images
camera=./camera.txt
```

the `yolo` folder is structured the same way as mentioned in the [YoloV3](https://github.com/ultralytics/yolov3) repo.

## Documentation and journey
Have a look at the journey here: https://firekind.github.io/project-fox