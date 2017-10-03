# Motivation
Devastating hurricanes are occurring with higher frequency due to global climate change.  After a hurricane, roads are often flooded or washed out, making them treacherous for motorists.

Using state of the art deep learning methods, I attempted to automatically annotate flooded, washed out, or otherwise severely damaged roads.  My goal is create a tool that can help detect and visualize anomalous roads in a simple user interface.

# Getting Started
## Prerequisites

Code written for Python 2.7.  Minor modification may be necessary for use with Python 3.

To install required Python libs, in your shell, run:
> pip install -r requirements.txt

## Training Data

My model is trained purely with retina (512x512) satellite imagery and corresponding street map images from [MapBox API](https://www.mapbox.com/api-documentation/) but this code will accept training images of other sizes or other sources.
Because I do not own the training images, no training data is copied here.  Consider signing up for a MapBox account to get an access token.  Please observe the licensing terms of any image you use for training.

I chose to train images from zoom level 16 (Open Street Map Slippy Map Format).  It is recommended to use training images from zoom 16 or above.

## Training the Road Segmentation Neural Network Model

Once you have training data, you are ready to train the model.

For an in-depth technical discussion of the model, please see blog post.

Command to train the model:

> python RoadSegmentor.py [config file]

You must create a config file to point to your datasets and specify neural network training hyper-parameters.  The config file contains paths to the training images and model training parameters.  See cfg/default.cfg for an example.

In your configuration file, specify the relative or absolute path to your training data.  You must also supply a plaintext file listing satellite image relative filepath under column label 'img'.  See data/tile_log.csv for an example.

With a Nvidia Quadro P5000 GPU (16MB GPU RAM), training with a batch size of 16 took about 5 hours on a dataset with about 9000 images.

## Generating Road Segmentation Tiles

For road change detection, you will need to generate 2 sets of road segmentations: pre-hurricane and post-hurricane.

Post Hurricane satellite imagery (in GeoTiff format) can be downloaded via the [DigitalGlobe Open Data Program](https://www.digitalglobe.com/opendata/).  For convenience, you can use data/tiffDownloader.py to download a set of geotiff files.  The script takes in a text file containing a list of http urls on separate lines.

In a data pre-processing step, [GDAL2Tiles](https://github.com/OSGeo/gdal) can be used to cut the GeoTiff files into standard OSM square tiles (use matching zoom level as the training set).

Command line tool:

> python roadSegmentationMaskGen.py [satellite_images_dir] [output_dir] [keras_model_filepath]

Note: If you use my pre-trained Keras/Tensorflow model, input images must be 512x512.

## Generating Annotated Map Tiles

After both the pre-hurricane segmentation tiles and post-hurricane segemntation tiles have been generated,  the next step is generating the difference.

This project comes with a command line tool that generates a pyramid hierarchy of annotated map tiles following the same hierarchy structure of the input directories.

> python postprocessing/anomalyMapGen.py [pre_event_segment_dir] [post_event_segment_dir] [street_map_dir] [output_dir]

### Demo

Embed here if possible!

### License
Code: MIT License

Road Segmentation Weight File: CC-BY-SA

Map Artifacts: CC-BY-SA

### Disclaimer
This project is for research purposes only.  Further testing and improvement is needed for real world use.
