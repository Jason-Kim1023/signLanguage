# SignLanguage

A real-time sign language recognition tool. (Tensorflow/Keras - Machine Learning, OpenCV - Data Collection, Numpy for Comprehensive Mathematical Functions - 2D Matricies)

Utilizes Tensorflow and Keras to create an Nueral Network that utilizes real-time image pixel data to predict the hand-sign of an individual.  OpenCV is utilized for data collection and for real-time predictions.  By flattening the 2D Matrix of the image data into a single numpy array, the 3 layered Nueral Network learns from the data and makes accurate predictions of what sign language sign the individual is making.  

### In attached Jupyter Notebook File, attempted to create a convolutional neural network, but it is a lot more demanding and CUDA Cores are too weak to run the program.

###NOTES###


-->  A CNN(convolutional neural network) would be more practical and would help this machine learning program run more smoothly. 

-->  A high number of inputs results in a OOM(out of memory) error, thus the image size had to be reduced to 64x64 pixels.  

-->  Numpy arrays are hard to append to for 2D matricies, so it is easiest to just create a normal 2D array and then type cast it to numpy.

-->  High numbers of inputs for pixel data results in a very demanding nueral network, so CUDA Cores must be utilized to run training models.  
    --> Traning defaults to utilizing the CPU, but you can utilize CUDA Cores from GPU to run your training models. Still demanding for high resolution and image sizes.
