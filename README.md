# Face Mask Detector
A face mask detector made in Pytorch (lightning?). Primarily made for The Sparks Foundation's Computer vision and IoT internship.
I could've copied the code from the TensorFlow tutorial, but I figured out it'd be a good chance to learn more about pytorch and challenge myself.

# Pitfalls
* **Face detection**: You can't really classify faces if you don't detect any. This is one failure point that's available to tinker with to hopefully achieve better results. 
Another thing is that the face detector model I'm using (res10_300x300_ssd_iter_140000) limits the maximum input image size, so that is another thing to consider.
* **Colored masks**: The model seems to do well enough on generic light blue masks, but fails to work properly on my black mask. This is probably an issue with the given dataset having less black masks. One solution I thought of was to convert images to grayscale but that seemed to make the model overall worse.
* **Other models?**: I did try with MobileNetV3 large/small with the dataset, but they seem to not train well. The model given in the pytorch lightning model seems to work much better.
* **Noise**: The model doesn't really deal well with noise. One solution I have in mind is to add noise while training,  my current experiments with this solution don't show a lot of success unfortunately.

#   Currently implemented:
*   Webcam classification
*   Image classification
*   Training

#   TODO
* Video classification
* Projectt cleanup
* Better checkpoints?

# Sources and References
*   [COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning - PyImageSearch](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

*   [How I built a Face Mask Detector for COVID-19 using PyTorch Lightning - Jad Haddad on Medium](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61)

