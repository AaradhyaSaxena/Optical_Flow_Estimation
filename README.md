# Unsupervised Optical Flow Estimation
deepwiki: https://deepwiki.com/AaradhyaSaxena/Optical_Flow_Estimation

Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It is the 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second.
</br></br>
We introduced an end-to-end unsupervised approach for estimating Optical Flow. 
We proposed a new unsupervised loss function. 
In our approach, we compute the bidirectional optical flow both in the forward and backward direction, performing a second pass with the two input images exchanged. 
The network architecture consists of a contracting path, followed by an expansive path with skip connections.
</br>
For each image pair I1 and I2, we get a pair of optical flows, from I1 to I2 and from I2 to I1, by flipping the images in the input layer before concatenation. 
Our unsupervised flow is based on the observation that a pixel in the first frame should be similar to the pixel in the second frame to which the flow maps it. 
We compare the I1* (reconstruction of I1 with the obtained flow) with I1, and similarly I2 with I2*. 
We take the Mean squared error (MSE), and MSE of the gradients of the image and corresponding reconstructed the image as the loss function. 
For the smoothness of the obtained flow, we employ an edge-aware smoothness regularization term to encourage local smoothness while ensuring sharpness at the edge. 
Further, for finding a better correspondence between both the output optical flows, we reconstruct I2* from I1 and then reconstruct I1** from I2* and minimize the misfit.
</br></br>
Estimating dense optical flow is one of the longstanding problems in computer vision, with a variety of applications.
Most of the CNN based approaches rely on the availability of a large amount of ground truth for supervised learning.
Due to the difficulty of obtaining ground truth in real scenes, such networks are trained on synthetically generated images, for which dense ground truth is easy to obtain in large amounts. 
However, because of the intrinsic difference between synthetic and real imagery and the limited variability of synthetic datasets, the generalization to real scene remains challenging.
The motivation for this project was to move forward in the direction of eliminating the need for ground truth for optical flow estimation.
</br>

### Result
The two image frames, output of the network and the ground truth are displayed below:
</br>

frame_1
<p align="center">
  <img src="images/testX1.png" width=500>
</p>
</br>

frame_2
<p align="center">
  <img src="images/testX2.png" width=500>
</p>
</br>

Output
<p align="center">
  <img src="images/testy.png" width=500>
</p>
</br>

ground truth
<p align="center">
  <img src="images/ground_truth.png" width=500>
</p>
</br>


