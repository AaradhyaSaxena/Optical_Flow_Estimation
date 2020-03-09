# Depth Estimation
.

We present an approach to estimate the pixel-wise depth of the objects using monocular images. 
We first estimate the point to point correspondance between different frames using Optical Flow.
Estimating dense optical flow is one of the longstanding problems in computer vision, with a variety of applications.
Most of the CNN based approaches rely on the availability of a large amount of ground truth for supervised learning.
Due to the difficulty of obtaining ground truth in real scenes, such networks are trained on synthetically generated images, for which dense ground truth is easy to obtain in large amounts. 
However, because of the intrinsic difference between synthetic and real imagery and the limited variability of synthetic datasets, the generalization to real scene remains challenging. 
In order to cope with the lack of labeled real-world training data, we propose a network based on unsupervised learning.
We introduce an end-to-end unsupervised approach that demonstrates the effectiveness of unsupervised learning for optical flow.
We propose a new unsupervised loss function. 
We compute bidirectional optical flow both in the forward and backward direction, performing a second pass with the two input images exchanged.
We train our network on MPI Sintel dataset. 
Thus, we move forward in the direction of eliminating the need for ground truth for optical flow estimation.
