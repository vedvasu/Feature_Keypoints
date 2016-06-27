# Feature_Keypoints
Key points descriptors can be used to determine the key points in an image. Useful information can be extracted from these Keypoints. 

Features Used : 

1) SIFT (Scale Invariant feature Transform):
    
    - Many features works on the principle of finding the corners in a image but if the image is rotated or the image is scaled the corners of a image maynot remain a cormer.
    - Mainly for variation in scale the problem arises. As a result SIFT was designed for scale invariant feature analysis.
    - It finds the Laplacian of Gaussian of the image with various scale variant sigma values.

* sift.detect() function finds the keypoint in the images. The results can be filtered.
*Each keypoint is a special structure which has many attributes like:
	- its (x,y) coordinates, 
	- size of the meaningful neighbourhood, 
	- angle which specifies its orientation, 
	- response that specifies strength of keypoints etc.

* Overview of Operations:
	- Step 1: Find space scale extreme (LoG of image to find Keypoints)
	- Step 2: Improve Keypoints and filter bad keypoints
	- Step 3: Orientation Assignment
	- Step 4: Create self descriptor