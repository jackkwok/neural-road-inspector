import cv2
import numpy as np

def ecc_align(im1, im2, number_of_iterations=2000):
	"""
		Find the translation motion matrix that align im2 to im1.

		The Enhanced Correlation Coefficient image alignment algorithm is based on a 2008 paper titled 
		Parametric Image Alignment using Enhanced Correlation Coefficient Maximization 
		by Georgios D. Evangelidis and Emmanouil Z. Psarakis. 
		They propose using a new similarity measure called Enhanced Correlation Coefficient (ECC) 
		for estimating the parameters of the motion model. 

		Parameters:
			im1: image 1
			im2: image 2

		Returns:
			the cv2.MOTION_TRANSLATION motion model 2x3 matrix that maps im2 to im1.
	"""

	# Convert images to grayscale
	im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	 
	# Find size of image1
	sz = im1.shape
	 
	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION
	 
	# Define 2x3 matrice and initialize the matrix to identity
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	 
	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;
	
	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	 
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
	 
	# Use warpAffine for Translation, Euclidean and Affine
	im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
	
	# Show final results
	cv2.imshow("Image 1", im1)
	cv2.imshow("Image 2", im2)
	cv2.imshow("Aligned Image 2", im2_aligned)
	cv2.waitKey(0)

	return cc, warp_matrix
