import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image from your computer
img = cv2.imread("C:/Users/lathi/OneDrive/Documents/360_F_828493018_Ntaia2HBMK7UHVFyP8jv0UrTcD7Fk7pw.jpg")   # change path if needed

if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image pixels
    pixels = np.float32(rgb_img.reshape((-1, 3)))

    # K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Number of clusters
    K = 3

    # Apply K-means
    ret, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(rgb_img.shape)

    # Display images
    plt.subplot(1,2,1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(segmented_img)
    plt.title("Segmented Image")
    plt.axis("off")

    plt.show()
