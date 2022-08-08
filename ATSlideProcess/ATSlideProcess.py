#!/usr/bin/python
import cv2
import numpy as np
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def image_align(im1, im2):
    # Convert images to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect features and compute descriptors.
    feature = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = feature.detectAndCompute(im1, None)
    keypoints2, descriptors2 = feature.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def print_slide_props(slide: open_slide):
    print(slide.properties)


lm1 = open_slide("../Assets/LM1.ndpi")
lm2 = open_slide("../Assets/LM2.ndpi")

print_slide_props(lm1)

tiles = DeepZoomGenerator(lm1, tile_size=256, overlap=0, limit_bounds=False)
