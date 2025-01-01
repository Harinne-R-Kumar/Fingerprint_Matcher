import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# Create directories if they do not exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to process the uploaded fingerprint image
def process_image(file_path):
    # Load the sample image
    sample = cv2.imread(file_path)
    sample = cv2.resize(sample, (500, 500))  # Resize for consistent processing

    # SIFT matching
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

    best_score = 0
    filename = None
    image = None
    kp1, kp2, mp = None, None, None

    # Directory containing real fingerprint images
    real_fingerprints_dir = 'SOCOFing/Real'
    counter=0

    for file in os.listdir(real_fingerprints_dir)[:1000]:
        if counter%10 ==0:
            print(counter)
        counter+=1
        fingerprint_image = cv2.imread(os.path.join(real_fingerprints_dir, file))
        fingerprint_image = cv2.resize(fingerprint_image, (500, 500))  # Resize for consistent processing
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        # Use FLANN matcher
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = [p for p, q in matches if p.distance < 0.7 * q.distance]  # Adjusted ratio test threshold

        keypoints = min(len(keypoints_1), len(keypoints_2))
        if keypoints > 0 and (len(match_points) / keypoints * 100) > best_score:
            best_score = len(match_points) / keypoints * 100
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    if filename:
        # Draw matches
        result_image = cv2.drawMatches(sample, kp1, image, kp2, mp[:50], None,  # Limit matches for clarity
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result_image_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
        cv2.imwrite(result_image_path, result_image)
        return result_image_path, filename, round(best_score, 2)
    return None, None, None


# Streamlit App
st.title("Fingerprint Matching")
st.write("Upload a fingerprint image to match it with the database.")

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png","bmp"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the image
    st.write("Processing the uploaded image...")
    result_image_path, matched_filename, score = process_image(file_path)

    if result_image_path:
        st.success("Match Found!")
        st.write(f"**Matched File:** {matched_filename}")
        st.write(f"**Match Score:** {score}%")
        
        # Display the result image
        result_image = Image.open(result_image_path)
        st.image(result_image, caption="Keypoint Matches Visualization", use_column_width=True)
    else:
        st.error("No match found!")
