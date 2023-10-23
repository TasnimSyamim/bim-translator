import os
import numpy as np

# Extract frames from the video
import cv2

video_path = 'Data (Video)/apa khabar/person_1/apa khabar.mp4'
save_location = 'Data (Video)/apa khabar/person_1/'

video = cv2.VideoCapture(video_path)

frame_count = 0

while video.isOpened():
    # Read the current frame
    ret, frame = video.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Process or save the frame as desired
    frame_count += 1

    # Example: Display and save the frame
    # Specify the save path for each frame
    frame_save_path = f'{save_location}/frame_{frame_count}.jpg'

    cv2.imshow('Frame', frame)
    cv2.imwrite(frame_save_path, frame)

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object and close windows
video.release()
cv2.destroyAllWindows()


