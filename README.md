# face-and-body-detection
Face and Body Detection with movement body language and save the photos and videos for specific movement

if len(faces) > 0:
    ''' When a face is detected, the program initiates video recording.
        The video recording continues as long as faces are detected.
        If no faces are detected, the program stops recording. '''
    if not recording:
        # Start recording
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_time2 = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Get current time and date
        out = cv2.VideoWriter(f'detected_face_video_{current_time2}.avi', fourcc, 20.0, (640, 480))
        recording = True

    
