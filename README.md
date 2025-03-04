# face-and-body-detection


    # Detect Face's
    '''The program uses Haar cascades (haarcascade_frontalface_default.xml) to detect faces in each frame.
        If a face is detected:
        It draws a rectangle around the face in the frame.
        It saves an image of the detected face region, timestamped with the current date and time.
        It starts recording a video of the feed if it is not already recording.'''
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 1, 1), 2)
        image_name = f'detected_face_{current_time}.png'
        # Include current time in image file name
        cv2.imwrite(image_name, frame[y:y+h, x:x+w])
        # Assuming only face is captured
        print(f"Face detected. Image saved as {image_name}")

    # Detect body
    ''' The program uses Haar cascades (haarcascade_fullbody.xml) to detect bodies in each frame.
        If a body is detected:
        It draws a rectangle around the body in the frame.
        It saves an image of the detected body region, timestamped with the current date and time.'''
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        body_name = f'detected_body_{current_time}.png'
        # Include current time in image file name
        cv2.imwrite(body_name, frame[y:y + h, x:x + w])
        print(f'Body detected. Image saved as {body_name}')
        # Assuming only body is captured

Additional Enhancements
# Use pre-trained DNN models for better accuracy.
# Implement a logging system.
# Add a GUI for better user experience. 
