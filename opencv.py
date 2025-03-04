import cv2
from datetime import datetime  # Import the datetime module

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize body detector (full body)
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False

while True:
    # Read a frame
    ret, frame = cap.read()

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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

    else:
        if recording:
            # Stop recording
            out.release()
            recording = False

    # Write frame to video if recording
    if recording:
        out.write(frame)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if recording:
            out.release()
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
