import cv2
import mediapipe as mp
import 


class holistic:

# ========================================== init =================================================
  def __init__(self):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    pTime = 0

    self.mp_drawing = mp_drawing
    self.mp_holistic = mp_holistic
    self.mp_drawing_styles = mp_drawing_styles
    self.pTime = pTime


# ======================================= simple holistic =============================================
  def simple_holistic(self, show_fps=True):
    cap = cv2.VideoCapture(0)

    with self.mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as holistic:

      while cap.isOpened():

        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          continue

        # To improve performance, optionally mark the image as not writeable to
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # face landmarks
        self.mp_drawing.draw_landmarks( image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style() )

        # pose landmarks
        self.mp_drawing.draw_landmarks( image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        flip_image = cv2.flip(image, 1)

        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
          print("killed")
          break
        
    cap.release()
    cv2.destroyAllWindows()


# ====================================== complex holistic ============================================
  def complex_holistic(self, show_fps=True):
    # capturing webcam 0
    cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with self.mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as holistic:
    
      while cap.isOpened():
      
        success, image = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # face landmarks
        self.mp_drawing.draw_landmarks( image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) )

        # Right hand
        self.mp_drawing.draw_landmarks( image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2) )

        # Left Hand
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2) )

        # Pose Detections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
        
        flip_image = cv2.flip(image, 1)

        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
          print("killed")
          break
        
    cap.release()
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
  h = holistic()
  h.complex_holistic()