import cv2
import mediapipe as mp
import time


class mediapipe:

# ============================================ init =================================================
  def __init__(self):
    
    # mediapipe solutions variable
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    pTime = 0

    self.mp_drawing = mp_drawing
    self.mp_holistic = mp_holistic
    self.mp_drawing_styles = mp_drawing_styles
    self.mp_face_detection = mp_face_detection
    self.mp_hands = mp_hands
    self.mp_pose = mp_pose
    self.mp_face_mesh = mp_face_mesh
    self.pTime = pTime



# ===================================================================================================
  def simple_holistic(self, show_fps=False):

    # capturing webcam 0 
    cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with self.mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as holistic:
      while cap.isOpened():
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          continue

        # Recolor Feed
        image.flags.writeable = False        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # face landmarks
        self.mp_drawing.draw_landmarks( 
          image, 
          results.face_landmarks, 
          self.mp_holistic.FACEMESH_CONTOURS, 
          landmark_drawing_spec=None, 
          connection_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(0, 167, 196), thickness=2, circle_radius=1) 
        )

        # pose landmarks
        self.mp_drawing.draw_landmarks( 
          image, 
          results.pose_landmarks, 
          self.mp_holistic.POSE_CONNECTIONS, 
          landmark_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
          connection_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2) 
        )

        # fliping image 
        flip_image = cv2.flip(image, 1)

        # fps
        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Showimg image
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        # quiting 
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break
        
    cap.release()
    cv2.destroyAllWindows()



# ===================================================================================================
  def complex_holistic(self, show_fps=True):
    
    # capturing webcam 0
    cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with self.mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as holistic:
      while cap.isOpened():
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          continue

        # Recolor Feed
        image.flags.writeable = False        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make Detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # face landmarks
        self.mp_drawing.draw_landmarks( 
          image, 
          results.face_landmarks, 
          self.mp_holistic.FACEMESH_TESSELATION, 
          landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), 
          connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1) 
        )
        
        # Right hand
        self.mp_drawing.draw_landmarks( 
          image, 
          results.right_hand_landmarks, 
          self.mp_holistic.HAND_CONNECTIONS, 
          landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3), 
          connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2) 
        )

        # Left Hand
        self.mp_drawing.draw_landmarks(
          image, 
          results.left_hand_landmarks, 
          self.mp_holistic.HAND_CONNECTIONS, 
          landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3), 
          connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2) 
        )

        # Pose landmarks
        self.mp_drawing.draw_landmarks(
          image, 
          results.pose_landmarks, 
          self.mp_holistic.POSE_CONNECTIONS, 
          landmark_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
          connection_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2) 
        )
        
        # fliping image
        flip_image = cv2.flip(image, 1)

        # fps
        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # showing fliped image
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        # quiting
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break
        
    cap.release()
    cv2.destroyAllWindows()



# =========================================================================================================================
  def face_mesh(self, show_fps=True, contours=True ):

    # capturing webcam 0
    cap = cv2.VideoCapture(0)

    # Initiate face mesh
    with self.mp_face_mesh.FaceMesh( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as face_mesh:
      while cap.isOpened():
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        # Recolor Feed
        image.flags.writeable = False        

        # Make Detections
        results = face_mesh.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   

        # face mesh results
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:

            self.mp_drawing.draw_landmarks( 
              image=image, 
              landmark_list=face_landmarks, 
              connections=self.mp_face_mesh.FACEMESH_TESSELATION, 
              landmark_drawing_spec=None,
              connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1 ) 
            )
            if contours==True:
              self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
              )

            
        # fliping image
        flip_image = cv2.flip(image, 1)

        # fps
        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # showing image
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        # quiting
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break
        
    cap.release()
    cv2.destroyAllWindows()



# ==================================================================================================
  def hand_detector(self, show_fps=True):
  
    # Capuring webcam 0
    cap = cv2.VideoCapture(0)
  
    with self.mp_hands.Hands( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as hands:
      while cap.isOpened():
        success, image = cap.read()
        
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        # Make Detections
        results = hands.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # hand detector results
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3), 
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2) 
              )
        
        # flipping image
        flip_image = cv2.flip(image, 1)

        # fps
        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # showing flipped image
        cv2.imshow('MediaPipe Holistic', flip_image)
        
        # quitting
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()



# ==================================================================================================
  def pose(self, show_fps=True):

    # capuring webcam 0
    cap = cv2.VideoCapture(0)

    with self.mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5 ) as pose:
      while cap.isOpened():
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        
        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        # Make Detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.mp_drawing.draw_landmarks( 
          image, 
          results.pose_landmarks, 
          self.mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
          connection_drawing_spec=self.mp_drawing_styles.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2) 
        )
        
        # flipping image
        flip_image = cv2.flip(image, 1)

        # fps
        if show_fps==True:
          cTime = time.time()
          fps = 1 / (cTime - self.pTime)
          self.pTime = cTime
          cv2.putText(flip_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # showing flipped image
        cv2.imshow('MediaPipe Holistic', flip_image)

        # quitting
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()

# ================================================ class end ===============================================================

if __name__ == '__main__':
  pipe = mediapipe()
  pipe.simple_holistic()