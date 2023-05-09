import cv2
import mediapipe as mp
import time



class handDetect():
    def __init__(self, mode= False, maxHands = 2
     , detectionCon = 0.5, trackCon = 0.5 , drawLine = False, drawCircle = False, drawCircleSelected = False):
            self.DrawLine = drawLine
            self.DrawCircle = drawCircle
            self.DrawCircleSelected = drawCircleSelected
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands( static_image_mode=mode,    max_num_hands= maxHands,  min_detection_confidence= detectionCon ,  min_tracking_confidence= trackCon)
            self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img ):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =  self.hands.process(imgRGB) 

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:
    
                if( self.DrawLine):                
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                  
        return img



    def findPosition(self, img, selectedID = 0):

        lmList = []
    
        if self.results.multi_hand_landmarks :
            hand = self.results.multi_hand_landmarks[0]

            for id , lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int( lm.x * w), int(lm.y *h)
                lmList.append([id , cx, cy])

                if self.DrawCircleSelected and selectedID == id :
                    cv2.circle(img, (cx, cy) , 10 , (255, 0, 255) , cv2.FILLED)
              
        return lmList


    
    def print_all_cirles(self, img  ):
    
        if self.results.multi_hand_landmarks :
           for hand in self.results.multi_hand_landmarks:

            for id , lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int( lm.x * w), int(lm.y *h)

                if self.DrawCircle:
                    cv2.circle(img, (cx, cy) , 10 , (255, 0, 255) , cv2.FILLED)
    


def main():



    cam = cv2.VideoCapture(0)
        
    pTime = 0 
    cTime = 0

    detector = handDetect(drawLine= True, drawCircle= False , drawCircleSelected= True)

    showFPS = False

    selectedID = 1
    
    while True:

        succes, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img , selectedID= selectedID)

        detector.print_all_cirles(img)

        if ( len(lmList) != 0):
            print( lmList[8])

        if(showFPS):
            cTime = time.time()
            fps =  1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)) , (10, 50), cv2.FONT_HERSHEY_PLAIN , 3, (255, 0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ ==  "__main__":
    main()
    