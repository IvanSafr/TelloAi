import cv2



def display(img, frameWidth, frameHeight, deadZone):
    cv2.line(img,(int(frameWidth/2) - deadZone, 0), (int(frameWidth/2) - deadZone, frameHeight),(255,255,0),3)
    cv2. line(img, (int(frameWidth/2) + deadZone, 0), (int(frameWidth/2) + deadZone, frameHeight),(255,255,0),3)
    cv2. line (img, (0,int(frameHeight / 2) - deadZone), (frameWidth, int (frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2. line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int (frameHeight / 2) + deadZone), (255, 255, 0), 3)
