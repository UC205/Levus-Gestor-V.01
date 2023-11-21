import time
import cvzone as cvz
import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pag
from screeninfo import get_monitors
from pynput.mouse import Button, Controller
from fingers import THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP

monitors = get_monitors()

if monitors:
    WIDTH = monitors[0].width
    HEIGHT = monitors[0].height
else:
    WIDTH = 1920
    HEIGHT = 1080

cap = cv.VideoCapture(0)
mouse = Controller()
state = ''

hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)


def get_distance(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) 
    return distance

class FPS:
    def __init__(self, avgCount=30):
        self.pTime = time.time()
        self.frameTimes = []
        self.avgCount = avgCount

    def update(self, frame=None, pos=(10, 30), bgColor=(255, 0, 255),
               textColor=(255, 255, 255), scale=3, thickness=3):
        cTime = time.time()
        frameTime = cTime - self.pTime
        self.frameTimes.append(frameTime)
        self.pTime = cTime

        if len(self.frameTimes) > self.avgCount:
            self.frameTimes.pop(0)

        avgFrameTime = sum(self.frameTimes) / len(self.frameTimes)
        fps = 1 / avgFrameTime

        if frame is not None:
            cvz.putTextRect(frame, f'FPS: {int(fps)}', pos,
                            scale=scale, thickness=thickness, colorT=textColor,
                            colorR=bgColor, offset=10)
        return fps, frame

run = True
fpsReader = FPS(avgCount=30)

while True:
    success, frame = cap.read()

    if success is False:
        break
    frame = cv.flip(frame, 4)
    result = hands.process(frame)

    # Rectangulo para centrar la mano
    height, width, _ = frame.shape
    rect_width = width // 1
    rect_height = height // 1
    start_point = (width // 2 - rect_width // 2, height // 2 - rect_height // 3)
    end_point = (width // 2 + rect_width // 2, height // 2 + rect_height // 3)
    color = (0, 255, 255)
    thickness = 2
    frame = cv.rectangle(frame, start_point, end_point, color, thickness)

    fps, frame = fpsReader.update(frame, pos=(10,20),
                                   bgColor=(100, 0, 100), textColor=(255, 255, 255),
                                   scale=1, thickness=1)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            # Pulgar
            xThumb, yThumb = handLms.landmark[THUMB_TIP].x * WIDTH, handLms.landmark[THUMB_TIP].y * HEIGHT
            # Indice
            xIndex, yIndex = handLms.landmark[INDEX_TIP].x * WIDTH, handLms.landmark[INDEX_TIP].y * HEIGHT
            # Medio
            xMiddle, yMiddle = handLms.landmark[MIDDLE_TIP].x * WIDTH, handLms.landmark[MIDDLE_TIP].y * HEIGHT
            # Anular
            xRing, yRing = handLms.landmark[RING_TIP].x * WIDTH, handLms.landmark[RING_TIP].y * HEIGHT
            # Menique
            xPinky, yPinky = handLms.landmark[PINKY_TIP].x * WIDTH, handLms.landmark[PINKY_TIP].y * HEIGHT

            # Click izquiero
            if state == 'LMB':
                # Distancia de pulgar y el meñique
                if get_distance((xThumb, yThumb), (xPinky, yPinky)) > 70:
                    state = ''
                    mouse.release(Button.left)
            elif state != 'LMB':
                # Comprueba si el pulgar y el meñique estan cerca
                if get_distance((xThumb, yThumb), (xPinky, yPinky)) < 70:
                    state = 'LMB'
                    mouse.press(Button.left)

            # Click derecho
            if state == 'RMB':
                # Distancia de pulgar y el anular
                if get_distance((xThumb, yThumb), (xRing, yRing)) > 70:
                    state = ''
                    mouse.release(Button.right)
            elif state != 'RMB':
                # Comprueba si el pulgar y el anular estan cerca
                if get_distance((xThumb, yThumb), (xRing, yRing)) < 70:
                    state = 'RMB'
                    mouse.press(Button.right)

            # Con el pulgar y el índice minizar las ventanas abiertas.
            if get_distance((xThumb, yThumb), (xIndex, yIndex)) < 70:
                pag.hotkey('win', 'm')
            # Abrir Windows con el dedo pulgar y el dedo del medio.
            if get_distance((xThumb, yThumb), (xMiddle, yMiddle)) < 70:
                pag.hotkey('win')
            # Con el índice y el medio aparece el teclado digital.
            if get_distance((xIndex, yIndex), (xMiddle, yMiddle)) < 70:
                pag.hotkey('win', 'ctrl', 'o')

            # Mover el cursor
            alpha = 0.5
            current_x, current_y = mouse.position
            x = (1 - alpha) * current_x + alpha * xIndex
            y = (1 - alpha) * current_y + alpha * yIndex
            mouse.position = (x, y)

cap.release()
cv.destroyAllWindows()
