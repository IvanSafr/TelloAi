#######################################################
# Импорт
#######################################################
from djitellopy import Tello
import cv2
import time
import numpy as np
#######################################################
from funck import display
#######################################################



#######################################################
# Инициализация связи с tello
#######################################################
me = Tello()

me.connect()
me.forward_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0
me.yaw_velocity = 0
me.speed = 0

# # Настройки видео
# me.set_video_bitrate(Tello.BITRATE_1MBPS)
# me.set_video_fps(Tello.FPS_5)

me.streamon()
print("await")

time.sleep(15)
print("await done")

print(f"{me.get_battery()}% power")


#######################################################
# Настройки
#######################################################
width = 1280
height = 720
deadZone = 120
########################################################
dedZoneS = deadZone**2
########################################################
for_back_speed = 30
left_right_speed = 20
up_down_speed = 30
yaw_speed = 50
########################################################




#######################################################
# Окна для настройки цветов
#######################################################
def nothing(x):
    pass
time.sleep(5)
cv2.namedWindow("settings")  # создаем окно настроек
cv2.namedWindow("res")  # создаем окно настроек

cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)


frames = me.get_frame_read()

time.sleep(3)
print("Начало основного цикла")

me.takeoff() # Взлет

time.sleep(3)

#######################################################
# Основной цикл полета
#######################################################
while True:
    #######################################################
    # Получение фрейма
    #######################################################
    frame = frames.frame
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Нужный цветовой формат
    img = cv2.resize(img, (width, height))

    #######################################################
    # Собираенм данные с ползунков
    #######################################################
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')

    # формируем начальный и конечный цвет фильтра
    hsv_min = np.array((h1, s1, v1), np.uint8)
    hsv_max = np.array((h2, s2, v2), np.uint8)



    #######################################################
    # Работа с видеопотоком
    #######################################################
    img = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    kernel = np.ones((3, 3), np.uint8)

    cl_s = cv2.dilate(thresh, kernel, iterations=15)
    edge = cv2.Canny(cl_s, 120, 180)

    contours, vtf = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    ##################################################################
    # Обработка и отрисовка
    ##################################################################
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    display(img, width, height, deadZone)

    ##################################################################################################
    # Индикация статуса заряда батареи
    ##################################################################################################
    charge = "charge: " + str(me.get_battery())

    cv2.putText(img, charge, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                2)


    ##################################################################################################

    if len(contours) == 0:
        continue

    contour = contours[0]

    #cv2.imshow("res", img)

    if cv2.contourArea(contour) < 10000:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    center = (int(x + w // 2), int(y + h // 2))
    img = cv2.circle(img, center, center[0] - x, (0, 255, 0), 5)
    tar_area = (center[0]-x)**2 * 3.14



    ##################################################################################################
    # Обработка измения положения
    ##################################################################################################
    # Движение влево-вправо
    if center[0] < width / 2 - deadZone:
        dirction = "left"

        me.forward_back_velocity = 0
        me.up_down_velocity = 0
        me.left_right_velocity = 0 #-left_right_speed
        me.yaw_velocity = -yaw_speed

    elif center[0] > width / 2 + deadZone:
        dirction = "right"

        me.forward_back_velocity = 0
        me.up_down_velocity = 0
        me.left_right_velocity = 0 #left_right_speedq
        me.yaw_velocity = yaw_speed

    # Движение вверх-вниз
    elif center[1] > height / 2 + deadZone:
        dirction = "down"

        me.forward_back_velocity = 0
        me.up_down_velocity = -up_down_speed
        me.left_right_velocity = 0
        me.yaw_velocity = 0

    elif center[1] < height / 2 - deadZone:
        dirction = "up"

        me.forward_back_velocity = 0
        me.up_down_velocity = up_down_speed
        me.left_right_velocity = 0
        me.yaw_velocity = 0

    # Движение вперед-назад
    elif tar_area*0.6 < dedZoneS:
        dirction = "forw"

        me.forward_back_velocity = for_back_speed
        me.up_down_velocity = 0
        me.left_right_velocity = 0
        me.yaw_velocity = 0

    elif tar_area*0.6 > dedZoneS:
        dirction = "back"

        me.forward_back_velocity = -for_back_speed
        me.up_down_velocity = 0
        me.left_right_velocity = 0
        me.yaw_velocity = 0

    else:
        dirction = ""

        me.up_down_velocity = 0
        me.left_right_velocity = 0
        me.yaw_velocity = 0


    # Отрисовка графической информации
    cv2.putText(img, dirction, (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, 2)
    cv2.imshow("res", img)

    # Отправка команды управления
    me.send_rc_control(me.left_right_velocity, me.forward_back_velocity, me.up_down_velocity, me.yaw_velocity)

    k = cv2.waitKey(1) & 0xff
    if k == ord("q"):
        cv2.destroyAllWindows()
        break

# Алгоритм посадки
me.land()
time.sleep(1)
me.streamoff()
