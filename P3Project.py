import numpy as np
import cv2
from random import randint
import time

#hello from anna

# running live feed
cap = cv2.VideoCapture(1)
ret, frame = cap.read()

cv2.imwrite('canny.PNG', frame)
frame = cv2.imread('canny.PNG')
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

# testing on image
#frame = cv2.imread('imagetest92.png')

# frame = cv2.imread('realboardflip.png')

#cv2.imwrite('canny.PNG', frame)
#frame = cv2.imread('canny.PNG')

redWons = cv2.imread('redWins.png')
blueWons = cv2.imread('blueWins.png')

redWon = cv2.resize(redWons, (400, 400))
blueWon = cv2.resize(blueWons, (400, 400))

h = frame.shape[0]
frameh = h
w = frame.shape[1]
framew = w

settime = 0.0

horizontal = []
vertical = []

redTotalScoreArray = []
blueTotalScoreArray = []

redCurrentScore = 0
blueCurrentScore = 0

redTotalScore = 0
blueTotalScore = 0

round = 1


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


class Puck:
    def __init__(self, x, y, w, h, valid, color, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.valid = valid
        self.color = color  # 0 is red, 1 is blue
        self.score = score

    def draw(self, frameVideo):
        if self.color is 0:
            text = "RED"
            col = (0, 0, 255)
        else:
            text = "BLUE"
            col = (255, 0, 0)

        cv2.rectangle(frameVideo, (x, y), (x + w, y + h), col, 1)
        cv2.putText(frameVideo, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col)

    def validate(self):

        if self.y + self.h > getValueY(horizontaly, 0, self.x) \
                and self.x + self.h > getValueX(verticalxleft, self.y) \
                and self.x < getValueX(verticalxright, self.y):
            self.valid = True
        else:
            self.valid = False

    def setScore(self, score):
        self.score = score



def getValueY(horizontaly, lineindex, linepos):
    val = horizontaly[lineindex][linepos]

    return val


def getValueX(verticalx, linepos):
    val = verticalx[linepos]

    return val


def getFunction(x1, y1, x2, y2):
    if y2 - y1 == 0:
        a = 1 / (x2 - x1)
    elif x2 - x1 == 0:
        a = (y2 - y1) / 1
    else:
        a = (y2 - y1) / (x2 - x1)

    b = a * x1 - y1
    b = b * -1
    # b = -x1*a+y1

    return a, b


def getScore(puck, array):

    ytop = getValueY(horizontaly, 0, puck.x)
    y3 = getValueY(horizontaly, 1, puck.x)
    y2 = getValueY(horizontaly, 2, puck.x)
    y1 = getValueY(horizontaly, 3, puck.x)

    xleft = getValueX(verticalxleft, puck.y)
    xright = getValueX(verticalxright, puck.y)

    ypos = puck.y + puck.h
    xpos = puck.x

    # adjust hanger threshold to make sure only hanging pucks scores 4
    # consider adding hangerthreshold in again # ypos > ytop > puck.y + hangerthreshold
    if ypos > ytop > puck.y and xleft < xpos + puck.w and xpos < xright:
        array.append(4)
        point = 4
    elif ytop <= ypos < y3 and xleft < xpos + puck.w and xpos < xright:
        array.append(3)
        point = 3
    elif y3 <= ypos < y2 and xleft < xpos + puck.w and xpos < xright:
        array.append(2)
        point = 2
    elif y2 <= ypos < y1 and xleft < xpos + puck.w and xpos < xright:
        array.append(1)
        point = 1
    else:
        point = 0

    return point


edges = cv2.Canny(frame, 75, 200)

starter = randint(0, 1)

if starter is 0:
    startString = ("RED STARTS")
    startColor = (0, 0, 255)
else:
    startString = ("BLUE STARTS")
    startColor = (255, 0, 0)

# HoughLinesP
"""dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) of the detected lines
rho : The resolution of the parameter r in pixels. We use 1 pixel.
theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
threshold: The minimum number of intersections to “detect” a line
minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
maxLineGap: The maximum gap between two points to be considered in the same line."""

lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 50, maxLineGap=1000, minLineLength=100)
# lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 50, maxLineGap=1000, minLineLength=300)
# why does scoring system only work with a theta accumulator of 360 and not 180??

anglevertical = 90
anglehorizontal = 0
anglethreshold = 10
thresholdverticalline = 50

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angleInDegrees = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

        # vertical = green
        if x1 < w / 5 * 2 or x1 > w / 5 * 3:
            if anglevertical - anglethreshold < angleInDegrees <= anglevertical:
                cv2.line(frame, (x2, y2), (x1, y1), (0, 255, 0), 1)
                vertical.append(Line(x2, y2, x1, y1))

            elif -anglevertical <= angleInDegrees < -anglevertical + anglethreshold:
                vertical.append(Line(x1, y1, x2, y2))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # horizontal = red
        if anglehorizontal - anglethreshold < angleInDegrees < anglehorizontal + anglethreshold:
            if x1 < w / 4 or x1 > w / 4 * 3:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                horizontal.append(Line(x1, y1, x2, y2))

############################ NEW LINE STUFF #####################
leftline = 0
size = 0
oldsize = 0
for i in range(0, len(vertical)):
    if vertical[i].x1 < w / 2:
        size = vertical[i].x1
        if size > oldsize:
            oldsize = size
            leftline = i

rightline = 0
scoringlines = []
oldsize = 2000  # just a random large number to make sure the first pos is smaller
for i in range(0, len(vertical)):
    if vertical[i].x1 > w / 2:
        if vertical[i].x1 < oldsize:
            oldsize = vertical[i].x1
            rightline = i

threshold = 100
oldsize = 0
ytop = 0
y3 = 0
y2 = 0
y1 = 0

for i in range(0, len(horizontal)):
    if horizontal[i].y1 < h / 2 - threshold:
        if horizontal[i].y1 > oldsize:
            oldsize = horizontal[i].y1
            ytop = i

oldsize = 2000
for i in range(0, len(horizontal)):
    if horizontal[i].y1 > horizontal[ytop].y1:
        if horizontal[i].y1 < oldsize:
            oldsize = horizontal[i].y1
            y3 = i

oldsize = 2000
for i in range(0, len(horizontal)):
    if horizontal[i].y1 > horizontal[y3].y1 + threshold:
        if horizontal[i].y1 < oldsize:
            oldsize = horizontal[i].y1
            y2 = i

oldsize = 2000
for i in range(0, len(horizontal)):
    if horizontal[i].y1 > horizontal[y2].y1 + threshold:
        if horizontal[i].y1 < oldsize:
            oldsize = horizontal[i].y1
            y1 = i

# here we now have the 4 horizontal lines - our scoring lines. These are appended to scoringlines array
cv2.line(frame, (horizontal[ytop].x1, horizontal[ytop].y1), (horizontal[ytop].x2, horizontal[ytop].y2), (255, 0, 0), 2)
cv2.putText(frame, "ytop", (horizontal[ytop].x1 + 20, horizontal[ytop].y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2)
scoringlines.append(horizontal[ytop])

cv2.line(frame, (horizontal[y3].x1, horizontal[y3].y1), (horizontal[y3].x2, horizontal[y3].y2), (255, 0, 0), 2)
cv2.putText(frame, "y3", (horizontal[y3].x1 + 20, horizontal[y3].y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
            2)
scoringlines.append(horizontal[y3])

cv2.line(frame, (horizontal[y2].x1, horizontal[y2].y1), (horizontal[y2].x2, horizontal[y2].y2), (255, 0, 0), 2)
cv2.putText(frame, "y2", (horizontal[y2].x1 + 20, horizontal[y2].y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
            2)
scoringlines.append(horizontal[y2])

cv2.line(frame, (horizontal[y1].x1, horizontal[y1].y1), (horizontal[y1].x2, horizontal[y1].y2), (255, 0, 0), 2)
cv2.putText(frame, "y1", (horizontal[y1].x1 + 20, horizontal[y1].y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
            2)
scoringlines.append(horizontal[y1])

cv2.line(frame, (vertical[leftline].x1, vertical[leftline].y1), (vertical[leftline].x2, vertical[leftline].y2),
         (255, 0, 0), 2)
cv2.putText(frame, "xleft", (vertical[leftline].x2, vertical[leftline].y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
            2)
cv2.line(frame, (vertical[rightline].x1, vertical[rightline].y1), (vertical[rightline].x2, vertical[rightline].y2),
         (255, 0, 0), 2)
cv2.putText(frame, "xright", (vertical[rightline].x2, vertical[rightline].y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 0, 0), 2)

horizontaly = [[]]
verticalxleft = []
verticalxright = []

# this gives us the 2 vertical lines as vertical[xleft] and vertical[xright]
# and we get the 4 horizontal lines as scoringlines[0] through scoringlines[3]


j = -1

for i in range(0, len(scoringlines)):
    a, b = getFunction(scoringlines[i].x1, scoringlines[i].y1, scoringlines[i].x2, scoringlines[i].y2)
    horizontaly.append([])
    j += 1

    for x in range(0, w):
        y = a * x + b

        horizontaly[j].append(y)

# put the x values of the left and right vertical line, respectively, into an array
# so we can compare x values of pucks to them at any given y value
a, b = getFunction(vertical[leftline].x1, vertical[leftline].y1, vertical[leftline].x2, vertical[leftline].y2)
for y in range(0, h):
    x = (y - b) / a

    verticalxleft.append(x)

a, b = getFunction(vertical[rightline].x1, vertical[rightline].y1, vertical[rightline].x2, vertical[rightline].y2)
for y in range(0, h):
    x = (y - b) / a

    verticalxright.append(x)

# declare variables that should not be reset (reset if declared within the while loop)

xleft = 0
xright = 0
ytop = 0

y1 = 0
y2 = 0
y3 = 0

xadjustment = 50
yadjustment = 50

# COLOR TRACKING THRESHOLDS
lower_red = np.array([0, 100, 150], np.uint8)
upper_red = np.array([20, 255, 255], np.uint8)

# red other part of the threshold, include?
# lower_red2 = np.array([160, 100, 150], np.uint8)
# upper_red2 = np.array([180, 255, 255], np.uint8)


# adjusted upper hue value from 130 to 150:
blue_lower = np.array([110, 50, 50])
blue_upper = np.array([150, 255, 255])

# dimension settings for size of rectangle
boxMin = 35
boxMax = 45

redWins = False
blueWins = False

while 1:

    # running live feed
    ret, frameVideo = cap.read()
    _, scorewindow = cap.read()
    frameVideo = cv2.rotate(frameVideo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    scorewindow = cv2.rotate(scorewindow, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ############## START COLOR TRACKING ###############

    # converting frame(img i.e BGR) to HSV (hue, saturation, value)
    hsv = cv2.cvtColor(frameVideo, cv2.COLOR_BGR2HSV)

    # finding the range of the three colors in the image/frame
    red = cv2.inRange(hsv, lower_red, upper_red)
    # red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Morphological transformation, dilation (remember from imageProcessing)
    kernel = np.ones((3, 3), "uint8")

    # red
    red = cv2.dilate(red, kernel)
    # res = cv2.bitwise_and(frameVideo, frameVideo, mask=red)

    # blue
    blue = cv2.dilate(blue, kernel)
    # res3 = cv2.bitwise_and(frameVideo, frameVideo, mask=blue)

    # tracking the red color - Understand this and the for-loop:
    contoursRed, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contoursRed2, hierarchy = cv2.findContours(red2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # tracking the blue color - Understand this and the for-loop:
    contoursBlue, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    redPucks = []
    bluePucks = []

    indexRed = 0
    indexBlue = 0

    # Show BLOB (red COLOR)
    for pic, contour in enumerate(contoursRed):
        x, y, w, h = cv2.boundingRect(contour)
        if w and h > boxMin and w and h < boxMax:
            redPucks.append(Puck(x, y, w, h, False, 0, 0))
            redPucks[indexRed].draw(frameVideo)
            indexRed += 1

    # Show BLOB (blue COLOR)
    for pic, contour in enumerate(contoursBlue):
        x, y, w, h = cv2.boundingRect(contour)
        if w and h > boxMin and w and h < boxMax:
            bluePucks.append(Puck(x, y, w, h, False, 1, 0))
            bluePucks[indexBlue].draw(frameVideo)
            indexBlue += 1

    redScore = True
    redScores = []

    blueScore = True
    blueScores = []

    scoringPucksRed = []
    scoringPucksBlue = []

    redValid = []
    blueValid = []

    # check pucks if valid or eligible for scoring points
    for i in range(len(redPucks)):
        redPucks[i].validate()
        if redPucks[i].valid:
            redValid.append(redPucks[i])

    for i in range(len(bluePucks)):
        bluePucks[i].validate()
        if bluePucks[i].valid:
            blueValid.append(bluePucks[i])

    # if pucks are eligible, check if a red puck is above all blue pucks and vice versa
    for i in range(len(redValid)):
        for j in range(len(blueValid)):
            if redValid[i].y < blueValid[j].y:
                redScore = True

            else:
                redScore = False
                break
        if redScore:
            scoringPucksRed.append(redValid[i])

    for i in range(len(blueValid)):
        for j in range(len(redValid)):
            if blueValid[i].y < redValid[j].y:
                blueScore = True
            else:
                blueScore = False
                break
        if blueScore:
            scoringPucksBlue.append(blueValid[i])

    ##################### RULES ########################

    # check if puck awards 4, 3, 2 or 1 point or none at all

    # hangerthreshold = 5

    if len(scoringPucksRed) is not 0:
        for i in range(0, len(scoringPucksRed)):
            point = getScore(scoringPucksRed[i], redScores)
            scoringPucksRed[i].setScore(point)

    if len(scoringPucksBlue) is not 0:
        for i in range(0, len(scoringPucksBlue)):
            point = getScore(scoringPucksBlue[i], blueScores)
            scoringPucksBlue[i].setScore(point)

    redCurrentScore = 0
    for i in range(len(redScores)):
        redCurrentScore += redScores[i]

    blueCurrentScore = 0
    for i in range(len(blueScores)):
        blueCurrentScore += blueScores[i]

    updatetime = time.time()
    if updatetime > settime + 5:
        if cv2.waitKey(10) & 0xFF == ord('s'):
            settime = time.time()
            redTotalScoreArray.append(redCurrentScore)
            blueTotalScoreArray.append(blueCurrentScore)
            round += 1
            blueTotalScore = 0
            redTotalScore = 0

            if starter == 0:
                starter = 1
            else:
                starter = 0

            if starter is 0:
                startString = ("RED STARTS")
                startColor = (0, 0, 255)
            else:
                startString = ("BLUE STARTS")
                startColor = (255, 0, 0)

            for i in range(len(blueTotalScoreArray)):
                blueTotalScore += blueTotalScoreArray[i]
                if blueTotalScore >= 15:
                    blueWins = True

            for i in range(len(redTotalScoreArray)):
                redTotalScore += redTotalScoreArray[i]
                if redTotalScore >= 15:
                    redWins = True
                    print('redWins', redWins)

    if cv2.waitKey(10) & 0xFF == ord('r'):
        round = 1
        blueTotalScoreArray = []
        redTotalScoreArray = []
        blueTotalScore = 0
        redTotalScore = 0
        redCurrentScore = 0
        blueCurrentScore = 0

        blueWins = False
        redWins = False

    score = np.zeros([200, 600, 3], dtype=np.uint8)
    score.fill(255)

    scoreboard = cv2.imread('scoreboard.png')
    width = int(scoreboard.shape[1] / 1.4)  # when straight
    height = int(scoreboard.shape[0] / 1.4)

    for i in range(0, len(redPucks)):

        if len(redPucks) is not 0:
            cv2.putText(scorewindow, str(redPucks[i].score), (redPucks[i].x, redPucks[i].y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for i in range(0, len(bluePucks)):

        if len(bluePucks) is not 0:
            cv2.putText(scorewindow, str(bluePucks[i].score), (bluePucks[i].x, bluePucks[i].y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    currentTextRedX = int(framew / 10 * 3)
    currentTextBlueX = int(framew / 10 * 7)
    totalTextRedX = int(framew / 10 * 4)
    totalTextBlueX = int(framew / 10 * 6)
    roundText = int(framew / 2)
    textY = int(frameh/10)
    paddingh = int(frameh/10)
    paddingw = int(framew/20)

    cv2.putText(scorewindow, str(redCurrentScore), (currentTextRedX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

    cv2.putText(scorewindow, str(redTotalScore), (totalTextRedX, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.putText(scorewindow, str(blueCurrentScore), (currentTextBlueX+paddingw, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
    cv2.putText(scorewindow, str(blueTotalScore), (totalTextBlueX, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    cv2.putText(scorewindow, str(round), (roundText, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

    cv2.putText(scorewindow, str(startString), (int(framew/10*3.5), textY+paddingh), cv2.FONT_HERSHEY_SIMPLEX, 1, startColor, 2)

    if blueWins:
        cv2.rectangle(scorewindow, (int(framew/4), int(frameh/4)), (int(framew/4*3), int(frameh/4*3)), (255, 0, 0), 4)
        cv2.putText(scorewindow, "BLUE WINS", (int(framew/10*3.5), int(frameh/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if redWins:
        cv2.rectangle(scorewindow, (int(framew / 4), int(frameh / 4)), (int(framew / 4 * 3), int(frameh / 4 * 3)),
                      (0, 0, 255), 4)
        cv2.putText(scorewindow, "RED WINS", (int(framew / 10 * 3.5), int(frameh / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    cv2.imshow("Feed", frameVideo)
    cv2.imshow("Score Window", scorewindow)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        frame.release()
        cv2.destroyAllWindows()

