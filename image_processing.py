import cv2
import numpy as np
import os

jump_thresh = 20

def preprocessing(frame):
    frame = getBlackOnly(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def drawLines(img, lines):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)


def get_strong_lines(edged):
    strong_lines = np.zeros([4, 1, 2])

    minLineLength = 2
    maxLineGap = 10

    lines = cv2.HoughLines(edged, 1, np.pi/180, 10, minLineLength, maxLineGap)

    n2 = 0
    for n1 in range(0, len(lines)):
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho *= -1
                    theta -= np.pi
                closeness_rho = np.isclose(
                    rho, strong_lines[0:n2, 0, 0], atol=10)
                closeness_theta = np.isclose(
                    theta, strong_lines[0:n2, 0, 1], atol=np.pi/36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
    return strong_lines


def process_line(img, frame):
    #cv2.imwrite('grid.jpg', img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    #lines = cv2.HoughLines(image=edges, rho = 1.1,theta = 1*np.pi/180, threshold = 135,minLineLength = 100,maxLineGap = 50)
    lines = cv2.HoughLinesP(image=edges, rho=1.2, theta=1 *
                            np.pi/180, threshold=160, minLineLength=100, maxLineGap=50)
    #print("p shape: ", lines.shape)
    # To ensure that the returned lines are not None
    # If lines are None, we can't use .any()
    #lines = get_strong_lines(edges)

    lines = np.array(lines)
    if not lines.any():
        return

    a, b, c = lines.shape

    for i in range(a):
        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i]
                 [0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

    #drawLines(frame, lines)

    key_input = cv2.waitKey(1)
    '''
    if key_input == ord('w'):
        val += 0.1
        print(val)
    elif key_input == ord('e'):
        val -= 0.1
        print(val)
    '''

# remove points that are close proximity with each other (except one)


def get_strong_points(points, frame=None):
    # FIRST HALF: single out points with multiple detected points

    # by default points are sorted by the x axis; due to method used in thresholding
    first_res = np.zeros([points.shape[0], 2])
    point_count = 0
    idx = 0
    # x_baseline = points[0][0]
    for i in range(len(points)):
        # group of similar x [idx,end_point)
        # refer to line 109 for end_point
        if i == len(points)-1 or points[i+1, 0]-points[i, 0] >= jump_thresh:
            # end_point is not included in the group
            end_point = min(i+1, len(points))
            '''
            cv2.line(frame, (points[idx][0], 0),
                     (points[idx][0], 1000), (255, 255, 0), 2)
            cv2.line(frame, (points[end_point-1][0], 0),
                     (points[end_point-1][0], 1000), (255, 0, 0), 2)
            '''
            if end_point-idx+1 <= 20:
                idx = i+1
                continue
            # at this point I can either slice it or keep a record of the slicing index;
            # not sure which is more efficient in python
            points[idx:end_point] = points[points[idx:end_point, 1].argsort()+idx]
            _idx = idx
            for j in range(idx, end_point):
                if j == end_point-1 or points[j+1, 1]-points[j, 1] >= jump_thresh:
                    '''
                    cv2.line(frame, (points[idx][0], 0),
                     (points[idx][0], 1000), (255, 255, 0), 2)
                    cv2.line(frame, (points[i][0], 0),
                            (points[i][0], 1000), (255, 0, 0), 2)
                    '''
                    mid_point = (int)((_idx+j)/2)
                    first_res[point_count] = points[mid_point]
                    point_count += 1
                    _idx = j
            idx = i+1
    first_res =  first_res[:point_count]

    if __debug__:
        for point in first_res.astype(int):
            cv2.circle(frame, point, 5, (255, 0, 0), 2)

    # SECOND HALF: single out points that are part of the edges of grid;
    # METHOD: check how many x and y coordinates are similar 
    # Because in a 3 by 3 grid within a square, a point should have 3 other points where x are similar and 3 other points where y are similar
    sim_num_thresh = 3
    sim_jump_thresh = 50

    # filter small similar x size
    first_res = first_res[first_res[:,0].argsort()]
    second_res = np.zeros(first_res.shape)

    idx = 0
    for i in range(len(first_res)):
        sim_count = 0
        base_x = first_res[i][0]
        for j in range(len(first_res)):
            if i==j:
                continue
            else:
                if abs(base_x-first_res[j][0])<=sim_jump_thresh:
                    sim_count+=1
                if sim_count>=sim_num_thresh:
                    second_res[idx] = first_res[i]
                    idx+=1
                    break
    
    if __debug__:
        for point in second_res.astype(int):
            cv2.circle(frame, point, 5, (0, 0, 255), 2)
    ##################### X coord FINISH #####################
    second_res = second_res[:idx]
    second_res = second_res[second_res[:,1].argsort()]

    # filter small similar y size
    res = np.zeros(second_res.shape)
    idx = 0

    for i in range(len(second_res)):
        sim_count = 0
        base_y = second_res[i][1]
        for j in range(len(second_res)):
            if i==j:
                continue
            else:
                if abs(base_y-second_res[j][1])<=sim_jump_thresh:
                    sim_count+=1
                if sim_count>=sim_num_thresh:
                    res[idx] = second_res[i]
                    idx+=1
                    break
    res = res[:idx]
    ##################### Y coord FINISH #####################
    return res

def harris_corner_detection(img, frame=None):
    # to get a sense of how much a value is; debugging purposes
    cv2.rectangle(frame, (0, 0), (jump_thresh, jump_thresh), (0, 0, 255), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find Harris corners
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_blur = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # honesty don't understand the significance of this
    blurred = np.float32(thresh_blur)
    dst = cv2.cornerHarris(blurred, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    thresh = 0.4*dst.max()
    points = np.zeros([dst.shape[0]*dst.shape[1], 2]).astype(int)
    idx = 0

    # Iterate through all the corners and draw them on the image (if they pass the threshold)
    for i in range(0, dst.shape[1]):
        for j in range(0, dst.shape[0]):
            if (dst[j, i] > thresh):
                # image, center pt, radius, color, thickness
                points[idx] = i, j
                idx += 1

    points = points[:idx]

    points = get_strong_points(points, frame).astype(int)
    if __debug__:
        for point in points:
            cv2.circle(frame, point, 10, (0, 255, 0), 3)
    
    return points
    '''
    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if(dst[j,i] > thresh):
                # image, center pt, radius, color, thickness                
                cv2.circle(frame, (i, j), 1, (0,255,0), 1)
    '''

    '''
    for i in range(len(res)):
        cv2.circle(img,(res[i,0],res[i,1]), 4, (255,0,0), 2)
    
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    '''


def getBlackOnly(img):
    # Convert to float and divide by 255:
    imgFloat = img.astype(np.float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)
    # Threshold image:
    binaryThresh = 110
    _, binaryImage = cv2.threshold(
        kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
    return binaryImage
