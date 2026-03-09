import os
import cv2
import numpy as np
import math


def brisk_feature_detect(img):
    brisk = cv2.BRISK_create(1, 2)
    (kp, des) = brisk.detectAndCompute(img, None)

    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    cv2.imshow("keypoints", img2)
    cv2.waitKey(0)


def fast_feature_detect(img):
    fast = cv2.FastFeatureDetector_create(threshold=1)

    kp = fast.detect(img, None)

    br = cv2.BRISK_create(10000000, 2)
    kp, des = br.compute(img, kp)

    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(kp))

    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    cv2.imshow("keypoints", img2)
    cv2.waitKey(0)


def feature_detect(img):

    orb = cv2.ORB_create(nfeatures=10000000, scoreType=cv2.ORB_FAST_SCORE)

    kp = orb.detect(img, None)

    kp, des = orb.compute(img, kp)

    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    cv2.imshow("keypoints", img2)
    cv2.waitKey(0)


def fast_feature_match(img1, img2):

    fast = cv2.FastFeatureDetector_create(threshold=60)

    MIN_MATCHES = 0
    cap = img1
    model = img2
    br = cv2.BRISK_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_m = fast.detect(img2, None)
    kp_model, des_model = br.compute(img2, kp_m)

    kp_f = fast.detect(img1, None)
    kp_frame, des_frame = br.compute(img1, kp_f)

    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > MIN_MATCHES:
        cap = cv2.drawMatches(
            model, kp_model, cap, kp_frame, matches[:MIN_MATCHES], 0, flags=2
        )
        cv2.imshow("frame", cap)
        cv2.waitKey(0)
    else:
        print(
            "Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES)
        )


def feature_match(img1, img2):
    MIN_MATCHES = 20
    cap = img1
    model = img2
    orb = cv2.ORB_create(nfeatures=10000000, scoreType=cv2.ORB_FAST_SCORE)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    min_x = math.inf
    min_y = math.inf
    max_x = 0
    max_y = 0

    all_matches_pos = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp_model[img1_idx].pt

        if x1 < min_x:
            min_x = x1
        if x1 > max_x:
            max_x = x1

        if y1 < min_y:
            min_y = y1
        if y1 > max_y:
            max_y = y1

    h = max_y - min_y
    w = max_x - min_x

    list_grouped_matches = []

    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp_model[img1_idx].pt
        (x2, y2) = kp_frame[img2_idx].pt
        i = 0
        found = False

        for existing_match in list_grouped_matches:
            if (
                abs(existing_match[0][1][0] - x2) < w
                and abs(existing_match[0][1][1] - y2) < h
            ):
                list_grouped_matches[i].append(((int(x1), int(y1)), (int(x2), int(y2))))
                found = True
                break
            i += 1

        if not found:
            tmp = list()
            tmp.append(((int(x1), int(y1)), (int(x2), int(y2))))
            list_grouped_matches.append(list(list(list(tmp))))

    list_grouped_matches_filtered = []

    for match_group in list_grouped_matches:
        if len(match_group) >= 4:
            list_grouped_matches_filtered.append(match_group)

    corners = cv2.goodFeaturesToTrack(model, 3, 0.01, 20)
    corners = np.int0(corners)

    upper_left = corners[1][0]
    upper_right = corners[0][0]
    down = corners[2][0]

    max_x = 0
    max_y = 0
    min_x = math.inf
    min_y = math.inf

    for cr in corners:
        x1 = cr[0][0]
        y1 = cr[0][1]

        if x1 < min_x:
            min_x = x1
        if x1 > max_x:
            max_x = x1

        if y1 < min_y:
            min_y = y1
        if y1 > max_y:
            max_y = y1

    origin = (int((max_x + min_x) / 2), int((min_y + max_y) / 2))

    doors_actual_pos = []
    for match in list_grouped_matches_filtered:

        index1, index2 = calculate_best_matches_with_modulus_angle(match)

        pos1_model = match[index1][0]
        pos2_model = match[index2][0]

        pos1_cap = match[index1][1]
        pos2_cap = match[index2][1]

        cap_size = [(pos1_cap[0] - pos2_cap[0]), (pos1_cap[1] - pos2_cap[1])]
        model_size = [(pos1_model[0] - pos2_model[0]), (pos1_model[1] - pos2_model[1])]
        scaled_upper_left = upper_left
        scaled_upper_right = upper_right
        scaled_down = down
        scaled_pos1_model = pos1_model

        pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
        pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

        ang = math.degrees(angle(pt1, pt2))

        new_upper_left = rotate(origin, scaled_upper_left, math.radians(ang))
        new_upper_right = rotate(origin, scaled_upper_right, math.radians(ang))
        new_down = rotate(origin, scaled_down, math.radians(ang))

        new_pos1_model = rotate(origin, scaled_pos1_model, math.radians(ang))

        offset = (new_pos1_model[0] - pos1_model[0], new_pos1_model[1] - pos1_model[1])

        move_dist = (pos1_cap[0] - pos1_model[0], pos1_cap[1] - pos1_model[1])

        moved_new_upper_left = (
            int(new_upper_left[0] + move_dist[0] - offset[0]),
            int(new_upper_left[1] + move_dist[1] - offset[1]),
        )
        moved_new_upper_right = (
            int(new_upper_right[0] + move_dist[0] - offset[0]),
            int(new_upper_right[1] + move_dist[1] - offset[1]),
        )
        moved_new_down = (
            int(new_down[0] + move_dist[0] - offset[0]),
            int(new_down[1] + move_dist[1] - offset[1]),
        )

        img = cv2.circle(
            cap, moved_new_upper_left, radius=4, color=(0, 0, 0), thickness=5
        )
        img = cv2.circle(
            cap, moved_new_upper_right, radius=4, color=(0, 0, 0), thickness=5
        )
        img = cv2.circle(cap, moved_new_down, radius=4, color=(0, 0, 0), thickness=5)

    for match in list_grouped_matches_filtered:

        img = cv2.circle(
            cap,
            (match[0][1][0], match[0][1][1]),
            radius=4,
            color=(0, 0, 0),
            thickness=5,
        )

    if len(matches) > MIN_MATCHES:

        cap = cv2.drawMatches(
            model, kp_model, cap, kp_frame, matches[:MIN_MATCHES], 0, flags=2
        )

        cv2.imshow("frame", cap)
        cv2.waitKey(0)

    else:
        print(
            "Not enough matches have been found - %d/%d" % (len(matches), MIN_MATCHES)
        )


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def scale_model_point_to_origin(origin, point, x_scale, y_scale):
    dx, dy = (point[0] - origin[0], point[1] - origin[1])
    return (dx * x_scale, dy * y_scale)


def calculate_best_matches_with_angle_checks(match_list):
    index1 = 0
    index2 = 0
    list_of_angles = []

    i = 0
    for match1 in match_list:
        j = 0
        for match2 in match_list:
            index1 = i
            index2 = j

            pos1_model = match_list[index1][0]
            pos2_model = match_list[index2][0]

            pos1_cap = match_list[index1][1]
            pos2_cap = match_list[index2][1]

            pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
            pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

            if pt1 == pt2 or pt1 == (0, 0) or pt2 == (0, 0):
                continue

            ang = math.degrees(angle(pt1, pt2))

            list_of_angles.append(ang)

            j += 1
        i += 1

    average_angle = average(list_of_angles)
    current_best = math.inf

    _index1 = 0
    _index2 = 0

    i = 0
    for match1 in match_list:
        j = 0
        for match2 in match_list:
            index1 = i
            index2 = j

            pos1_model = match_list[index1][0]
            pos2_model = match_list[index2][0]

            pos1_cap = match_list[index1][1]
            pos2_cap = match_list[index2][1]

            pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
            pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

            if pt1 == pt2 or pt1 == (0, 0) or pt2 == (0, 0):
                continue

            ang = math.degrees(angle(pt1, pt2))

            diff = average_angle - ang

            if diff < current_best:
                current_best = diff
                _index1 = i
                _index2 = j
            j += 1
        i += 1

    return _index1, _index2


def average(lst):
    return sum(lst) / len(lst)


def calculate_best_matches_with_modulus_angle(match_list):
    index1 = 0
    index2 = 0
    best = math.inf

    i = 0
    for match1 in match_list:
        j = 0
        for match2 in match_list:

            pos1_model = match_list[i][0]
            pos2_model = match_list[j][0]

            pos1_cap = match_list[i][1]
            pos2_cap = match_list[j][1]

            pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
            pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

            if pt1 == pt2 or pt1 == (0, 0) or pt2 == (0, 0):
                continue

            ang = math.degrees(angle(pt1, pt2))
            diff = ang % 30

            if diff < best:
                best = diff
                index1 = i
                index2 = j

            j += 1
        i += 1
    return index1, index2


def calculate_best_matches_distance(match_list):
    max = 0
    index1 = 0
    index2 = 0

    i = 0
    for match1 in match_list:
        j = 0
        for match2 in match_list:

            dist = abs(match1[1][0] - match2[1][0]) + abs(match1[1][1] - match2[1][1])
            if dist > max:
                max = dist
                index1 = i
                index2 = j
            j += 1
        i += 1

    return index1, index2


if __name__ == "__main__":
    door_image_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Models/Doors/door.png"
    )
    example_image_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Examples/example.png"
    )

    img1 = cv2.imread(example_image_path, 0)
    img2 = cv2.imread(door_image_path, 0)
    feature_match(img1, img2)
