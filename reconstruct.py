# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    color_image = cv2.imread("images/pattern001.jpg")
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)
    corr_img = np.zeros((proj_mask.shape[0], proj_mask.shape[1], 3))
    RGB_values = []

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
                               fx=scale_factor, fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        scan_bits[on_mask] = scan_bits[on_mask] | bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            [x_p, y_p] = binary_codes_ids_codebook[scan_bits[y,x]]

            if x_p >= 1279 or y_p >= 799: # filter
                continue
            else:
                projector_points.append([[x_p, y_p]])
                camera_points.append([[x/2.0, y/2.0]])
                corr_img[y,x,2] = np.uint8((x_p/1280.0)*255)
                corr_img[y,x,1] = np.uint8((y_p/960.0)*255)
                RGB_values.append([color_image[y, x]])

    cv2.imwrite("Results/correspondances.png", np.array(corr_img).astype('uint8'))

    # now that we have 2D-2D correspondances, we can triangulate 3D points!
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    camera_points = np.array(camera_points).astype(np.float32)
    projector_points = np.array(projector_points).astype(np.float32)
    normalized_cam_points = cv2.undistortPoints(camera_points, camera_K, camera_d)
    normalized_proj_points = cv2.undistortPoints(projector_points, projector_K, projector_d)
    P1 = np.eye(4).astype(np.float32)[:3]
    P2 = np.concatenate((projector_R, projector_t), axis=1)
    out = cv2.triangulatePoints(P1, P2, normalized_cam_points, normalized_proj_points)
    out = np.transpose(out)
    points_3d = cv2.convertPointsFromHomogeneous(out)

    RGB_values = np.array(RGB_values).astype(np.float32)
    print RGB_values.shape, points_3d.shape

    len_points_3d = points_3d.shape[0]
    updated_points_3d = []
    updated_RGB_values = []
    for i in range(len(points_3d)):
        if points_3d[i][0][2] > 200 and points_3d[i][0][2] < 1400:
            updated_points_3d.append(points_3d[i])
            updated_RGB_values.append(RGB_values[i])
    updated_points_3d = np.array(updated_points_3d)
    updated_RGB_values = np.array(updated_RGB_values)

    print("write output color point cloud", updated_points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p,c in zip(updated_points_3d, updated_RGB_values):
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],c[0,2], c[0,1], c[0,0]))

    return updated_points_3d
    
def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud", points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return points_3d#, camera_points, projector_points
    
if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)

