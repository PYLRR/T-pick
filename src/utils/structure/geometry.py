import numpy as np
import geopy.distance

M_PER_DEG = 110_574

def distance_point_point(P, A, fast=False):
    if fast:
        return np.sqrt(np.sum((A - P) ** 2)) * M_PER_DEG
    return geopy.distance.geodesic(A, P).m

def distance_point_segment(P, A, B, fast=False):
    AP = P - A
    AB = B - A

    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        t = 0
    else:
        t = max(0, min(1,np.dot(AP, AB) / AB_squared))

    closest = A + t * AB
    d = distance_point_point(closest, P, fast)

    return closest, d

def segment_from_points_pairs(points):
    return [(points[i], points[i+1]) for i in range(0, len(points)-1)]

def get_azimuth(pos1, pos2):
    dlat, dlon = pos2[0] - pos1[0], pos2[1] - pos1[1]
    r = np.arctan2(dlon, dlat)
    r = r if r > 0 else r + 2 * np.pi
    return r