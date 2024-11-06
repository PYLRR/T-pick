import numpy as np

from utils.structure.geometry import distance_point_point, segment_from_points_pairs, distance_point_segment


class Structure():
    def __init__(self, pos, name):
        self.pos, self.name = pos, name

    def get_distance(self, point, fast=False):
        return None

    def get_type(self):
        return self.__class__.__name__

    def __str__(self):
        return f"{self.get_type()}-{self.name}"

class Seamount(Structure):
    def get_distance(self, point, fast=False):
        return self.pos, distance_point_point(self.pos, point, fast)

class Ridge(Structure):
    def get_distance(self, point, fast=False):
        segments = segment_from_points_pairs(self.pos)
        proj_dist = [distance_point_segment(point, s[0], s[1], fast) for s in segments]
        closest = np.argmin([p[1] for p in proj_dist])
        proj, dist = proj_dist[closest]
        return proj, dist

    def get_distance_from_start(self, point, fast=False):
        # note : we consider the first point to be the "start"
        segments = segment_from_points_pairs(self.pos)
        proj_dist = [distance_point_segment(point, s[0], s[1], fast) for s in segments]
        closest = np.argmin([p[1] for p in proj_dist])
        proj, _ = proj_dist[closest]
        d = np.sum([distance_point_point(segments[i][0], segments[i][1], fast) for i in range(closest)])
        d += distance_point_point(segments[closest][0], proj, fast)
        return d

class TransformFault(Structure):
    def get_distance(self, point, fast=False):
        proj, dist = distance_point_segment(point, self.pos[0], self.pos[1], fast)
        return proj, dist

    def get_distance_from_start(self, point, fast=False):
        # note : we consider the first point to be the "start"
        proj, dist = distance_point_segment(point, self.pos[0], self.pos[1], fast)
        d = distance_point_point(self.pos[0], proj, fast)
        return d

class StructureList():
    def __init__(self, structures):
        self.structures = []
        for s in structures:
            self.add_structure(s)

    def add_structure(self, structure):
        assert isinstance(structure, Structure), "StructureList must be fed with Structures"
        self.structures.append(structure)

    def get_closest(self, point):
        dist_min, struct_min, point_min = np.inf, None, None
        for struct in self.structures:
            point_, dist = struct.get_distance(point)
            if dist < dist_min:
                dist_min, struct_min, point_min = dist, struct, point_
        return struct_min, point_min, dist_min


def load_xy_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
    res = []
    for line in lines:
        if line[0]==">":
            continue # comment
        res.append(np.array(line.split(), dtype=np.float32)[::-1])
    return np.array(res)