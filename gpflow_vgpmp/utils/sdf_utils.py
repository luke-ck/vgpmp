import argparse
import pickle
import time
from subprocess import call

import numpy as np
import tensorflow as tf

__all__ = 'sdf_utils'


def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (end - start) * 1000.0))

        return ret

    return wrap


class SignedDistanceField(object):
    """Data is stored in the following way
        data[x, y, z]
    update to integrate tensorflow
    """

    def __init__(self, data, origin, delta):
        self.data = data
        self.nx, self.ny, self.nz = data.shape
        self.origin = origin
        self.delta = delta
        self.min_coords = origin
        self.max_coords = self.origin + delta * np.array(data.shape)

        self.data_tf = tf.constant(self.data, dtype=tf.float64)
        self.delta_tf = tf.constant([delta], dtype=tf.float64)
        self.origin_tf = tf.constant(self.origin, dtype=tf.float64)
        self.min = tf.reshape(tf.constant([0, 0, 0], dtype=tf.int64), (1, 3))
        self.max = tf.reshape(tf.constant(
            [self.nx - 1, self.ny - 1, self.nz - 1], dtype=tf.int64), (1, 3))

    def resize(self, ratio):
        self.data *= ratio
        self.data_tf *= ratio

        self.delta *= ratio
        self.origin *= ratio

        self.delta_tf *= ratio
        self.origin_tf *= ratio

    def _rel_pos_to_idxes(self, rel_pos):
        i_min = np.array([0, 0, 0], dtype=np.int)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int)
        idx = ((rel_pos - self.origin) / self.delta).astype(int)
        return np.clip(idx, i_min, i_max)

    def _rel_pos_to_idxes_tf(self, rel_pos):

        return tf.clip_by_value(tf.cast(
            ((rel_pos - self.origin_tf) / self.delta_tf), dtype=tf.int64
        ), self.min, self.max)

    def get_distance(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        return self.data[idxes[..., 0], idxes[..., 1], idxes[..., 2]]

    def get_distance_tf(self, rel_pos):
        idxes = self._rel_pos_to_idxes_tf(rel_pos)
        assert idxes.shape[0] == rel_pos.shape[0]
        return tf.gather_nd(self.data_tf, idxes)

    def get_distance_grad(self, rel_pos):
        idxes = self._rel_pos_to_idxes(rel_pos)
        i_min = np.array([0, 0, 0], dtype=np.int)
        i_max = np.array([self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.int64)
        neighbor1 = np.clip(idxes + 1, i_min, i_max)
        neighbor2 = np.clip(idxes - 1, i_min, i_max)
        dx = (
                     self.data[neighbor1[..., 0], idxes[..., 1], idxes[..., 2]]
                     - self.data[neighbor2[..., 0], idxes[..., 1], idxes[..., 2]]
             ) / (2 * self.delta)

        dy = (
                     self.data[idxes[..., 0], neighbor1[..., 1], idxes[..., 2]]
                     - self.data[idxes[..., 0], neighbor2[..., 1], idxes[..., 2]]
             ) / (2 * self.delta)

        dz = (
                     self.data[idxes[..., 0], idxes[..., 1], neighbor1[..., 2]]
                     - self.data[idxes[..., 0], idxes[..., 1], neighbor2[..., 2]]
             ) / (2 * self.delta)
        return np.stack([dx, dy, dz], axis=-1)

    def get_distance_grad_tf(self, rel_pos):
        idxes = tf.cast(self._rel_pos_to_idxes_tf(rel_pos), dtype=tf.int64)

        neighbor1 = tf.math.minimum(
            tf.math.maximum(idxes + 1, self.min), self.max)
        neighbor2 = tf.math.minimum(tf.math.maximum(
            idxes - 1, self.min), self.max)  # P x 3

        dx1 = tf.gather_nd(self.data_tf, tf.stack(
            [neighbor1[..., 0], idxes[..., 1], idxes[..., 2]], axis=-1))
        dx2 = tf.gather_nd(self.data_tf, tf.stack(
            [neighbor2[..., 0], idxes[..., 1], idxes[..., 2]], axis=-1))
        dy1 = tf.gather_nd(self.data_tf, tf.stack(
            [idxes[..., 0], neighbor1[..., 1], idxes[..., 2]], axis=-1))
        dy2 = tf.gather_nd(self.data_tf, tf.stack(
            [idxes[..., 0], neighbor2[..., 1], idxes[..., 2]], axis=-1))
        dz1 = tf.gather_nd(self.data_tf, tf.stack(
            [idxes[..., 0], idxes[..., 1], neighbor1[..., 2]], axis=-1))
        dz2 = tf.gather_nd(self.data_tf, tf.stack(
            [idxes[..., 0], idxes[..., 1], neighbor2[..., 2]], axis=-1))
        dx = (
                     dx1 - dx2
             ) / (2 * self.delta)

        dx = tf.where(dx == 0, tf.cast(0.1, dtype=tf.float64), dx)

        dy = (
                     dy1 - dy2
             ) / (2 * self.delta)
        dy = tf.where(dy == 0, tf.cast(0.1, dtype=tf.float64), dy)

        dz = (
                     dz1 - dz2
             ) / (2 * self.delta)

        dz = tf.where(dz == 0, tf.cast(0.1, dtype=tf.float64), dz)
        return tf.stack([dx, dy, dz], axis=-1)

    def trim(self, dim, center=True):
        x_padding = (self.nx - dim) / 2
        y_padding = (self.ny - dim) / 2
        z_padding = (self.nz - dim) / 2
        if center:
            assert min(min(x_padding, y_padding), z_padding) >= 0
            self.data = self.data[
                        x_padding: self.nx - x_padding,
                        y_padding: self.ny - y_padding,
                        z_padding: self.nz - z_padding,
                        ]
        self.data = self.data[:dim, :dim, :dim]  # even odd
        self.origin = (
                self.origin +
                np.array([x_padding, y_padding, z_padding]) * self.delta
        )
        self.nx, self.ny, self.nz = dim, dim, dim
        self.max_coords = self.origin + self.delta * np.array(self.data.shape)
        self.data_tf = tf.constant(self.data, dtype=tf.float64)
        self.origin_tf = tf.constant(self.origin)
        self.max = tf.constant(
            [self.nx - 1, self.ny - 1, self.nz - 1], dtype=tf.float64)

    def dump(self, pkl_file):
        data = {"data": self.data, "origin": self.origin, "delta": self.delta}
        pickle.dump(data, open(pkl_file, "wb"), protocol=2)

    def visualize(self, max_dist=0.1):
        try:
            from mayavi import mlab
        except ImportError:
            print("mayavi is not installed!")

        figure = mlab.figure("Signed Density Field")
        # The dimensions will be expressed in cm for better visualization.
        SCALE = 100
        data = np.copy(self.data)
        data = np.minimum(max_dist, data)
        xmin, ymin, zmin = SCALE * self.origin
        xmax, ymax, zmax = SCALE * self.max_coords
        delta = SCALE * self.delta
        xi, yi, zi = np.mgrid[xmin:xmax:delta,
                     ymin:ymax:delta, zmin:zmax:delta]
        data[data <= 0] -= 0.5  # 0.2

        data = -data
        xi = xi[: data.shape[0], : data.shape[1], : data.shape[2]]
        yi = yi[: data.shape[0], : data.shape[1], : data.shape[2]]
        zi = zi[: data.shape[0], : data.shape[1], : data.shape[2]]
        grid = mlab.pipeline.scalar_field(xi, yi, zi, data)
        vmin = np.min(data)
        vmax = np.max(data)

        mlab.pipeline.volume(grid, vmin=vmin, vmax=(vmax + vmin) / 2)
        mlab.axes()
        mlab.show()

    @classmethod
    def from_sdf(cls, sdf_file):

        with open(sdf_file, "r") as file:
            lines = file.readlines()
            nx, ny, nz = map(int, lines[0].split(" "))
            x0, y0, z0 = map(float, lines[1].split(" "))
            delta = float(lines[2].strip())
            data = np.zeros([nx, ny, nz])
            for i, line in enumerate(lines[3:]):
                idx = i % nx
                idy = int(i / nx) % ny
                idz = int(i / (nx * ny))
                val = float(line.strip())
                data[idx, idy, idz] = val
        return cls(data, np.array([x0, y0, z0]), delta)

    @classmethod
    def from_pkl(cls, pkl_file):
        data = pickle.load(open(pkl_file, "r"))
        return cls(data["data"], data["origin"], data["delta"])

    @tf.custom_gradient
    def check_gradients_sdf(self, x):

        def grad(upstream):
            tf.print(upstream[3:])
            return upstream

        return x, grad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Path to .sdf or .pkl file.",
        default="../robot_model/panda_arm/hand_finger.sdf",
        required=False,
    )
    parser.add_argument(
        "-v", action="store_true", help="Visualize signed density field."
    )
    parser.add_argument(
        "-n", action="store_true", help="Visualize signed density field as voxels."
    )
    parser.add_argument("--export", "-e", help="Export to a pickle file.")

    args = parser.parse_args()
    filename = args.file
    if filename.endswith(".sdf"):
        sdf = SignedDistanceField.from_sdf(filename)
    elif filename.endswith(".pkl"):
        sdf = SignedDistanceField.from_pkl(filename)

    if sdf is not None:
        print(
            "sdf info:",
            "delta: ", sdf.delta, "\n",
            "shape: ", sdf.data.shape, "\n",
            "origin: ", sdf.origin, "\n",
            "cells with signed distance > 0.01: ", (sdf.data > 0.01).sum(), "\n",
            "total volume: ", sdf.delta * np.array(sdf.data.shape),
        )

    if args.v:
        sdf.visualize()
    if args.export:
        sdf.dump(args.export)
    if args.n:
        import pymesh

        voxel = (np.abs(sdf.data) <= 0.01).astype(np.int)
        pymesh.save_mesh("test.obj", voxel)
        call(["meshlab", "test.obj"])
