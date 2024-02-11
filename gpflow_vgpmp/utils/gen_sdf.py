# The following were taken from https://github.com/liruiw/OMG-Planner and adapted

import multiprocessing
import os
import sys

import numpy as np

PADDING = 20

__all__ = "gen_sdf"

path_sdfgen = "path-to-your-VTK-to-build-SDF-from-obj"


def generate_sdf(path_to_sdfgen, obj_filename, delta, padding):
    """ Converts mesh to an sdf object """

    # create the SDF using binary tools, avoid overwrite
    obj_filename = obj_filename[:-1]
    dummy_cmd = "cp %s %s" % (
        obj_filename, obj_filename.replace(".obj", ".dummy.obj"))
    os.system(dummy_cmd)
    sdfgen_cmd = '%s "%s" %f %d' % (
        path_to_sdfgen,
        obj_filename.replace(".obj", ".dummy.obj"),
        delta,
        padding,
    )
    print("SDF Command: %s" % sdfgen_cmd)
    os.system(sdfgen_cmd)

    sdf_filename = obj_filename.replace(".obj", ".dummy.sdf")
    sdf_dim_filename = obj_filename.replace(".obj", "_vgpmp.sdf")
    rename_cmd = "mv %s %s" % (sdf_filename, sdf_dim_filename)
    os.system(rename_cmd)
    clean_cmd = "rm %s; rm %s" % (
        obj_filename.replace(".obj", ".dummy.obj"),
        sdf_filename.replace(".sdf", ".vti"),
    )
    os.system(clean_cmd)
    print("Rename Output Location", sdf_dim_filename)
    return


def do_job_convert_obj_to_sdf(obj_id_dim_padding):
    x, dim, extra_padding = obj_id_dim_padding
    file = os.path.join(prefix, str(file_list_all[x]), surfix)

    extent_file = file.replace(".obj", ".extent.txt")[:-1]
    sdf_file = file.replace(".obj", ".sdf")
    extent = np.loadtxt(extent_file).astype(np.float32)
    REG_SIZE = 0.2
    scale = np.max(extent) / REG_SIZE
    extra_padding = min(int(extra_padding * scale), 30)
    dim = max(min(dim * scale, 100), 32)
    delta = np.max(extent) / dim
    padding = extra_padding  #
    # dim = dim + extra_padding * 2
    generate_sdf(path_sdfgen, file, delta, padding)


def gen_sdf(random_paths=None, dimension=32):  # modify dimension
    global prefix, surfix, file_list_all
    surfix = "model_normalized.obj"

    file_list_all = random_paths
    prefix = ""

    object_numbers = len(file_list_all)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores // 3)
    padding = 20  # one with padding, one without
    param_list = [
        list(a)
        for a in zip(
            range(object_numbers), [dimension] * object_numbers, [padding] * object_numbers
        )
    ]
    pool.map(do_job_convert_obj_to_sdf, param_list)  # 32, 40


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_list_all = [sys.argv[1]]
        prefix = "model_normalized.obj"
        surfix = ""
        dim = 32
        pad = 8
        if len(sys.argv) > 2:
            pad = int(sys.argv[2])

        do_job_convert_obj_to_sdf((0, dim, pad))
