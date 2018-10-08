import sys
import os
import json
import pickle
import argparse
import datetime, time
import random
import numpy as np
from easydict import EasyDict as edict

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule


class Timer(object):
    '''
    A simple timer.
    '''
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time   = 0.0
        self.calls        = 0
        self.start_time   = 0.0
        self.duration     = 0.0
        self.average_time = 0.0

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
    
    def elapsed(self):
        return str(datetime.timedelta(seconds=int(self.total_time)))

def clip_by_boundary(points, x_min = 0.0, x_max = 70.4, y_min = -40.0, y_max = 40.0, z_min = -3.5, z_max = 3.5):
    bound_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
    bound_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
    bound_z = np.logical_and(points[:, 2] >= z_min, points[:, 2] <= z_max)

    bound = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
    return points[bound]

def clip_by_projection(points, P, y2, x2, y1=0, x1=0):
    pts_2D = projectToImage_kitti(points[:,0:3].transpose(), P)
    pts_2D = pts_2D.transpose()
    clipped_idx = (pts_2D[:, 0] <= x2+500) & (pts_2D[:, 0] >= x1-500) & (pts_2D[:, 1] <= y2+150) & (pts_2D[:, 1] >= y1-150)
    return points[clipped_idx]

def projectToImage_kitti(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D

mod = SourceModule("""
__global__ void voxelize_cu(float *voxel, 
							float* points, 
							int vx_stride_x, 
							int vx_stride_y, 
							int points_stride, 
							int N,
							int h,
							int w,
							float q_x, 
							float q_y, 
							float q_z)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < N) {
    int coord_x = 0;
    int coord_y = 0;
    int coord_z = 0;

    float* x = (float*)((char*)points + tid*points_stride) + 0;
    float* y = (float*)((char*)points + tid*points_stride) + 1;
    float* z = (float*)((char*)points + tid*points_stride) + 2;

    *y += 40.0;
    *z += 3.5;

    coord_x = floor(*x/q_x);
    coord_y = floor(*y/q_y);
    coord_z = floor(*z/q_z);

    float* p = (float*)((char*)voxel + coord_x*vx_stride_x + coord_y*vx_stride_y) + coord_z;
        *p = 1.0;    
    }
}
""")



gpu_voxelize = mod.get_function("voxelize_cu")

def gpu_voxel(points, 
		      q_lvl, 
		      size,
		      img_size,
		      P,
		      boundary):

	points_copy = np.copy(points)
	voxel_timer = Timer()
	voxel_timer.tic()
	voxel = np.zeros((size[0],size[1],size[2]), dtype=np.float32)
	voxel_N = np.int32(voxel.shape[0])
	voxel_stride = np.int32(voxel.strides[0])

	# cprint("voxel stride: {}".format(voxel.strides))

	voxel_bytes = voxel.size * voxel.dtype.itemsize	

	points_N = np.int32(points.shape[0])
	points_stride = np.int32(points.strides[0])
	points_bytes = points.size * points.dtype.itemsize

	q_x = np.float32(q_lvl[0])
	q_y = np.float32(q_lvl[1])
	q_z = np.float32(q_lvl[2])

	voxel_gpu = drv.mem_alloc(voxel_bytes)
	points_gpu = drv.mem_alloc(points_bytes)

	drv.memcpy_htod(voxel_gpu, voxel)
	drv.memcpy_htod(points_gpu, points)

	bdim = (16, 1, 1)
	gdim = (8096,1)

	start = drv.Event()
	end = drv.Event()
	start.record() # start timing

	gpu_voxelize(voxel_gpu, 
				points_gpu, 
				np.int32(voxel.strides[0]), 
				np.int32(voxel.strides[1]), 
				points_stride, 
				points_N,
				np.float32(img_size[0]),
				np.float32(img_size[1]),
				np.float32(q_lvl[0]), 
				np.float32(q_lvl[1]), 
				np.float32(q_lvl[2]),  
				grid=gdim, 
				block=bdim)

	end.synchronize()
	drv.memcpy_dtoh(voxel, voxel_gpu)
	drv.memcpy_dtoh(points, points_gpu)

	return voxel, points


def cpu_voxel(points, 
		      q_lvl, 
		      size,
		      img_size,
		      P,
		      boundary):

	points = clip_by_boundary(points)
	points = clip_by_projection(points, P, img_size[0], img_size[1])
	points[:,1] = points[:,1] - boundary[3]
	points[:,2] = points[:,2] - boundary[5]
	points = np.floor(points/q_lvl)
	points = points.astype(int)

	voxel = np.zeros((size[0],size[1],size[2]), dtype=np.float32)
	for i in points:
		voxel[i[0], i[1], i[2]] = 1.0

	return voxel, points

def main():
	x_min = 0.0
	x_max = 70.4
	y_min = -40.0
	y_max = 40.0
	z_min = -3.5
	z_max = 3.5
	img_w = 1224.0
	img_h = 370.0

	for repeat in range(3000):
		name = '000000'
		calib_dir = '../datasets/kitti/training/calib/'	
		P = load_P(calib_dir, int(name))
		calib_mat = read_calib_mat(calib_dir, int(name))	
		
		q_lvl = [0.25,0.25,0.25]
		img_size = [img_h, img_w]
		x_range = x_max - x_min
		y_range = y_max - y_min
		z_range = z_max - z_min
		voxel_size_x = (int(x_range/q_lvl[0]))
		voxel_size_y = (int(y_range/q_lvl[1]))
		voxel_size_z = (int(z_range/q_lvl[2]))
		voxel_size = [voxel_size_x, voxel_size_y, voxel_size_z]
		boundary = [x_max, x_min, y_max, y_min, z_max, z_min]

		voxel = np.zeros((voxel_size_x, voxel_size_y, voxel_size_z), dtype=np.int32)

		f_lidar = '../datasets/kitti/training/velodyne/{}.bin'.format(name)
		setting_timer = Timer()
		setting_timer.tic()

		points = np.fromfile(f_lidar, dtype=np.float32).reshape(-1, 4)
		points = points[:,0:3]

		points = clip_by_boundary(points)
		points = clip_by_projection(points, P, img_size[0], img_size[1])
		points = np.ascontiguousarray(points, dtype=np.float32)
		voxel_gpu, points_gpu = gpu_voxel(points, 
						  q_lvl, 
						  voxel_size,
						  img_size,
						  P,
						  boundary)

		print ('[timing] gpu voxelizer: %.4f sec' % setting_timer.toc(average=False))

		setting_timer.tic()

		points = np.fromfile(f_lidar, dtype=np.float32).reshape(-1, 4)
		points = points[:,0:3]	

		voxel_cpu, points_cpu = cpu_voxel(points, q_lvl, voxel_size, img_size, P, boundary)

		print ('[timing] cpu voxelizer: %.4f sec' % setting_timer.toc(average=False))

if __name__ == '__main__':
	main()

