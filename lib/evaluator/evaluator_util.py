import os 
from lib.renderer.mesh import load_scan
import numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
import lib.renderer.opengl_util as opengl_util
from lib.renderer.gl.init_gl import initialize_GL_context
from lib.renderer.gl.color_render import ColorRender
from lib.renderer.camera import Camera

import torch 
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.structures import Pointclouds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NormalRender():
    def __init__(self, size=512, egl=True):
        self.egl = True
        self.size = size
        initialize_GL_context(width=size, height=size, egl=egl)
        self.cam = Camera(width=size, height=size)
        self.rndr = ColorRender(width=size, height=size, egl=egl)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def init_cam(self):
        ortho_ratio = 1 * (512 / self.size)
        self.cam.ortho_ratio = ortho_ratio
        self.cam.near = -1
        self.cam.far = 1
        self.cam.sanity_check()
        self.cam.width = 2
        self.cam.height = 2

    def get_normal_depth_map(self, vertices, faces, normals, faces_normals):

        dic = {
            'ortho_ratio': 1,
            'scale': 1,
            'center': np.zeros(3),
            'R' : np.eye(3)
        }
        calib = opengl_util.load_calib(dic, render_size=self.size)
        extrinsic = calib[:4, :4]
        intrinsic = calib[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        self.init_cam()
        self.cam.set_projection_matrix(calib_mat[:3,:4])

        self.rndr.set_mesh(
            vertices, faces, normals, faces_normals
        )

        self.rndr.set_norm_mat(1, np.zeros(3))

        self.rndr.set_camera(self.cam)
        self.rndr.display()
        normal = self.normal_render_result()
        depth = self.depth_render_result()
        self.rndr.cleanup()

        return normal, depth
    
    def depth_render_result(self):
        cam_render = self.rndr.get_color(2)
        cam_render[:, :, -1] -= 0.5
        cam_render[:, :, -1] *= 2.0
        return cam_render

    def normal_render_result(self):
        cam_render = self.rndr.get_color(0)
        return cam_render

    def get_proj_pcls(self, mesh_file, calib=None, z_norm=False, scale = 100.0):
        vertices, faces, norms, face_normals, _, _  = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )
        if calib is not None:
            vertices *= scale
            vertices = projection(vertices, calib)
            vertices[:, 1] *= -1
        if z_norm:
            vertices[:, 2] -= vertices[:, 2].mean()

        normal, depth = self.get_normal_depth_map(vertices, faces, norms, face_normals)
        proj_pcl = invert_projection(depth)

        return proj_pcl, vertices, faces

    def render_normal_map_from_mesh(self, mesh_file, calib_path=None):
        vertices, faces, norms, face_normals, _, _  = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )
        if calib_path is not None:
            scale = 100.0
            vertices *= scale
            calib = load_calib(calib_path)
            vertices = projection(vertices, calib)
            vertices[:, 1] *= -1
        normal, depth = self.get_normal_depth_map(vertices, faces, norms, face_normals)

        mask = normal[:,:,3]
        normal = normal[:,:,:3]
        normal = (normal + 1) / 2
        tmp = np.zeros_like(normal)
        tmp[mask==1] = normal[mask==1]
        normal = tmp
        return normal, mask

    
    def get_ori_pcls(self, mesh_file, calib_path=None, z_norm=True):
        import trimesh
        mesh = trimesh.load(mesh_file)
        vertices = mesh.vertices 
        if calib_path is not None:
            scale = 100.0
            vertices *= scale
            calib = load_calib(calib_path)
            vertices = np.matmul(calib[:3, :3], vertices.T).T + calib[:3, 3]
            vertices[:, 1] *= -1
        if z_norm:
            vertices[:, 2] -= vertices[:, 2].mean()
        return vertices, vertices, None 

def invert_projection(depth):
    mask = depth[:, :, -1] == 1
    z_ndc = depth[:, :, 0]

    z_ndc = z_ndc[mask]
    x = np.linspace(-1, 1, 512)
    y = np.linspace(-1, 1, 512)

    xv, yv = np.meshgrid(x,y)
    x_ndc = xv[mask]
    y_ndc = yv[mask]

    y_ndc = -y_ndc
    z_ndc = -z_ndc
    
    pcls = np.stack([x_ndc, y_ndc, z_ndc], axis= -1)
    pcls_view = pcls
    return pcls_view

def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    return calib_mat

def projection(vertices, calib):
    vertices = np.matmul(calib[:3, :3], vertices.T).T + calib[:3, 3]
    return vertices

def get_chamfer_distance(gt_pcls, pred_pcls):
    gt_pcls = torch.from_numpy(gt_pcls).float()
    pred_pcls = torch.from_numpy(pred_pcls).float()
    tgt_points = Pointclouds(torch.unsqueeze(gt_pcls,0)).to(device)
    pred_points = Pointclouds(torch.unsqueeze(pred_pcls, 0)).to(device)
    chamfer_dist = chamfer_distance(tgt_points, pred_points)[0] * 100

    return chamfer_dist.cpu()




from typing import Union
from pytorch3d.loss.chamfer import (
    _handle_pointcloud_input,
    _validate_chamfer_reduction_inputs,
    knn_points
    )

def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    cham_x = torch.sqrt(cham_x)
    cham_y = torch.sqrt(cham_y)
    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)

    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        y_lengths_clamped = y_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_y /= y_lengths_clamped
        if return_normals:
            cham_norm_x /= x_lengths_clamped
            cham_norm_y /= y_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals

from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
def point_mesh_distance(meshes, pcls):
    # source code of pytorch3D.loss.point_mesh_face_distance
    
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()    # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed() # (sum(V_n), 3)
    faces_packed = meshes.faces_packed() # (sum(F_n), 3)
    tris = verts_packed[faces_packed]    # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()    # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()    # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = torch.sqrt(point_to_face) * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist

def get_p2s_distance(vertices, faces, pcls):
    from lib.common.render import Render
    render = Render(size=512, device=device)
    mesh = render.VF2Mesh(vertices, faces)
    pcls = torch.from_numpy(pcls).float()
    # for single pcl
    pcl = Pointclouds(torch.unsqueeze(pcls, 0))
    pcl = pcl.to(device)
    p2s_dist = point_mesh_distance(mesh, pcl) * 100
    return p2s_dist

from pytorch3d.structures import Pointclouds

def estimate_normals(xyzs):
    """Estimate point normals. The pointcloud should be a sphere-like shape.

    Args:
        xyzs (tensor): Nx3 or BxNx3.
    """
    if xyzs.ndim == 2:
        xyzs = xyzs[None, ...]

    normals = estimate_pointcloud_normals(
        xyzs * 1000.,
        neighborhood_size=6,
        disambiguate_directions=False,
        use_symeig_workaround=True
    )
    normals = normals / (
        torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-8
    )
    return normals

def test_normals(normals):
    no_batch_dim = True
    outside_direcion = torch.tensor([0,0,1]).to(device=normals.device)
    
    normals = torch.sign(
        torch.sum(normals * outside_direcion, dim=-1, keepdim=True)) * normals
    
    normals = (normals + 1.0) / 2 

    if no_batch_dim:
        normals = normals[0]

    return normals

def get_pcls_normal_map(renderer, pcl):
    pcl_tensor = torch.from_numpy(pcl).float().to(device)
    normals = estimate_normals(pcl_tensor)
    normals = test_normals(normals)
    verts = pcl_tensor.to(device)
    rgb = normals.to(device)
    gt_pcls = Pointclouds(points=[verts], features=[rgb])
    images = renderer(gt_pcls)[0]
    return images

# from pytorch3d.ops import sample_points_from_meshes

# Data structures and functions for rendering
# from pytorch3d.structures import Pointclouds
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     OrthographicCameras,
#     PointsRasterizationSettings,
#     PointsRenderer,
#     PointsRasterizer,
#     AlphaCompositor,
# )
# # Setup
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")


# R, T = look_at_view_transform(20, 0, 0)
# cameras = OrthographicCameras(device=device, R=R, T=T)
# raster_settings = PointsRasterizationSettings(
#     image_size=512, 
#     radius = 0.003,
#     points_per_pixel = 10
# )

# rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
# renderer = PointsRenderer(
#     rasterizer=rasterizer,
#     compositor=AlphaCompositor(background_color=(0, 0, 0))
# )


# def get_normal_map(vertices, faces, normals, faces_normals, size=512):
#     dic = {
#         'ortho_ratio': 1,
#         'scale': 1,
#         'center': np.zeros(3),
#         'R' : np.eye(3)
#     }
#     calib = opengl_util.load_calib(dic, render_size=size)
#     extrinsic = calib[:4, :4]
#     intrinsic = calib[4:8, :4]
#     calib_mat = np.matmul(intrinsic, extrinsic)
#     cam.width = 2
#     cam.height = 2
#     cam.set_projection_matrix(calib_mat[:3,:4])

#     rndr.set_mesh(
#         vertices, faces, normals, faces_normals
#     )

#     rndr.set_norm_mat(1, np.zeros(3))

#     rndr.set_camera(cam)
#     rndr.display()
#     normal = nomal_render_result(rndr)
#     return normal

# def get_depth_map(vertices, faces, normals, faces_normals, output_path=None, size=512):
#     initialize_GL_context(width=size, height=size, egl=egl)

#     cam = Camera(width=size, height=size)
#     ortho_ratio = 1 * (512 / size)
#     cam.ortho_ratio = ortho_ratio
#     cam.near = -1
#     cam.far = 1
#     cam.sanity_check()
#     cam.width = 2
#     cam.height = 2

#     dic = {
#         'ortho_ratio': 1,
#         'scale': 1,
#         'center': np.zeros(3),
#         'R' : np.eye(3)
#     }
#     calib = opengl_util.load_calib(dic, render_size=size)
#     extrinsic = calib[:4, :4]
#     intrinsic = calib[4:8, :4]
#     calib_mat = np.matmul(intrinsic, extrinsic)
#     cam.width = 2
#     cam.height = 2
#     cam.set_projection_matrix(calib_mat[:3,:4])

#     rndr = ColorRender(width=size, height=size, egl=egl)
#     rndr.set_mesh(
#         vertices, faces, normals, faces_normals
#     )

#     rndr.set_norm_mat(1, np.zeros(3))

#     rndr.set_camera(cam)
#     rndr.display()
#     depth = depth_render_result(rndr, output_path)

#     rndr.cleanup()
#     return depth