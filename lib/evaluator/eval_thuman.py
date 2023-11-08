import trimesh
from PIL import Image
import torch 
import os 

# pytorch3d pipeline:render normal from ground truth mesh
from lib.common.render import Render, cleanShader
from lib.evaluator.evaluator_util import *

import os 
import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    BlendParams,
    MeshRenderer,
    look_at_view_transform,
    OrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_camera(scale, device):
    R, T = look_at_view_transform(20, 0, 0)

    camera = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=100.0,
        zfar=-100.0,
        max_y=100.0,
        min_y=-100.0,
        max_x=100.0,
        min_x=-100.0,
        scale_xyz=(scale * np.ones(3), ),
    )

    return camera

def init_renderer(cam, device):
    raster_settings_mesh = RasterizationSettings(
                image_size=512,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
            )
    meshRas = MeshRasterizer(cameras=cam, raster_settings=raster_settings_mesh)
    blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
    renderer = MeshRenderer(
            rasterizer=meshRas,
            shader=cleanShader(device=device, cameras=cam, blend_params=blendparam),
        )
    return renderer

def project_mesh(render, mesh, calib=None, scale=100.0):
    if calib is not None:
        verts_gt = torch.as_tensor(mesh.vertices * scale).float()
        proj_verts = projection(verts_gt, calib)
        proj_verts[:, 1] *= -1
    else:
        proj_verts = torch.as_tensor(mesh.vertices).float()
    faces_gt = torch.as_tensor(mesh.faces).long()

    proj_mesh = render.VF2Mesh(proj_verts, faces_gt)
    return proj_mesh

def get_normal_img(renderer, mesh):
    rendered_img = (
        renderer(mesh[0])[0:1, :, :, :3] - 0.5
    ) * 2.0
    rendered_img = ((rendered_img + 1.0) * 0.5)[0]
    return rendered_img
# THUMAN2.0 TESTSET 
# ours (radius=0.005) others render from mesh  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mesh_dir="/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/thuman2/scans"
subject_names = ['0500', '0501', '0502', '0503', '0504']
calib_dir="/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/thuman2/thuman2_perspective_36views"
output_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
pred_mesh_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
compare_methods = [
    'icon-filter',
    'icon-nofilter',
    'pamir',
    'pifu',
    # 'pointhuman_1_outputs'
]

views = ["%03d" %(i) for i in range(0,360,10)]
obj_idxs = [i for i in range(0, 360, 10)]
view2obj={}
for view, obj_idx in zip(views, obj_idxs):
    view2obj[view] = f'est_mesh_{obj_idx}.obj'

scale = 100.0
camera = get_camera(scale, device)
icon_renderer = init_renderer(camera, device)
render = Render(size=512, device=device)

radius = 0.005
calc_mesh_dist = True
output_normal_map = False

def init_point_renderer(radius, device):
    R, T = look_at_view_transform(20, 0, 0)
    cameras = OrthographicCameras(device=device, R=R, T=T)
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = radius,
        points_per_pixel = 10
    )

    point_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    point_renderer = PointsRenderer(
        rasterizer=point_rasterizer,
        compositor=AlphaCompositor(background_color=(0, 0, 0))
    )

    return point_renderer

pcls_renderer = init_point_renderer(radius, device)


score = []
from tqdm import tqdm
for subject_name in tqdm(subject_names):
    mesh_path = os.path.join(mesh_dir, subject_name, f"{subject_name}.obj")
    mesh = trimesh.load(mesh_path)
    for view in views:
        calib_path = os.path.join(calib_dir, subject_name, "calib", f'{view}.txt')
        calib = load_calib(calib_path)
        gt_mesh = project_mesh(render, mesh, calib, scale)
        gt_normal_imgs = get_normal_img(icon_renderer, gt_mesh)

        # for chamfer and p2s
        if calc_mesh_dist:
            gt_pcl, gt_vertices, gt_faces = get_proj_pcls(mesh_path, calib_path)

        normal_imgs = [gt_normal_imgs]
        tmp_dict = {}
        tmp_dict['subject_name'] = subject_name
        tmp_dict['view'] = view
        for method in compare_methods:
            if 'pointhuman' not in method:
                # pred_mesh_path = os.path.join(pred_mesh_dir, method, f'cape/{subject_name}/{view2obj[view]}')
                # pred_pcl, _, _ = get_proj_pcls(pred_mesh_path)
                # pred_normal_imgs = get_pcls_normal_map(pcls_renderer, pred_pcl)

                pred_mesh_path = os.path.join(pred_mesh_dir, method, f'thuman2/{subject_name}/{view2obj[view]}')
                pred_mesh = trimesh.load(pred_mesh_path)
                pred_mesh = project_mesh(render, pred_mesh, calib=None)
                pred_normal_imgs = get_normal_img(icon_renderer, pred_mesh)
                normal_img_output = os.path.join(pred_mesh_dir, method, f'normal_map_from_mesh/{view}')
                
                if calc_mesh_dist:
                    pred_pcl, _, _ = get_proj_pcls(pred_mesh_path)

            else:
                pred_mesh_path = os.path.join(pred_mesh_dir, method, f'thuman2/{subject_name}/{view}/est_scan.obj')
                pred_pcl, _, _ = get_ori_pcls(pred_mesh_path, calib_path, z_norm=False)
                pred_normal_imgs = get_pcls_normal_map(pcls_renderer, pred_pcl)
                normal_img_output = os.path.join(pred_mesh_dir, method, f'normal_map_{radius}/{view}')

            error = (((pred_normal_imgs - gt_normal_imgs)**2).sum(dim=2).mean())
            tmp_dict[f'{method}_nc']=error.cpu().numpy()

            if not os.path.exists(normal_img_output):
                os.makedirs(normal_img_output)
            
            if output_normal_map:
                Image.fromarray(
                    (
                        pred_normal_imgs.cpu().numpy() * 255.0
                    ).astype(np.uint8)
                ).save(os.path.join(normal_img_output, f'{subject_name}.jpg'))
            if calc_mesh_dist :
                chamfer_dist = get_chamfer_distance(gt_pcl, pred_pcl).cpu().numpy()
                p2s_dist = get_p2s_distance(gt_vertices, gt_faces, pred_pcl).cpu().numpy()
                tmp_dict[f'{method}_chamfer'] = chamfer_dist
                tmp_dict[f'{method}_p2s'] = p2s_dist
        score.append(tmp_dict)
        # break
    # break

output_root = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"

npy_path = os.path.join(output_root, f'THUMAN_NC_{radius}_mesh.npy')
np.save(npy_path, score, allow_pickle=True)

