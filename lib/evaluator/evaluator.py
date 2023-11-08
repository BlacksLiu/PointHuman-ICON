import trimesh
from PIL import Image
import torch 
import os 

# pytorch3d pipeline:render normal from ground truth mesh
from lib.common.render import Render, cleanShader
from lib.evaluator.evaluator_util import (
    NormalRender,
    projection,
    load_calib,
    get_chamfer_distance,
    get_p2s_distance,
    get_pcls_normal_map,

) 

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


def get_pifuhd_calib(calib_path):
    # loading calibration data
    param = np.load(calib_path, allow_pickle=True)
    # pixel unit / world unit is equal to 1
    # pixel unit / uv unit ---> is ortho_ratio
    ortho_ratio = param.item().get('ortho_ratio')
    # world unit / model unit
    scale = param.item().get('scale')

    # camera center world coordinate
    center = param.item().get('center')
    # model rotation
    R = param.item().get('R')
    #translate the position of camera into world coordinate origin. 
    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    # Match camera space to image pixel space
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    #render code, this part flip(axis =0),therefore, y need change
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio


    #uv space is [-1,1] we map [-256,255]->[-1,1]
    # Match image pixel space to image uv space
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(512 // 2)
    uv_intrinsic[1, 1] = 1.0 / float(512 // 2)
    uv_intrinsic[2, 2] = 1.0 / float(512 // 2)

    # Transform under image pixel space
    trans_intrinsic = np.identity(4)

    intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
    calib = np.matmul(intrinsic, extrinsic)
    
    calib[:3,:3] /= scale
    return calib, scale


# ours (radius=0.005) others render from mesh

dataset = 'thuman2'
# cape 
if dataset == 'cape':
    mesh_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/cape/scans"
    subject_names = sorted([i.split('.')[0] for i in os.listdir(mesh_dir)])
    views = [
        '000', '120', '240'
    ]
    calib_dir =  "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/cape_3views/"
    output_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
    pred_mesh_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
    view2obj = {
        '000': 'est_mesh_0.obj',
        '120': 'est_mesh_120.obj',
        '240': 'est_mesh_240.obj',
    }
elif dataset == 'thuman2':
    dat_root = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/thuman2"
    mesh_dir = os.path.join(dat_root, 'scans')
    subject_names = sorted([subject.split('/')[-1] for subject in np.loadtxt(os.path.join(dat_root, 'test.txt'), dtype=str).tolist()])
    # subject_names = sorted([i.split('.')[0] for i in os.listdir(mesh_dir)])
    views = [
        f'{i:03d}' for i in range(0,360,10)
    ]
    calib_dir =  "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/thuman2_36views/"
    output_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
    pred_mesh_dir = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results"
    view2obj={}
    for view in views:
        view2obj[view] = f'est_mesh_{int(view)}.obj'

compare_methods = [
    # 'pifu',
    # 'open-pifuhd',
    # 'pamir',
    # 'icon-nofilter',
    # 'icon-filter',
    'econ',
    # 'pointhuman_1_outputs',
]
print(f"running method:{compare_methods[0]}, dataset: {dataset}")

# mesh normal rendering
# scale = 100.0
# camera = get_camera(scale, device)
# icon_renderer = init_renderer(camera, device)
# render = Render(size=512, device=device)

radius = 0.005
calc_mesh_dist = True
calc_NC = True
output_normal_map = True

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

# point cloud renderer 
pcls_renderer = init_point_renderer(radius, device)

normalRender = NormalRender()

score = []
from tqdm import tqdm
for subject_name in tqdm(subject_names):
    mesh_path = os.path.join(mesh_dir, subject_name, f"{subject_name}.obj")
    # mesh = trimesh.load(mesh_path)
    for view in tqdm(views):
        calib_path = os.path.join(calib_dir, subject_name, "calib", f'{view}.txt')
        calib = load_calib(calib_path)
        if calc_NC:
            # to ndc 
            # gt_mesh = project_mesh(render, mesh, calib, scale)
            # gt_normal_imgs = get_normal_img(icon_renderer, gt_mesh)

            gt_pcl, _, _ = normalRender.get_proj_pcls(mesh_path, calib)
            gt_normal_imgs = get_pcls_normal_map(pcls_renderer, gt_pcl)

        # for chamfer and p2s
        if calc_mesh_dist:
            gt_pcl, gt_vertices, gt_faces = normalRender.get_proj_pcls(mesh_path, calib)

        tmp_dict = {}
        tmp_dict['subject_name'] = subject_name
        tmp_dict['view'] = view
        for method in compare_methods:
            if 'pointhuman' not in method:
                pred_mesh_path = os.path.join(pred_mesh_dir, method, f'{dataset}/{subject_name}/{view2obj[view]}')
                if 'econ' == method:
                    pred_mesh_path = os.path.join(pred_mesh_dir, method, f'{dataset}/{subject_name}/{dataset}-{subject_name}-{int(view):03d}_final.obj')
                
                if 'open-pifuhd' == method:
                    pred_mesh_path = os.path.join(pred_mesh_dir, method, f'{dataset}/{subject_name}/est_mesh_{view}.obj')


                if calc_NC:
                    # render normal from pointclouds
                    pred_pcl, _, _ = normalRender.get_proj_pcls(pred_mesh_path)
                    pred_normal_imgs = get_pcls_normal_map(pcls_renderer, pred_pcl)
                    normal_img_output = os.path.join(pred_mesh_dir, method, f'normal_map/from_pointcloud/{dataset}/{view}')

                    # render normal from mesh 
                    # pred_mesh = trimesh.load(pred_mesh_path)
                    # pred_mesh = project_mesh(render, pred_mesh, calib=None)
                    # pred_normal_imgs = get_normal_img(icon_renderer, pred_mesh)
                    # normal_img_output = os.path.join(pred_mesh_dir, method, f'normal_map_from_mesh/{view}')
                
                if calc_mesh_dist:
                    pred_pcl, _, _ = normalRender.get_proj_pcls(pred_mesh_path)
                    # trimesh.Trimesh(gt_pcl).export(f'{method}_pred_proj.obj')
            else:
                pred_mesh_path = os.path.join(pred_mesh_dir, method, f'{dataset}/{subject_name}/{view}/est_scan.obj')
                if calc_NC:
                    pred_pcl, _, _ = normalRender.get_ori_pcls(pred_mesh_path, calib_path, z_norm=False)
                    pred_normal_imgs = get_pcls_normal_map(pcls_renderer, pred_pcl)
                    normal_img_output = os.path.join(pred_mesh_dir, method, f'normal_map/from_pointcloud/{dataset}/{view}')
            
            if calc_NC:
                error = (((pred_normal_imgs - gt_normal_imgs)**2).sum(dim=2).mean())
                tmp_dict[f'{method}_nc']=error.cpu().numpy()
            
            if output_normal_map:
                if not os.path.exists(normal_img_output):
                    os.makedirs(normal_img_output)
                Image.fromarray(
                    (
                        np.concatenate(
                            (gt_normal_imgs.cpu().numpy(), pred_normal_imgs.cpu().numpy()
                        ), axis=1) * 255.0
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
    
output_root = f"/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/results/{compare_methods[0]}"
os.makedirs(output_root, exist_ok=True)
npy_path = os.path.join(output_root, f'MESH_{dataset}_{compare_methods[0]}.npy')
np.save(npy_path, score, allow_pickle=True)
import pandas as pd 
df = pd.DataFrame(score)
tmp_dict = {}

if dataset == "cape":
    split_path = "/mnt/local4T/pengfei/projects/PointHuman/PointHuman-ICON/data/cape/test150.txt"
    data = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')       #去除文本中的换行符
            ann = ann.split('/')[1]
            data.append(ann)
    simple_subject_name = data[:50]
    hard_subject_name = data[50:]

    easy_df = df.loc[df['subject_name'].isin(simple_subject_name)]
    hard_df = df.loc[df['subject_name'].isin(hard_subject_name)]

    print(f"easy objs num:{len(easy_df)}, hard objs num:{len(hard_df)}")


    calc_mesh_dist = True
    for method in compare_methods:
        if calc_mesh_dist and calc_NC:
            tmp_dict[method] = {
                'nc_easy': easy_df[f'{method}_nc'].mean(),
                'nc_hard': hard_df[f'{method}_nc'].mean(),
                'chamfer_easy': easy_df[f'{method}_chamfer'].mean(),
                'chamfer_hard': hard_df[f'{method}_chamfer'].mean(),
                'p2s_easy': easy_df[f'{method}_p2s'].mean(),
                'p2s_hard': hard_df[f'{method}_p2s'].mean(),
            }
        elif calc_NC:
            tmp_dict[method] = {
                'nc_easy': easy_df[f'{method}_nc'].mean(),
                'nc_hard': hard_df[f'{method}_nc'].mean(),
            }
        else: 
            tmp_dict[method] = {
                'chamfer_easy': easy_df[f'{method}_chamfer'].mean(),
                'chamfer_hard': hard_df[f'{method}_chamfer'].mean(),
                'p2s_easy': easy_df[f'{method}_p2s'].mean(),
                'p2s_hard': hard_df[f'{method}_p2s'].mean(),
            }
elif dataset == "thuman2":
    for method in compare_methods:
        if calc_mesh_dist and calc_NC:
            tmp_dict[method] = {
                'nc': df[f'{method}_nc'].mean(),
                'chamfer': df[f'{method}_chamfer'].mean(),
                'p2s': df[f'{method}_p2s'].mean(),
            }
        elif calc_NC:
            tmp_dict[method] = {
                'nc': df[f'{method}_nc'].mean(),
            }
        else: 
            tmp_dict[method] = {
                'chamfer': df[f'{method}_chamfer'].mean(),
                'p2s': df[f'{method}_p2s'].mean(),
            }

import json
with open(f"{os.path.join(output_root, f'MESH_{dataset}_{compare_methods[0]}.json')}", 'w') as fp:
    fp.write(json.dumps(tmp_dict, indent=4))
    