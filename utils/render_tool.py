from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
# os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import torch
from visualize.simplify_loc2rot import joints2smpl
import pyrender
import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
import math
# import ffmpeg
from PIL import Image
from tqdm import tqdm

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def xrot_trans_to_pose(angle, translation):
    # [R ; T]
    # [0 ; 1]
    cos_angle, sin_angle = np.cos(angle), np.sin(angle) 
    return np.array([[1, 0, 0, translation[0]],
                     [0, cos_angle, -sin_angle, translation[1]],
                     [0, sin_angle, cos_angle, translation[2]],
                     [0, 0, 0, 1]])

def render_smpl_mesh(motions, poses, with_floor=False, **kwargs):
    bg_color = [1, 1, 1, 0.8]
    ambient_light = (0.4, 0.4, 0.4)
    smpl_mesh_base_color = (0.11, 0.53, 0.8, 0.5)

    frames, njoints, nfeats = motions.shape
    min_fj_coord = motions.min(axis=0).min(axis=0)
    max_fj_coord = motions.max(axis=0).max(axis=0)

    height_offset = min_fj_coord[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    print(f'Running SMPLify, it may take a few minutes.')
    motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

    vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints

    min_fj_coord = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    max_fj_coord = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= min_fj_coord[1] + 1e-5

    
    minx = min_fj_coord[0] - 0.5
    maxx = max_fj_coord[0] + 0.5
    minz = min_fj_coord[2] - 0.5 
    maxz = max_fj_coord[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    floor_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
    floor_mesh.visual.face_colors = [0, 0, 0, 0.21]
    floor_pyrender = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

    video_frames = []

    for i in tqdm(range(frames)):
        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=smpl_mesh_base_color
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
        
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        if with_floor:
            scene.add(floor_pyrender, pose=xrot_trans_to_pose(np.pi * 0.5, np.array([0, min_fj_coord[1].cpu().numpy(), 0])))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())

        scene.add(camera, pose=xrot_trans_to_pose(-np.pi / 6, np.array([(minx+maxx).cpu().numpy()/2, 
                                                                       1.5, 
                                                                       max(4, minz.cpu().numpy()+(1.5-min_fj_coord[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())])))
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        video_frames.append(color)

        r.delete()

    out = np.stack(video_frames, axis=0)
    return out

def render_smpl_mesh_still(motions, poses, with_floor=False, **kwargs):
    bg_color = [1, 1, 1, 0.8]
    ambient_light = (0.4, 0.4, 0.4)
    smpl_mesh_base_color = (0.11, 0.53, 0.8, 0.5)

    frames, njoints, nfeats = motions.shape
    min_fj_coord = motions.min(axis=0).min(axis=0)
    max_fj_coord = motions.max(axis=0).max(axis=0)

    height_offset = min_fj_coord[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    print(f'Running SMPLify, it may take a few minutes.')
    # motion_tensor, opt_dict = j2s.joint2smpl(motions[..., :5])  # [batch, njoints, 6, nframes]
    # np.save("test_tensor.npy", motion_tensor.cpu().numpy())
    motion_tensor = torch.from_numpy(np.load("test_tensor.npy")).cuda()
    print(motion_tensor[0], motion_tensor.shape)
    
    altered_motion_tensor = torch.zeros_like(motion_tensor)
    # ~9, 
    altered_motion_tensor[0, :, :] = motion_tensor[0, :, :, 0:1]
    altered_motion_tensor[0, 18, :] = 0
    altered_motion_tensor[0, 18, 0] = 1
    altered_motion_tensor[0, 18, 4] = 1
    altered_motion_tensor[0, 19, :] = 0
    altered_motion_tensor[0, 19, 0] = 1
    altered_motion_tensor[0, 19, 4] = 1
    altered_motion_tensor[0, 17, :] = 0
    altered_motion_tensor[0, 17, 0] = 1
    altered_motion_tensor[0, 17, 4] = 1
    altered_motion_tensor[0, 16, :] = 0
    altered_motion_tensor[0, 16, 0] = 1
    altered_motion_tensor[0, 16, 4] = 1
    altered_motion_tensor[0, 15, :] = 0
    altered_motion_tensor[0, 15, 0] = 1
    altered_motion_tensor[0, 15, 4] = 1
    altered_motion_tensor[0, 12, :] = 0
    altered_motion_tensor[0, 12, 0] = 1
    altered_motion_tensor[0, 12, 4] = 1
    altered_motion_tensor[0, 13, :] = 0
    altered_motion_tensor[0, 13, 0] = 1
    altered_motion_tensor[0, 13, 4] = 1
    altered_motion_tensor[0, 14, :] = 0
    altered_motion_tensor[0, 14, 0] = 1
    altered_motion_tensor[0, 14, 4] = 1
    # altered_motion_tensor[0, 0, :] = motion_tensor[0, 0, :, 0:1]
    # altered_motion_tensor[0, 1, :] = motion_tensor[0, 1, :, 0:1]
    # altered_motion_tensor[0, 2, :] = motion_tensor[0, 2, :, 0:1]
    # altered_motion_tensor[0, 3, :] = motion_tensor[0, 3, :, 0:1]
    # altered_motion_tensor[0, 5, :] = 1

    vertices = rot2xyz(altered_motion_tensor, mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices',
                                    vertstrans=True)

    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints

    min_fj_coord = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    max_fj_coord = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= min_fj_coord[1] + 1e-5

    
    minx = min_fj_coord[0] - 0.1
    maxx = max_fj_coord[0] + 0.1
    minz = min_fj_coord[2] - 0.1 
    maxz = max_fj_coord[2] + 0.1
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    floor_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)
    floor_mesh.visual.face_colors = [0, 0, 0, 0.21]
    floor_pyrender = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

    video_frames = []

    for i in tqdm(range(frames)):
        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=smpl_mesh_base_color
        )
        mesh.export("smpl_mesh.stl")
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
        
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        if with_floor:
            scene.add(floor_pyrender, pose=xrot_trans_to_pose(np.pi * 0.5, np.array([0, min_fj_coord[1].cpu().numpy(), 0])))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())

        scene.add(camera, pose=xrot_trans_to_pose(0, np.array([(minx+maxx).cpu().numpy()/2, 
                                                                       0.5, 
                                                                       max(4, minz.cpu().numpy()+(1.5-min_fj_coord[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())])))
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        video_frames.append(color)

        r.delete()

    out = np.stack(video_frames, axis=0)
    return out

body_part_dict = {'full-body': [0, 1, 2, 3, 4], 'upper-body': [2, 3, 4], 'lower-body': [0, 1]}
def render_skeleton(motions, poses, body_part, with_floor=False, with_trajectory=False):
    import matplotlib
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import mpl_toolkits.mplot3d.axes3d as p3
    matplotlib.use('Agg')
    
    assert body_part in body_part_dict.keys()
    
    data = motions.copy().reshape(len(motions), -1, 3)
    
    nb_joints = motions.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    # leftfoot, rightfoot, torso, left hand, right hand, ...
    colors = ['red', 'blue', 'black', 'darkorange', 'deepskyblue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        ax = p3.Axes3D(fig)
        
        init()
        
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        if with_floor:
            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                         MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if with_trajectory and index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        foot_contact = [0, 0, 0, 0]
        foot_contact_joint = [19, 20, 14, 15] if nb_joints == 21 else [7, 10, 8, 11]
        if poses is not None:
            foot_contact = poses[index, 259:] # left foot, left heel, right foot, right heel 
        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            # setup joint marker style
            ax.set_prop_cycle(marker=['o'])
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            if i in body_part_dict[body_part]:
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                        color=color)

        for i, joint in enumerate(foot_contact_joint):
            if foot_contact[i] > 0:
                ax.set_prop_cycle(marker=['*'])
                ax.plot3D(data[index, joint:joint+1, 0], data[index, joint:joint+1, 1], data[index, joint:joint+1, 2], linewidth=linewidth,
                        color="white")                        
        #         print(trajec[:index, 0].shape)
        # Set to default marker style
        ax.set_prop_cycle(marker=['none'])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=96)
        io_buf.seek(0)
        # print(fig.bbox.bounds)
        arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()
        plt.close()
        return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)
