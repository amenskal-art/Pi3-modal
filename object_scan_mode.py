# ==========================================
# Pi3X -> TSDF FUSION -> CLEAN WATERTIGHT MESH
# ==========================================
# WHY THIS EXISTS:
# Concatenating per-view point clouds can never fully remove double walls,
# because every view's predicted depth has its own non-rigid bias. Rigid ICP
# can't close a gap that varies across the surface, and MLS smoothing gets
# confused when its neighborhood spans both walls.
#
# TSDF fusion integrates each view's DEPTH MAP into a signed-distance volume.
# Depth biases average out along each camera ray -> exactly ONE surface
# crossing -> marching cubes extracts a single clean watertight mesh.
# High-frequency depth noise is averaged away in the same step.
#
#   pip install open3d

import torch
import numpy as np
import os
import open3d as o3d

from pi3.utils.basic import load_multimodal_data
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X

# ==========================================
# USER SETTINGS
# ==========================================
data_path = './input_data'       # .mp4 file OR directory of images
output_dir = './Pi3_Outputs'

CONF_THRESH   = 0.4    # was 0.1 in old script. 0.1 keeps garbage that becomes
                       # floaters/ghosts. Tune 0.3 - 0.6. Higher = cleaner but
                       # may open holes in weak-texture areas.
VOXEL_DIVISOR = 350    # TSDF resolution = scene_diagonal / VOXEL_DIVISOR.
                       # Bigger = finer detail (and more RAM). 250-500 typical.
SDF_TRUNC_MUL = 4.0    # truncation band = voxel * this. 3-5 is standard.
TAUBIN_ITERS  = 15     # volume-preserving post-smooth. 0 to disable.
MIN_CLUSTER_FRAC = 0.05  # drop disconnected mesh islands smaller than 5% of
                         # the biggest one (kills leftover floaters).

RUN_POISSON_TOO = True   # ALSO produce a Poisson mesh from the TSDF cloud
POISSON_DEPTH   = 9      # 9-10. TSDF cloud has clean oriented normals, so
                         # Poisson actually behaves here.

# ==========================================
# OUTPUT / INPUT VALIDATION
# ==========================================
os.makedirs(output_dir, exist_ok=True)
mesh_path    = os.path.join(output_dir, 'tsdf_mesh.ply')
cloud_path   = os.path.join(output_dir, 'tsdf_cloud_poisson_ready.ply')
poisson_path = os.path.join(output_dir, 'poisson_mesh.ply')

if not os.path.exists(data_path):
    raise ValueError(f"Input path not found: {data_path}. Execution stopped.")

print(f"\n[Success] Data found at: {data_path}")

# ==========================================
# INFERENCE (same as before)
# ==========================================
interval = 10 if data_path.endswith('.mp4') else 1
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print(f'\nUsing device: {device_name} | sampling interval: {interval}')

conditions = dict(intrinsics=None, poses=None, depths=None)
imgs, conditions = load_multimodal_data(data_path, conditions,
                                        interval=interval, device=device)
use_multimodal = any(v is not None for v in conditions.values())

print("Loading Pi3X model...")
model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
if not use_multimodal:
    model.disable_multimodal()
model = model.to(device)

print("Running model inference...")
with torch.no_grad():
    if device_name == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs=imgs, **conditions)
    else:
        res = model(imgs=imgs, **conditions)

# ==========================================
# MASKS: confidence + depth-edge rejection
# ==========================================
conf = torch.sigmoid(res['conf'][..., 0])                       # (1,V,H,W)
masks = conf > CONF_THRESH
non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)  # (1,V,H,W)
masks = torch.logical_and(masks, non_edge)[0].cpu().numpy()     # (V,H,W)

local_pts = res['local_points'][0].float().cpu().numpy()        # (V,H,W,3) cam frame
world_pts = res['points'][0].float().cpu().numpy()              # (V,H,W,3) world frame

if 'camera_poses' not in res:
    raise KeyError("res has no 'camera_poses' key -- print(res.keys()) and "
                   "point me at the cam2world pose tensor, bro.")
poses = res['camera_poses'][0].float().cpu().numpy()            # (V,4,4) cam2world

num_views, H, W = local_pts.shape[:3]
print(f"\n{num_views} views @ {W}x{H}")

# ==========================================
# INTRINSICS: least-squares fit from the model's own camera-frame points
# (u = fx * X/Z + cx  ,  v = fy * Y/Z + cy)
# No dependency on any specific rays_d output key.
# ==========================================
def fit_intrinsics(local_v, mask_v):
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    X, Y, Z = local_v[..., 0], local_v[..., 1], local_v[..., 2]
    valid = mask_v & np.isfinite(Z) & (Z > 1e-6)
    if valid.sum() < 100:
        return None
    x, y = (X[valid] / Z[valid]), (Y[valid] / Z[valid])
    u, v = uu[valid], vv[valid]
    A = np.stack([x, np.ones_like(x)], axis=1)
    fx, cx = np.linalg.lstsq(A, u, rcond=None)[0]
    A = np.stack([y, np.ones_like(y)], axis=1)
    fy, cy = np.linalg.lstsq(A, v, rcond=None)[0]
    return fx, fy, cx, cy

# ==========================================
# TSDF VOLUME SETUP (scale-adaptive: Pi3 output scale is arbitrary)
# ==========================================
all_valid = world_pts[masks]
all_valid = all_valid[np.isfinite(all_valid).all(axis=1)]
bb_min, bb_max = all_valid.min(axis=0), all_valid.max(axis=0)
scene_diag = float(np.linalg.norm(bb_max - bb_min))
voxel = scene_diag / VOXEL_DIVISOR
depth_far = float(np.percentile(local_pts[..., 2][masks], 99)) * 1.05

print(f"Scene diagonal: {scene_diag:.3f} | TSDF voxel: {voxel:.5f} | depth trunc: {depth_far:.3f}")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel,
    sdf_trunc=voxel * SDF_TRUNC_MUL,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# ==========================================
# INTEGRATE EVERY VIEW (this is where double walls die)
# ==========================================
imgs_np = imgs[0].permute(0, 2, 3, 1).float().cpu().numpy()  # (V,H,W,3) in 0..1

print("\nIntegrating views into TSDF volume...")
integrated = 0
for v in range(num_views):
    mask_v = masks[v]
    intr = fit_intrinsics(local_pts[v], mask_v)
    if intr is None:
        print(f"  -> view {v}: too few valid points, skipped")
        continue
    fx, fy, cx, cy = intr

    depth = local_pts[v, ..., 2].astype(np.float32).copy()
    bad = (~mask_v) | ~np.isfinite(depth) | (depth <= 0)
    depth[bad] = 0.0                                   # 0 = ignored by Open3D

    color = np.ascontiguousarray(
        (np.clip(imgs_np[v], 0, 1) * 255).astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=depth_far,
        convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(poses[v])                # world -> camera
    volume.integrate(rgbd, intrinsic, extrinsic)
    integrated += 1

print(f"Integrated {integrated}/{num_views} views.")

# ==========================================
# EXTRACT MESH (marching cubes) + CLEANUP
# ==========================================
print("\nExtracting mesh from TSDF (marching cubes)...")
mesh = volume.extract_triangle_mesh()

# Kill disconnected floater islands
tri_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
tri_clusters = np.asarray(tri_clusters)
cluster_sizes = np.asarray(cluster_sizes)
if len(cluster_sizes) > 1:
    keep = cluster_sizes[tri_clusters] >= cluster_sizes.max() * MIN_CLUSTER_FRAC
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    print(f"Removed {int((~keep).sum())} floater triangles "
          f"({len(cluster_sizes)} clusters found).")

if TAUBIN_ITERS > 0:
    print(f"Taubin smoothing ({TAUBIN_ITERS} iters, volume-preserving)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERS)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=False)
print(f"\n[SAVED] Main mesh -> {mesh_path}")

# ==========================================
# BONUS: TSDF point cloud (already has clean oriented normals)
# + optional Poisson on top of it
# ==========================================
pcd = volume.extract_point_cloud()   # points + colors + PROPER normals
o3d.io.write_point_cloud(cloud_path, pcd, write_ascii=False)
print(f"[SAVED] Poisson-ready cloud -> {cloud_path}")

if RUN_POISSON_TOO:
    print(f"\nRunning Poisson (depth={POISSON_DEPTH}) on TSDF cloud...")
    pmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    # Trim the low-density "balloon" regions Poisson hallucinates
    densities = np.asarray(densities)
    pmesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.03))
    pmesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(poisson_path, pmesh, write_ascii=False)
    print(f"[SAVED] Poisson mesh -> {poisson_path}")

print("\nDone. Single wall, smooth surface, no MLS mush.")
