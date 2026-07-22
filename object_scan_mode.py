# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
# Install dependency first if needed:  pip install open3d

import torch
import numpy as np
import os
import open3d as o3d
from scipy.spatial import cKDTree

# Import Pi3 modules
from pi3.utils.basic import load_multimodal_data
from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3x import Pi3X

# ==========================================
# USER SETTINGS (edit these paths)
# ==========================================
# Path to your input: either a single .mp4 video file,
# or a directory containing your input image(s)
data_path = './input_data'

# Where the final PLY will be saved
output_dir = './Pi3_Outputs'

# ==========================================
# TERRITORY LOCK SETTINGS (the anti-distortion gate)
# ==========================================
# Any new point landing closer than this to already-scanned surface
# is considered "already covered" and is REJECTED. It can never push,
# thicken, or distort the existing wall. (meters, in model scale)
COVERAGE_RADIUS = 0.02

# If a view contributes less than this fraction of NEW territory,
# the entire view is deleted as useless and never used.
MIN_NOVEL_RATIO = 0.10

# If ICP alignment quality (fitness) is below this, the view is
# untrustworthy -> deleted instead of smearing the scan.
MIN_ICP_FITNESS = 0.30

# ==========================================
# 2. PLANAR PROJECTION (THE "SURFACE IRON")
# ==========================================
def apply_planar_projection(points_np, colors_np=None, k_neighbors=30, iterations=2):
    """
    Vectorized MLS Smoothing. Acts as a surface iron to remove tiny 
    high-frequency bumps (elevations) for perfect Poisson reconstruction.
    """
    points = np.copy(points_np.astype(np.float64))

    for it in range(iterations):
        print(f"  -> Ironing surface bumps: Iteration {it + 1}/{iterations}...")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
        normals = np.asarray(pcd.normals)

        tree = cKDTree(points)
        _, idx = tree.query(points, k=k_neighbors)

        neighbors = points[idx]
        centroids = np.mean(neighbors, axis=1)
        vector_to_point = points - centroids

        distance_to_plane = np.sum(vector_to_point * normals, axis=1)
        points = points - (distance_to_plane[:, np.newaxis] * normals)

    return points, colors_np

# ==========================================
# 3. OUTPUT / INPUT VALIDATION
# ==========================================
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'perfect_poisson_ready_mesh.ply')

if not os.path.exists(data_path):
    raise ValueError(f"Input path not found: {data_path}. Execution stopped.")

print(f"\n[Success] Data loaded from: {data_path}")

# ==========================================
# 4. MAIN INFERENCE PIPELINE
# ==========================================
interval = 10 if data_path.endswith('.mp4') else 1
conditions_path = None
ckpt = None
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'\nUsing device: {device_name}')
print(f'Sampling interval: {interval}')

device = torch.device(device_name)
conditions = dict(intrinsics=None, poses=None, depths=None)

imgs, conditions = load_multimodal_data(data_path, conditions, interval=interval, device=device)
use_multimodal = any(v is not None for v in conditions.values())
if not use_multimodal:
    print("No multimodal conditions found.")

print("Loading Pi3X model...")
if ckpt is not None:
    model = Pi3X(use_multimodal=use_multimodal).eval()
    if ckpt.endswith('.safetensors'):
        from safetensors.torch import load_file
        weight = load_file(ckpt)
    else:
        weight = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(weight, strict=False)
else:
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    if not use_multimodal:
        model.disable_multimodal()
model = model.to(device)

print("Running model inference...")
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        res = model(imgs=imgs, **conditions)

masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
masks = torch.logical_and(masks, non_edge)[0]

# ==========================================
# 5. PER-VIEW EXTRACTION
# ==========================================
num_views = res['points'][0].shape[0]
pcds = []

print(f"\nExtracting {num_views} separate views into distinct point clouds...")
for v in range(num_views):
    view_mask = masks[v]
    v_points = res['points'][0, v][view_mask].cpu().numpy()
    v_colors = imgs[0, v].permute(1, 2, 0)[view_mask].cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(v_colors.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcds.append(pcd)

# ==========================================
# 6. FIRST-SCAN-WINS FUSION (TERRITORY LOCK)
# ==========================================
# Core rule: the overlap between views is used for ICP ALIGNMENT ONLY.
# After alignment, any point falling inside already-scanned territory
# is rejected before it can touch the master cloud. Existing geometry
# is never pushed, never thickened, never distorted.

print("\nPerforming First-Scan-Wins fusion...")

master_pcd = pcds[0]
master_points = np.asarray(master_pcd.points)
master_colors = np.asarray(master_pcd.colors)
master_normals = np.asarray(master_pcd.normals)
master_tree = cKDTree(master_points)

kept_views = 1
deleted_views = []

for v in range(1, num_views):
    source_pcd = pcds[v]

    # --- Step A: align using FULL overlap (alignment needs the overlap) ---
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, master_pcd, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    if reg_p2l.fitness < MIN_ICP_FITNESS:
        print(f"  [View {v}] DELETED -> ICP fitness {reg_p2l.fitness:.2f} too low (unreliable alignment).")
        deleted_views.append(v)
        continue

    source_pcd.transform(reg_p2l.transformation)

    src_points = np.asarray(source_pcd.points)
    src_colors = np.asarray(source_pcd.colors)
    src_normals = np.asarray(source_pcd.normals)

    # --- Step B: the territory gate ---
    # Distance of every new point to the nearest already-scanned point.
    dist, _ = master_tree.query(src_points, k=1, workers=-1)
    novel_mask = dist > COVERAGE_RADIUS
    novel_ratio = float(novel_mask.mean()) if len(novel_mask) > 0 else 0.0

    # --- Step C: delete useless angles entirely ---
    if novel_ratio < MIN_NOVEL_RATIO:
        print(f"  [View {v}] DELETED -> only {novel_ratio*100:.1f}% new territory (already scanned).")
        deleted_views.append(v)
        continue

    # --- Step D: accept ONLY the points in empty space ---
    new_points = src_points[novel_mask]
    new_colors = src_colors[novel_mask]
    new_normals = src_normals[novel_mask]

    master_points = np.vstack([master_points, new_points])
    master_colors = np.vstack([master_colors, new_colors])
    master_normals = np.vstack([master_normals, new_normals])

    # Rebuild master cloud + tree so the next view aligns against
    # the full accumulated territory (better than aligning to view 0 only).
    master_pcd = o3d.geometry.PointCloud()
    master_pcd.points = o3d.utility.Vector3dVector(master_points)
    master_pcd.colors = o3d.utility.Vector3dVector(master_colors)
    master_pcd.normals = o3d.utility.Vector3dVector(master_normals)
    master_tree = cKDTree(master_points)

    kept_views += 1
    print(f"  [View {v}] KEPT -> fitness {reg_p2l.fitness:.2f}, "
          f"{novel_ratio*100:.1f}% new territory, +{len(new_points)} points.")

print(f"\nFusion summary: {kept_views}/{num_views} views used, "
      f"{len(deleted_views)} deleted: {deleted_views}")

final_pcd = master_pcd

# ==========================================
# 7. THE PERFECT CLEANUP PIPELINE
# ==========================================
# A. Light uniformity pass (double walls no longer exist thanks to the
#    territory lock, so this is just for even point spacing).
print("\nUniformizing point spacing...")
final_pcd = final_pcd.voxel_down_sample(voxel_size=0.01)

# B. Statistical cleanup to destroy any leftover ghosting floaters
print("Executing Statistical Outlier Removal to destroy ghosting...")
final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)

# C. Iron out the tiny elevations for a silky smooth surface
#    (now mostly polishing the seams between view patches)
clean_points = np.asarray(final_pcd.points)
clean_colors = np.asarray(final_pcd.colors)

print("\nApplying Surface Iron (Vectorized MLS) to smooth tiny bumps...")
smoothed_points, smoothed_colors = apply_planar_projection(
    clean_points, clean_colors, k_neighbors=30, iterations=2
)

final_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
final_pcd.colors = o3d.utility.Vector3dVector(smoothed_colors)

# D. Bake perfect normals into the mesh for Poisson
print("\nCalculating and orienting pristine surface normals for Poisson...")
final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=40))
final_pcd.orient_normals_consistent_tangent_plane(k=40)

# E. Save using Open3D natively so the normals are actually embedded in the PLY
print(f"\nSaving final Perfect Point Cloud to: {save_path}")
o3d.io.write_point_cloud(save_path, final_pcd, write_ascii=False)
print("Processing fully complete! Your mesh is now 100% Poisson-ready.")
