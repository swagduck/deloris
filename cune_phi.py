# upt_holo_full.py
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from voxel_generator import generate_cube, generate_sphere  # must return arrays shape (N,3) dtype float32

# -----------------------
# Config
# -----------------------
RES = 36
FPS = 15
FRAMES = 30 # <-- ĐÃ GIẢM TỪ 150 XUỐNG 30 ĐỂ TĂNG TỐC
OUT_VIDEO = "UPT_PHI_CLEAN_FULL.mp4"
EXPORT_PHASE_DIR = "phase_maps"    # set None to disable
EXPORT_INT_DIR = "int_maps"        # set None to disable
CHUNK_SIZE = 2000                  # chunk per-source to limit memory
WAVELENGTH = 532e-9                # 532 nm
K = 2 * np.pi / WAVELENGTH
AMP_SCALE = 50.0
EPS = 1e-8
BIAS_R = 0.01

os.makedirs(EXPORT_PHASE_DIR, exist_ok=True) if EXPORT_PHASE_DIR else None
os.makedirs(EXPORT_INT_DIR, exist_ok=True) if EXPORT_INT_DIR else None

# -----------------------
# Grid and flattened coordinates
# -----------------------
x = np.linspace(-0.05, 0.05, RES).astype(np.float32)
y = np.linspace(-0.05, 0.05, RES).astype(np.float32)
z = np.linspace(-0.05, 0.05, RES).astype(np.float32)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
Xf, Yf, Zf = X.ravel().astype(np.float32), Y.ravel().astype(np.float32), Z.ravel().astype(np.float32)

# -----------------------
# Voxels
# -----------------------
# Truyền RES (36) và các tham số khác vào generator
# (Dựa trên config của cune_phi và file UPT_Holo.pdf)
cube = generate_cube(side=0.03, N=RES).astype(np.float32)
sphere = generate_sphere(cube, radius=0.04).astype(np.float32)

# -----------------------
# Utility functions
# -----------------------
def P_phi_k(coords):
    r = np.linalg.norm(coords, axis=1)
    R_max = 0.03 * np.sqrt(3)
    return np.clip(1.0 - (r / R_max)**2, 0.3, 1.0).astype(np.float32)

def ease(t):
    return t * t * (3 - 2 * t)

# -----------------------
# Compute field (chunked, entanglement toggle)
# -----------------------
def compute_field_chunked(frame, entangled=True, chunk_size=CHUNK_SIZE):
    t = frame * 0.08
    morph = ease((np.sin(2 * np.pi * 0.12 * t) + 1) / 2)
    coords_full = ((1 - morph) * cube + morph * sphere).astype(np.float32)
    P_full = P_phi_k(coords_full).astype(np.float32)
    mask = P_full > 0.3
    coords = coords_full[mask]
    P_active = P_full[mask]

    Np = Xf.shape[0]
    field = np.zeros(Np, dtype=np.complex64)

    Xf_b = Xf[None, :]  # (1, Np)
    Yf_b = Yf[None, :]
    Zf_b = Zf[None, :]

    rng = np.random.RandomState(frame)  # reproducible per-frame randomness
    random_phase_offset = 0.0 if entangled else rng.uniform(-2*np.pi, 2*np.pi)

    for s0 in range(0, coords.shape[0], chunk_size):
        chunk = coords[s0:s0 + chunk_size].astype(np.float32)        # (C,3)
        Pch = P_active[s0:s0 + chunk_size].astype(np.float32)[:, None]  # (C,1)

        dx = Xf_b - chunk[:, 0:1]   # (C, Np)
        dy = Yf_b - chunk[:, 1:2]
        dz = Zf_b - chunk[:, 2:3]
        r = np.sqrt(dx * dx + dy * dy + dz * dz + EPS, dtype=np.float32)  # (C,Np)

        amp = (Pch * AMP_SCALE) / (r + BIAS_R)  # (C,Np)
        phase = (K * r + t * 4.0 + random_phase_offset).astype(np.float32)  # (C,Np)

        # use trig to control dtype and stability
        field += np.sum(amp * (np.cos(phase) + 1j * np.sin(phase)), axis=0)

    intensity = (np.abs(field) ** 2).astype(np.float32)
    phase_map = np.angle(field).astype(np.float32)  # in [-pi, pi]
    return intensity, phase_map, morph

# -----------------------
# Figure helper
# -----------------------
def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 4))
    return buf[:, :, :3].copy()

# -----------------------
# Main render loop (safe writer)
# -----------------------
def main_render(entangled=True, display_mode="3d_scatter"):
    fig = plt.figure(figsize=(10, 6), facecolor='black', dpi=100)
    ax = fig.add_subplot(111, projection='3d') if display_mode == "3d_scatter" else fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.axis('off')

    print("🎬 Rendering video...")
    with imageio.get_writer(OUT_VIDEO, fps=FPS, quality=9) as writer:
        for frame in range(FRAMES):
            I, phase, morph = compute_field_chunked(frame, entangled=entangled, chunk_size=CHUNK_SIZE)

            # export maps optionally
            if EXPORT_INT_DIR:
                I3 = I.reshape((RES, RES, RES))
                np.save(os.path.join(EXPORT_INT_DIR, f"int_frame_{frame:03d}.npy"), I3)
            if EXPORT_PHASE_DIR:
                np.save(os.path.join(EXPORT_PHASE_DIR, f"phase_frame_{frame:03d}.npy"), phase.reshape((RES, RES, RES)))

            # determine display indices
            thresh = np.percentile(I, 95.0)
            idx = np.where(I > thresh)[0]

            ax.cla()
            ax.set_facecolor('black')
            ax.axis('off')

            if display_mode == "3d_scatter":
                if len(idx) == 0:
                    current = ((1 - morph) * cube + morph * sphere)
                    ax.scatter(current[:, 0], current[:, 1], current[:, 2], c='white', s=4, alpha=0.6)
                else:
                    ax.scatter(Xf[idx], Yf[idx], Zf[idx], c=phase[idx], cmap='hsv', s=8, alpha=1.0, linewidths=0)
                ax.view_init(elev=25, azim=frame * 1.8)
                ax.set_xlim(-0.05, 0.05); ax.set_ylim(-0.05, 0.05); ax.set_zlim(-0.05, 0.05)
                ax.set_title(f"UPT-Φ | morph={morph:.2f}", color='cyan', fontsize=14, pad=20)
            else:
                I3 = I.reshape((RES, RES, RES))
                mip = I3.max(axis=2)
                ax.imshow(np.log1p(mip), cmap='inferno', origin='lower')
                ax.set_title(f"UPT-Φ MIP | morph={morph:.2f}", color='cyan')

            img = fig_to_rgb(fig)
            writer.append_data(img)
            print(f"Frame {frame+1}/{FRAMES} written", end='\r')
    plt.close(fig)
    print(f"\n✅ Saved: {OUT_VIDEO}")

if __name__ == "__main__":
    main_render(entangled=True, display_mode="3d_scatter")