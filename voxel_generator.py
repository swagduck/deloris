import numpy as np

# ================================
# Voxel Generator — Mandelbulb, lưới tam giác, custom
# ================================

def generate_cube(side=0.03, N=8):
    gx, gy, gz = np.mgrid[-side:side:N*1j, -side:side:N*1j, -side:side:N*1j]
    return np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32).T

def generate_sphere(cube_coords, radius=0.04):
    radii = np.linalg.norm(cube_coords, axis=1)
    radii[radii == 0] = 1.0
    normalized = cube_coords / radii[:, None]
    return normalized * radius

def generate_hexagonal_grid(side=0.03, N=8):
    coords = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x = (i - N/2) * side * 0.5
                y = (j - N/2) * side * np.sqrt(3)/2
                z = (k - N/2) * side * 0.8
                if (i + j + k) % 2 == 0:
                    coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)

def generate_mandelbulb(N=1000, power=8, scale=0.03):
    coords = []
    for _ in range(N):
        x, y, z = np.random.uniform(-1, 1, 3)
        r = np.sqrt(x**2 + y**2 + z**2)
        if r > 1: continue
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        new_r = r ** power
        new_theta = theta * power
        new_phi = phi * power
        nx = new_r * np.sin(new_theta) * np.cos(new_phi)
        ny = new_r * np.sin(new_theta) * np.sin(new_phi)
        nz = new_r * np.cos(new_theta)
        coords.append([nx * scale, ny * scale, nz * scale])
    return np.array(coords, dtype=np.float32)

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    cube = generate_cube()
    sphere = generate_sphere(cube)
    hexgrid = generate_hexagonal_grid()
    mandelbulb = generate_mandelbulb()

    print("Cube:", cube.shape)
    print("Sphere:", sphere.shape)
    print("HexGrid:", hexgrid.shape)
    print("Mandelbulb:", mandelbulb.shape)
