"""
Interactive 3D volume data visualization using PyVista.

Supports:
  - Volume rendering
  - Isosurface extraction
  - Orthogonal slice views (XY, XZ, YZ planes)

Usage:
    python scripts/visualize.py --file data/volume.vti --mode volume
    python scripts/visualize.py --file data/volume.vti --mode volume --cmap coolwarm
    python scripts/visualize.py --file data/volume.vti --mode isosurface --isovalue 0.5
    python scripts/visualize.py --file data/volume.vti --mode slices --array Pressure

Available colormaps:
    viridis   — perceptually uniform, good general purpose (default)
    plasma    — perceptually uniform, high contrast
    coolwarm  — diverging blue-to-red, good for signed data
    hot       — black-red-yellow-white, good for intensity fields
    gray      — grayscale, good for medical/CT data
    jet       — classic rainbow (not perceptually uniform)
    turbo     — improved rainbow with better perceptual uniformity
"""

import argparse
import numpy as np
import pyvista as pv


def load_vti(path: str, array_name: str = None) -> tuple[pv.ImageData, str]:
    """Load a .vti file and return the grid and active array name."""
    grid = pv.read(path)
    if not isinstance(grid, pv.ImageData):
        raise ValueError(f"Expected a vtkImageData (.vti) file, got {type(grid)}")

    # Pick array to visualize
    available = grid.array_names
    if not available:
        raise ValueError("No data arrays found in the .vti file.")

    if array_name:
        if array_name not in available:
            raise ValueError(f"Array '{array_name}' not found. Available: {available}")
        name = array_name
    else:
        name = available[0]
        if len(available) > 1:
            print(f"Multiple arrays found: {available}. Using '{name}'. Use --array to select.")

    print(f"Loaded: {path}")
    print(f"  Dimensions : {grid.dimensions}")
    print(f"  Spacing    : {grid.spacing}")
    print(f"  Array      : '{name}'  (min={grid[name].min():.4f}, max={grid[name].max():.4f})")
    return grid, name


def make_demo_vti(shape=(64, 64, 64)) -> tuple[pv.ImageData, str]:
    """Generate a demo .vti-style grid with a Gaussian scalar field."""
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    values = np.exp(-(X**2 + Y**2 + Z**2) / 0.3).flatten(order="F")

    grid = pv.ImageData()
    grid.dimensions = np.array(shape) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.cell_data["values"] = values.astype(np.float32)
    return grid, "values"


COLORMAPS = ["viridis", "plasma", "coolwarm", "hot", "gray", "jet", "turbo"]


def visualize_volume(grid: pv.ImageData, array: str, cmap: str = "viridis"):
    """Direct volume rendering."""
    pl = pv.Plotter()
    pl.add_volume(grid, scalars=array, cmap=cmap, opacity="sigmoid")
    pl.add_scalar_bar(title=array)
    pl.show(title=f"Volume Rendering [{cmap}]")


def visualize_isosurface(grid: pv.ImageData, array: str, isovalue: float):
    """Extract and render an isosurface at the given value."""
    surface = grid.contour([isovalue], scalars=array)
    if surface.n_points == 0:
        print(f"No isosurface found at isovalue={isovalue}. Try a different value.")
        return
    pl = pv.Plotter()
    pl.add_mesh(surface, scalars=array, cmap="viridis", smooth_shading=True)
    pl.add_scalar_bar(title=array)
    pl.show(title=f"Isosurface (isovalue={isovalue})")


def visualize_slices(grid: pv.ImageData, array: str):
    """Render three orthogonal slices through the center of the volume."""
    cx, cy, cz = [s / 2 for s in grid.dimensions]
    slices = [
        grid.slice(normal="x", origin=(cx, cy, cz)),
        grid.slice(normal="y", origin=(cx, cy, cz)),
        grid.slice(normal="z", origin=(cx, cy, cz)),
    ]
    pl = pv.Plotter()
    for s in slices:
        pl.add_mesh(s, scalars=array, cmap="viridis", show_edges=False)
    pl.add_scalar_bar(title=array)
    pl.show(title="Orthogonal Slices")


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D volume visualizer for .vti files")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a .vti file (omit to use a demo volume)")
    parser.add_argument("--mode", type=str, default="volume",
                        choices=["volume", "isosurface", "slices"],
                        help="Visualization mode")
    parser.add_argument("--array", type=str, default=None,
                        help="Name of the data array to visualize (default: first array)")
    parser.add_argument("--isovalue", type=float, default=0.5,
                        help="Isovalue for isosurface mode")
    parser.add_argument("--cmap", type=str, default="viridis", choices=COLORMAPS,
                        help="Colormap for volume rendering (default: viridis)")
    args = parser.parse_args()

    if args.file:
        grid, array = load_vti(args.file, args.array)
    else:
        print("No file provided — using demo Gaussian volume (64x64x64)")
        grid, array = make_demo_vti()

    if args.mode == "volume":
        visualize_volume(grid, array, cmap=args.cmap)
    elif args.mode == "isosurface":
        visualize_isosurface(grid, array, args.isovalue)
    elif args.mode == "slices":
        visualize_slices(grid, array)


if __name__ == "__main__":
    main()
