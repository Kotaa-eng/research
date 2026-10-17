"""保存済み3D光強度を、黒い立方体背景・白い表面で可視化する。

伝搬・Bessel生成・圧縮の再計算はしない。必要ライブラリは既存のNumPyと
Matplotlibだけ。再生像は共通強度閾値を超えたボクセルの外表面を表示する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2 or not np.all(np.isfinite(centres)):
        raise ValueError("each coordinate axis needs >=2 finite centres")
    if not np.all(np.diff(centres) > 0):
        raise ValueError("coordinates must be increasing")
    return np.r_[centres[0]-(centres[1]-centres[0])/2,
                 (centres[:-1]+centres[1:])/2,
                 centres[-1]+(centres[-1]-centres[-2])/2]


def _boundary_quads(mask: np.ndarray, edges: tuple) -> tuple[np.ndarray, np.ndarray]:
    """隣接する白いセル間の内部面を除き、外側の四角形面と法線だけを作る。"""
    if mask.ndim != 3 or any(len(e) != n+1 for e, n in zip(edges, mask.shape)):
        raise ValueError("voxel mask and coordinate edges differ")
    faces, normals = [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            neighbour = np.zeros_like(mask)
            destination, source = [slice(None)]*3, [slice(None)]*3
            if sign > 0:
                destination[axis], source[axis] = slice(None, -1), slice(1, None)
            else:
                destination[axis], source[axis] = slice(1, None), slice(None, -1)
            neighbour[tuple(destination)] = mask[tuple(source)]
            indices = np.argwhere(mask & ~neighbour)
            if not len(indices):
                continue
            quads = np.empty((len(indices), 4, 3), dtype=float)
            quads[:, :, axis] = edges[axis][indices[:, axis]+int(sign > 0)][:, None]
            for corner, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                quads[:, corner, other[0]] = edges[other[0]][indices[:, other[0]]+u]
                quads[:, corner, other[1]] = edges[other[1]][indices[:, other[1]]+v]
            normal = np.zeros((len(indices), 3))
            normal[:, axis] = sign
            faces.append(quads)
            normals.append(normal)
    if not faces:
        return np.empty((0, 4, 3)), np.empty((0, 3))
    return np.concatenate(faces), np.concatenate(normals)


def _draw_surface(ax, faces: np.ndarray, normals: np.ndarray, edges: tuple,
                  title: str, elevation: float, azimuth: float, box_aspect: tuple) -> None:
    # 黒い3つの壁と白い表面。形状を変えず、面の向きで陰影だけ付ける。
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis.line.set_color("black")
    ax.grid(False)
    if len(faces):
        light = np.array([-.4, -.8, 1.2])
        light /= np.linalg.norm(light)
        brightness = .85+.15*(normals @ light)
        colours = np.c_[np.repeat(brightness[:, None], 3, axis=1), np.ones(len(brightness))]
        surface = Poly3DCollection(faces, facecolors=colours, edgecolors="none",
                                  linewidths=0, zsort="average")
        ax.add_collection3d(surface)
    else:
        ax.text2D(.5, .5, "No voxels above threshold", transform=ax.transAxes,
                  ha="center", color="red")
    # 表示の横/奥/高さはz/x/y。ラベルには実際の物理座標と単位を残す。
    ax.set(xlim=(edges[0][0], edges[0][-1]), ylim=(edges[1][0], edges[1][-1]),
           zlim=(edges[2][0], edges[2][-1]), xlabel="z [mm]", ylabel="x [mm]",
           zlabel="y [um]", title=title)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("red")
    ax.tick_params(labelsize=8, pad=0)
    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elevation, azim=azimuth)


def save_3d_surfaces(directory: str | Path, volumes: dict, target: np.ndarray,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray, *,
                     kind: str = "cgh", threshold_fraction: float = .20,
                     max_shape: tuple = (128, 96, 160), elevation: float = 25,
                     azimuth: float = -65, box_aspect: tuple = (1.5, 1.0, .9),
                     show: bool = False) -> dict:
    """3つの立体図をPNG保存。max_shapeの順番は[z,y,x]。

    目標は幾何形状（target>0）で表示。直接/提案の再生像は、どちらも
    直接法の全ROI最大強度×threshold_fractionで表面を抽出する。
    表示用のサンプリングや二値化を誤差評価・伝搬入力に戻さない。
    """
    if kind not in ("cgh", "complex"):
        raise ValueError("kind must be cgh or complex")
    if not np.isfinite(threshold_fraction) or not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must be in (0,1]")
    if len(max_shape) != 3 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2 for n in max_shape):
        raise ValueError("max_shape must contain three integers >=2")
    if (len(box_aspect) != 3 or not np.all(np.isfinite(box_aspect))
            or min(box_aspect) <= 0 or not np.isfinite(elevation) or not np.isfinite(azimuth)):
        raise ValueError("invalid view/aspect settings")
    target = np.asarray(target)
    expected = (len(z), len(y), len(x))
    if target.shape != expected:
        raise ValueError("target must have [z,y,x] shape")
    fields = [volumes["direct_"+kind], volumes["proposed_"+kind]]
    if any(field.shape != expected for field in fields):
        raise ValueError("replayed fields must have [z,y,x] shape")
    intensities = [np.abs(field)**2 for field in fields]
    if (not np.all(np.isfinite(target)) or np.any(target < 0)
            or any(not np.all(np.isfinite(value)) for value in intensities)):
        raise ValueError("volume contains invalid intensity values")
    reference_peak = float(intensities[0].max())
    if reference_peak <= 0:
        raise ValueError("direct volume has no positive intensity")
    threshold = threshold_fraction*reference_peak
    steps = tuple(max(1, int(np.ceil(n/cap))) for n, cap in zip(expected, max_shape))
    sample = tuple(slice(None, None, step) for step in steps)
    # 軸ごとの物理座標を維持。表示アスペクトだけを意図的に引き伸ばす。
    sampled_x, sampled_y, sampled_z = np.asarray(x)[sample[2]], np.asarray(y)[sample[1]], np.asarray(z)[sample[0]]
    edges = (_cell_edges(sampled_z*1e3), _cell_edges(sampled_x*1e3), _cell_edges(sampled_y*1e6))
    masks = [(target[sample] > 0).transpose(0, 2, 1),
             (intensities[0][sample] >= threshold).transpose(0, 2, 1),
             (intensities[1][sample] >= threshold).transpose(0, 2, 1)]
    meshes = [_boundary_quads(mask, edges) for mask in masks]
    name = "CGH replay" if kind == "cgh" else "Finite complex field"
    titles = ("Target geometry (not propagated)", f"Direct {name}", f"Proposed {name}")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    fig = plt.figure(figsize=(18, 6), facecolor="white", constrained_layout=True)
    for col, (mesh, title) in enumerate(zip(meshes, titles), 1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        _draw_surface(ax, *mesh, edges, title, elevation, azimuth, box_aspect)
    fig.suptitle(f"Replay surfaces: I >= {100*threshold_fraction:g}% of DIRECT peak; shared threshold; axes stretched",
                 fontsize=11)
    paths = {"comparison": str(out/f"volume_3d_surface_{kind}.png")}
    fig.savefig(paths["comparison"], dpi=180)
    figures.append(fig)
    for label, title, mesh in zip(("target", "direct", "proposed"), titles, meshes):
        fig = plt.figure(figsize=(7, 7), facecolor="white", constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        _draw_surface(ax, *mesh, edges, title, elevation, azimuth, box_aspect)
        fig.suptitle("Geometry only; axes stretched" if label == "target" else
                     f"I >= {100*threshold_fraction:g}% of DIRECT peak; axes stretched", fontsize=10)
        paths[label] = str(out/f"{label}_3d_surface_{kind}.png")
        fig.savefig(paths[label], dpi=180)
        figures.append(fig)
    metadata = {"source_kind": kind, "array_order": "[z,y,x]",
                "representation": "outer voxel faces; no smoothing or modification of simulated fields",
                "threshold_fraction_of_direct_peak": threshold_fraction,
                "direct_peak_intensity": reference_peak, "shared_replay_threshold": threshold,
                "target_display": "target>0 geometry, separate from measured intensity threshold",
                "native_shape_zyx": list(expected), "sampling_step_zyx": list(steps),
                "sampled_shape_zyx": list(target[sample].shape),
                "display_axes": ["z [mm]", "x [mm]", "y [um]"],
                "box_aspect": list(box_aspect), "elevation_deg": elevation, "azimuth_deg": azimuth,
                "thresholded_voxel_counts": [int(mask.sum()) for mask in masks],
                "surface_face_counts": [len(mesh[0]) for mesh in meshes], "files": paths,
                "scope": "Visualization only; original phase masks, arrays, summary metrics and timings unchanged."}
    metadata_path = out/f"view_settings_{kind}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    paths["settings"] = str(metadata_path)
    if show:
        # 対話GUIバックエンドでユーザーが--showを指定した場合に回転・ズームできる。
        plt.show()
    for fig in figures:
        plt.close(fig)
    return paths


def export_from_saved(result_dir: str | Path, **view_options) -> dict:
    root = Path(result_dir)
    with np.load(root/"volume_data.npz", allow_pickle=False) as saved:
        kind = view_options.get("kind", "cgh")
        if kind not in ("cgh", "complex"):
            raise ValueError("kind must be cgh or complex")
        volumes = {key: saved[key] for key in ("direct_"+kind, "proposed_"+kind)}
        target, x, y, z = (saved[key] for key in ("target_intensity", "x_m", "y_m", "z_m"))
    return save_3d_surfaces(root/"3d_surface", volumes, target, x, y, z, **view_options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", help="folder containing volume_data.npz")
    parser.add_argument("--threshold", type=float, default=.20)
    parser.add_argument("--kind", choices=("cgh", "complex"), default="cgh")
    parser.add_argument("--elevation", type=float, default=25)
    parser.add_argument("--azimuth", type=float, default=-65)
    parser.add_argument("--show", action="store_true", help="also open interactive Matplotlib windows")
    args = parser.parse_args()
    paths = export_from_saved(args.result_dir, threshold_fraction=args.threshold,
        kind=args.kind, elevation=args.elevation, azimuth=args.azimuth, show=args.show)
    for name, path in paths.items():
        print(name+": "+str(Path(path).resolve()))


if __name__ == "__main__":
    main()
