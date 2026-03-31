import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile

# Directories
colon_dir = Path(__file__).resolve().parent
seg_dir = colon_dir / "mesmer_segmentations_normal"

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
MARKERS_CSV = colon_dir / "markers.csv"

# Segmentation variants: name -> (directory, inverse_transform)
# The inverse transform undoes the spatial transform that was applied to the
# image before running Mesmer, so the label mask aligns with the original image.
SEG_VARIANTS: dict[str, tuple[Path, callable]] = {
    "normal":  (colon_dir / "mesmer_segmentations_normal",  lambda s: s),
    # All PNGs are saved at original image dimensions (1440×1920): the segmentation
    # pipeline already applied the inverse spatial transform before saving, so no
    # further transform is needed here.
    "rot180":  (colon_dir / "mesmer_segmentations_rot180",  lambda s: np.flipud(s)),
    "hf":      (colon_dir / "mesmer_segmentations_hf",      lambda s: np.flipud(s)),
    "vf":      (colon_dir / "mesmer_segmentations_vf",      lambda s: s),
    "gaussian": (colon_dir / "mesmer_segmentation_gaussian", lambda s: s),
}


def _extract_zips() -> None:
    """Extract mesmer_segmentations_*-<timestamp>.zip into their variant directories.

    The zip name is expected to match ``mesmer_segmentations_<variant>-*.zip``.
    Extraction is skipped when the target directory already exists.
    PNGs may be at the root of the archive or inside a single sub-directory —
    both layouts are handled.
    """
    for zip_path in sorted(colon_dir.glob("mesmer_segmentations_*-*.zip")):
        # Derive variant name: strip trailing "-<timestamp>" from the stem
        # e.g. mesmer_segmentations_rot90-20260330T151644Z-3-001 -> rot90
        stem_no_ext = zip_path.stem  # mesmer_segmentations_rot90-20260330T...
        # Split on the first hyphen that separates the variant from the timestamp
        prefix = stem_no_ext.split("-")[0]  # mesmer_segmentations_rot90
        variant = prefix.removeprefix("mesmer_segmentations_")  # rot90

        target_dir = colon_dir / f"mesmer_segmentations_{variant}"
        if target_dir.exists():
            continue

        print(f"Extracting {zip_path.name} -> {target_dir.name}/")
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as zf:
            png_members = [m for m in zf.namelist() if m.lower().endswith(".png")]
            for member in png_members:
                # Flatten: always write to target_dir/<filename>
                fname = Path(member).name
                data = zf.read(member)
                (target_dir / fname).write_bytes(data)


def _load_markers_csv(n_channels: int) -> list[str] | None:
    """Load channel names from markers.csv (1-indexed channel column)."""
    if not MARKERS_CSV.exists():
        return None
    df = pd.read_csv(MARKERS_CSV)
    if "channel_name" not in df.columns or "channel" not in df.columns:
        return None
    df = df.sort_values("channel").reset_index(drop=True)
    names = df["channel_name"].astype(str).tolist()
    if len(names) != n_channels:
        print(f"  [warn] markers.csv has {len(names)} entries but image has {n_channels} channels; ignoring CSV")
        return None
    return names


def _parse_channel_names(tif: tifffile.TiffFile) -> list[str] | None:
    """Extract channel names from OME-XML metadata, or return None."""
    try:
        ome_xml = tif.ome_metadata
        if ome_xml is None:
            return None
        root = ET.fromstring(ome_xml)
        channels = root.findall(f".//{{{OME_NS}}}Channel")
        names = [ch.get("Name") or ch.get("Fluor") or ch.get("ID") for ch in channels]
        names = [n for n in names if n]
        return names if names else None
    except Exception:
        return None


def process_region(tiff_path: Path) -> None:
    # Derive stem: strip .ome.tiff
    stem = tiff_path.name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    normal_seg_path = SEG_VARIANTS["normal"][0] / f"{stem}.png"
    if not normal_seg_path.exists():
        print(f"  [skip] no normal segmentation found for {stem}")
        return

    print(f"Processing {stem} ...")

    # --- Read image ---
    with tifffile.TiffFile(tiff_path) as tif:
        arr = tif.asarray()  # shape may be (C,Y,X), (Z,C,Y,X), (T,Z,C,Y,X), etc.
        channel_names = _parse_channel_names(tif)

    # Normalise to (C, Y, X): squeeze length-1 dims then handle 2D/3D
    arr = arr.squeeze()
    if arr.ndim == 2:
        # Single-channel grayscale → add C dim
        arr = arr[np.newaxis, :, :]            # (1, Y, X)
    elif arr.ndim == 3:
        # Could be (C, Y, X) or (Z, Y, X) — assume (C, Y, X) as per OME convention
        pass
    elif arr.ndim >= 4:
        # (Z, C, Y, X) or larger: take the middle Z slice
        while arr.ndim > 3:
            mid = arr.shape[0] // 2
            arr = arr[mid]

    n_channels = arr.shape[0]

    # Priority: markers.csv > OME-XML metadata > generic ch0..chN
    channel_names = (
        _load_markers_csv(n_channels)
        or (channel_names if channel_names and len(channel_names) == n_channels else None)
        or [f"ch{i}" for i in range(n_channels)]
    )

    # --- Parse image element ---
    img_el = sd.models.Image2DModel.parse(arr, dims=("c", "y", "x"))
    img_el.coords["c"] = np.array(channel_names)

    # --- Read & parse segmentations (all variants) ---
    # "normal" must exist (checked above); others are included if their PNG is present.
    labels: dict[str, sd.models.Labels2DModel] = {}
    for variant, (seg_variant_dir, inv_fn) in SEG_VARIANTS.items():
        variant_png = seg_variant_dir / f"{stem}.png"
        if not variant_png.exists():
            if variant != "normal":
                print(f"  [skip variant] {variant} — no PNG for {stem}")
            continue

        seg = np.asarray(iio.imread(variant_png))
        # Some segmentation PNGs may be (H, W, C); take first channel if so
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        # Apply inverse spatial transform so labels align with the original image
        seg = np.ascontiguousarray(inv_fn(seg))

        label_key = f"{stem}_labels" if variant == "normal" else f"{stem}_labels_{variant}"
        labels[label_key] = sd.models.Labels2DModel.parse(seg, dims=("y", "x"))

    # --- Build SpatialData ---
    sdata = sd.SpatialData(
        images={f"{stem}_image": img_el},
        labels=labels,
    )

    # --- Write ---
    path_write = colon_dir / f"{stem}.zarr"
    if path_write.exists():
        shutil.rmtree(path_write)
    sdata.write(path_write)

    print(f"  Written to {path_write}")
    print(f"  {sdata}")


if __name__ == "__main__":
    _extract_zips()
    tiff_files = sorted(colon_dir.glob("*.ome.tiff")) + sorted(colon_dir.glob("*.ome.tif"))
    if not tiff_files:
        print("No OME-TIFF files found in", colon_dir)
    for tiff_path in tiff_files:
        process_region(tiff_path)
    print("done")
