from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ========= Path configuration =========
input_dir = Path(r"C:\Users\user\Downloads\newfig2\Usable 2\processed\adjusted_output")
output_dir = input_dir / "left_half_two_rotated_shapes_final_with_diagonals"
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "left_half_two_rotated_shapes_summary.csv"

# Supported image formats
image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Horizontal edge scale factor: only affects horizontal edge length,
# does not affect the locked left edge length
SIDE_SCALE = 0.97


def detect_left_4_contours(img_bgr, bin_thresh=100, min_area=200):
    """
    Detect bright contours only in the left region, return up to 4 contours
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    left_half = gray[:, :w // 3]

    _, th = cv2.threshold(left_half, bin_thresh, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)

    selected = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        if area < min_area:
            continue
        if x >= left_half.shape[1]:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        selected.append((x, y, ww, hh, area, c))

    selected = sorted(selected, key=lambda t: (t[1], t[0]))
    return selected[:4]


def annotate_and_extract_points(img_bgr, contours_info):
    """
    Draw contour points on the image and extract the highest/lowest point of each contour
    """
    annot = img_bgr.copy()
    results = []

    for idx, (x, y, ww, hh, area, c) in enumerate(contours_info, start=1):
        c_global = c.copy()
        pts = c_global[:, 0, :]

        top_pt = tuple(pts[np.argmin(pts[:, 1])])
        bottom_pt = tuple(pts[np.argmax(pts[:, 1])])

        # cv2.drawContours(annot, [c_global], -1, (0, 0, 255), 3)
        cv2.circle(annot, top_pt, 6, (0, 255, 0), -1)      # Green: highest point
        cv2.circle(annot, bottom_pt, 6, (255, 0, 0), -1)   # Blue: lowest point

        label_pos = (int(x), max(30, int(y) - 10))
        cv2.putText(
            annot,
            f"C{idx}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        results.append({
            "contour_id": f"C{idx}",
            "top_point": top_pt,
            "bottom_point": bottom_pt,
            "top_x": int(top_pt[0]),
            "top_y": int(top_pt[1]),
            "bottom_x": int(bottom_pt[0]),
            "bottom_y": int(bottom_pt[1]),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(ww),
            "bbox_h": int(hh),
            "area": int(area),
            "height_px": int(bottom_pt[1] - top_pt[1]),
        })

    return annot, results


def choose_reference_contour(results):
    """
    Select the reference contour:
    1. First find the top two contours (sorted by top_y)
    2. Then choose the one with the greater own height among these two
    """
    top_two_contours = sorted(results, key=lambda r: (r["top_y"], r["top_x"]))[:2]

    if len(top_two_contours) == 0:
        return None
    if len(top_two_contours) == 1:
        return top_two_contours[0]

    ref_contour = max(
        top_two_contours,
        key=lambda r: (r["bottom_y"] - r["top_y"], -r["top_x"])
    )
    return ref_contour


def get_top_two_contours(results):
    """
    Return the top two contours, sorted from left to right by x
    """
    top_two = sorted(results, key=lambda r: (r["top_y"], r["top_x"]))[:2]
    return sorted(top_two, key=lambda r: r["top_x"])


def get_bottom_two_contours(results):
    """
    Return the bottom two contours, sorted from left to right by x
    """
    bottom_two = sorted(results, key=lambda r: (r["bottom_y"], r["bottom_x"]), reverse=True)[:2]
    return sorted(bottom_two, key=lambda r: r["bottom_x"])


def get_scale_from_ref(ref_contour_top, ref_contour_bottom):
    """
    Use the height difference of the reference contour itself = 300 microns
    """
    ref_pixel_height = ref_contour_bottom[1] - ref_contour_top[1]
    if ref_pixel_height <= 0:
        return None

    pixel_per_um = ref_pixel_height / 300.0
    um_per_pixel = 300.0 / ref_pixel_height
    return ref_pixel_height, pixel_per_um, um_per_pixel


def build_left_shape_manual_locked_edge(
    top_point,
    bottom_point,
    ref_contour_top,
    ref_contour_bottom,
    top_dx_um,
    top_dy_um,
    bottom_dx_um,
    bottom_dy_um,
    side_scale=1.0,
):
    """
    Larger shape on the left:
    - Top-left vertex: highest point shifted right by top_dx_um and down by top_dy_um
    - Bottom-left vertex: lowest point shifted right by bottom_dx_um and up by bottom_dy_um
    - The left edge is fully locked and is not affected by side_scale
    - The horizontal edge is perpendicular to the locked left edge,
      and its length = left edge length * side_scale
    """
    scale = get_scale_from_ref(ref_contour_top, ref_contour_bottom)
    if scale is None:
        return None
    ref_pixel_height, pixel_per_um, um_per_pixel = scale

    p1_left_top = np.array([
        top_point[0] + top_dx_um * pixel_per_um,
        top_point[1] + top_dy_um * pixel_per_um
    ], dtype=np.float64)

    p2_left_bottom = np.array([
        bottom_point[0] + bottom_dx_um * pixel_per_um,
        bottom_point[1] - bottom_dy_um * pixel_per_um
    ], dtype=np.float64)

    v = p2_left_bottom - p1_left_top
    left_edge_px = np.linalg.norm(v)
    if left_edge_px < 1e-6:
        return None

    u = v / left_edge_px
    width_px = left_edge_px * side_scale
    width_um = width_px * um_per_pixel
    left_edge_um = left_edge_px * um_per_pixel

    # Perpendicular direction, expand to the right
    n = np.array([u[1], -u[0]], dtype=np.float64)

    p4_right_top = p1_left_top + n * width_px
    p3_right_bottom = p2_left_bottom + n * width_px

    center = (p1_left_top + p2_left_bottom + p3_right_bottom + p4_right_top) / 4.0

    return {
        "mode": "manual_locked_edge",
        "reference_pixel_height": float(ref_pixel_height),
        "pixel_per_um": float(pixel_per_um),
        "um_per_pixel": float(um_per_pixel),

        "left_edge_px": float(left_edge_px),
        "left_edge_um": float(left_edge_um),
        "width_px": float(width_px),
        "width_um": float(width_um),

        "p1_left_top": tuple(np.round(p1_left_top).astype(int)),
        "p2_left_bottom": tuple(np.round(p2_left_bottom).astype(int)),
        "p3_right_bottom": tuple(np.round(p3_right_bottom).astype(int)),
        "p4_right_top": tuple(np.round(p4_right_top).astype(int)),
        "center": tuple(center),
    }


def build_right_shape_auto_bottom_x(
    top_anchor_point,
    bottom_anchor_point,
    direction_top_point,
    direction_bottom_point,
    ref_contour_top,
    ref_contour_bottom,
    top_dx_um,
    top_dy_um,
    bottom_dy_um,
    side_scale=1.0,
):
    """
    Smaller shape on the right:
    - Top-left vertex: obtained by shifting the highest point of the upper-right contour
    - Left edge direction: determined by the upper-right contour's own highest point -> lowest point
    - Bottom-left vertex:
        y = lowest point of the lower-right contour shifted upward by bottom_dy_um
        x = automatically computed according to the direction
    - Once the left edge is locked, it is not affected by side_scale
    - The horizontal edge is perpendicular to the locked left edge,
      and its length = left edge length * side_scale
    """
    scale = get_scale_from_ref(ref_contour_top, ref_contour_bottom)
    if scale is None:
        return None
    ref_pixel_height, pixel_per_um, um_per_pixel = scale

    # Fixed offset for the top-left vertex
    p1_left_top = np.array([
        top_anchor_point[0] + top_dx_um * pixel_per_um,
        top_anchor_point[1] + top_dy_um * pixel_per_um
    ], dtype=np.float64)

    # Left edge direction determined by the upper-right contour's own highest -> lowest point
    dir_top = np.array(direction_top_point, dtype=np.float64)
    dir_bottom = np.array(direction_bottom_point, dtype=np.float64)
    v_dir = dir_bottom - dir_top

    dir_len = np.linalg.norm(v_dir)
    if dir_len < 1e-6:
        return None

    u = v_dir / dir_len

    # Bottom-left point: y is locked to the lower-right contour's lowest point shifted upward by bottom_dy_um
    y_bottom_locked = float(bottom_anchor_point[1] - bottom_dy_um * pixel_per_um)

    if abs(u[1]) < 1e-8:
        return None

    # p2 = p1 + t*u, and p2_y = y_bottom_locked
    p2_left_bottom = p1_left_top + (y_bottom_locked - p1_left_top[1]) / u[1] * u

    left_edge_px = np.linalg.norm(p2_left_bottom - p1_left_top)
    if left_edge_px < 1e-6:
        return None

    width_px = left_edge_px * side_scale
    width_um = width_px * um_per_pixel
    left_edge_um = left_edge_px * um_per_pixel

    # Perpendicular direction, expand to the right
    n = np.array([u[1], -u[0]], dtype=np.float64)

    p4_right_top = p1_left_top + n * width_px
    p3_right_bottom = p2_left_bottom + n * width_px

    center = (p1_left_top + p2_left_bottom + p3_right_bottom + p4_right_top) / 4.0

    auto_bottom_dx_px = p2_left_bottom[0] - bottom_anchor_point[0]
    auto_bottom_dx_um = auto_bottom_dx_px * um_per_pixel

    return {
        "mode": "auto_bottom_x",
        "reference_pixel_height": float(ref_pixel_height),
        "pixel_per_um": float(pixel_per_um),
        "um_per_pixel": float(um_per_pixel),

        "direction_top_x": float(dir_top[0]),
        "direction_top_y": float(dir_top[1]),
        "direction_bottom_x": float(dir_bottom[0]),
        "direction_bottom_y": float(dir_bottom[1]),

        "bottom_anchor_x": float(bottom_anchor_point[0]),
        "bottom_anchor_y": float(bottom_anchor_point[1]),
        "bottom_anchor_y_shifted": float(y_bottom_locked),
        "auto_bottom_dx_px": float(auto_bottom_dx_px),
        "auto_bottom_dx_um": float(auto_bottom_dx_um),

        "left_edge_px": float(left_edge_px),
        "left_edge_um": float(left_edge_um),
        "width_px": float(width_px),
        "width_um": float(width_um),

        "p1_left_top": tuple(np.round(p1_left_top).astype(int)),
        "p2_left_bottom": tuple(np.round(p2_left_bottom).astype(int)),
        "p3_right_bottom": tuple(np.round(p3_right_bottom).astype(int)),
        "p4_right_top": tuple(np.round(p4_right_top).astype(int)),
        "center": tuple(center),
    }


def draw_shape(img, shape, label, color=(255, 255, 0)):
    """
    Draw the square/rotated shape outline + two thin red diagonals + a more visible center point
    """
    p1 = tuple(shape["p1_left_top"])
    p2 = tuple(shape["p2_left_bottom"])
    p3 = tuple(shape["p3_right_bottom"])
    p4 = tuple(shape["p4_right_top"])

    pts = np.array([p1, p2, p3, p4], dtype=np.int32)

    # Outline
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=3)

    # Two diagonals: thin red lines
    cv2.line(img, p1, p3, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(img, p2, p4, (0, 0, 255), 1, cv2.LINE_AA)

    # Center point: make it more visible
    cx, cy = shape["center"]
    center_pt = (int(round(cx)), int(round(cy)))
    cv2.circle(img, center_pt, 10, (255, 255, 255), -1)  # White outer ring
    cv2.circle(img, center_pt, 6, color, -1)             # Colored inner circle

    cv2.putText(
        img,
        label,
        (p1[0], max(30, p1[1] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
    )


all_rows = []

for img_path in input_dir.iterdir():
    if not img_path.is_file():
        continue
    if img_path.suffix.lower() not in image_exts:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Read failed: {img_path.name}")
        continue

    try:
        contours_info = detect_left_4_contours(img, bin_thresh=200, min_area=300)

        if len(contours_info) < 4:
            print(f"Fewer than 4 contours: {img_path.name}")
            continue

        annot_img, results = annotate_and_extract_points(img, contours_info)

        ref_contour = choose_reference_contour(results)
        if ref_contour is None:
            print(f"Failed to find reference contour: {img_path.name}")
            continue

        ref_contour_top = ref_contour["top_point"]
        ref_contour_bottom = ref_contour["bottom_point"]

        top_two = get_top_two_contours(results)
        bottom_two = get_bottom_two_contours(results)

        if len(top_two) < 2 or len(bottom_two) < 2:
            print(f"Fewer than 2 top/bottom contours: {img_path.name}")
            continue

        left_top_contour = top_two[0]
        right_top_contour = top_two[1]
        left_bottom_contour = bottom_two[0]
        right_bottom_contour = bottom_two[1]

        # Larger shape on the left: left edge fully locked, side_scale only affects the horizontal edge
        left_shape = build_left_shape_manual_locked_edge(
            top_point=left_top_contour["top_point"],
            bottom_point=left_bottom_contour["bottom_point"],
            ref_contour_top=ref_contour_top,
            ref_contour_bottom=ref_contour_bottom,
            top_dx_um=40,
            top_dy_um=50,
            bottom_dx_um=50,
            bottom_dy_um=40,
            side_scale=SIDE_SCALE
        )
        if left_shape is None:
            print(f"Left shape construction failed: {img_path.name}")
            continue

        draw_shape(annot_img, left_shape, "LeftShape", color=(255, 255, 0))

        # Smaller shape on the right: x of bottom-left vertex is computed automatically,
        # and side_scale only affects the horizontal edge after the left edge is locked
        right_shape = build_right_shape_auto_bottom_x(
            top_anchor_point=right_top_contour["top_point"],
            bottom_anchor_point=right_bottom_contour["bottom_point"],
            direction_top_point=right_top_contour["top_point"],
            direction_bottom_point=right_top_contour["bottom_point"],
            ref_contour_top=ref_contour_top,
            ref_contour_bottom=ref_contour_bottom,
            top_dx_um=45,
            top_dy_um=40,
            bottom_dy_um=40,
            side_scale=0.96
        )
        if right_shape is None:
            print(f"Right shape construction failed: {img_path.name}")
            continue

        draw_shape(annot_img, right_shape, "RightShape", color=(255, 0, 255))

        # Center-point connecting line
        c1 = np.array(left_shape["center"])
        c2 = np.array(right_shape["center"])

        dx_px = abs(c2[0] - c1[0])
        dy_px = abs(c2[1] - c1[1])
        center_distance_px = np.linalg.norm(c2 - c1)

        dx_um = dx_px * left_shape["um_per_pixel"]
        dy_um = dy_px * left_shape["um_per_pixel"]
        center_distance_um = center_distance_px * left_shape["um_per_pixel"]

        cv2.line(
            annot_img,
            (int(round(c1[0])), int(round(c1[1]))),
            (int(round(c2[0])), int(round(c2[1]))),
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        mid_x = int(round((c1[0] + c2[0]) / 2))
        mid_y = int(round((c1[1] + c2[1]) / 2))

        cv2.putText(
            annot_img,
            f"D={center_distance_um:.2f}um",
            (mid_x + 180, mid_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annot_img,
            f"Dx={dx_um:.2f}um",
            (mid_x + 180, mid_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annot_img,
            f"Dy={dy_um:.2f}um",
            (mid_x + 180, mid_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # info_text_1 = f"Ref contour: {ref_contour['contour_id']}"
        # info_text_2 = f"SIDE_SCALE = {SIDE_SCALE:.3f}"
        # info_text_3 = f"Right auto dx = {right_shape['auto_bottom_dx_um']:.2f}um"
        #
        # cv2.putText(
        #     annot_img, info_text_1, (30, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        # )
        # cv2.putText(
        #     annot_img, info_text_2, (30, 75),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        # )
        # cv2.putText(
        #     annot_img, info_text_3, (30, 110),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        # )

        out_img_path = output_dir / f"{img_path.stem}_two_rotated_shapes_final{img_path.suffix}"
        cv2.imwrite(str(out_img_path), annot_img)

        row = {
            "image_name": img_path.name,

            "ref_contour_id": ref_contour["contour_id"],
            "ref_contour_top_x": int(ref_contour_top[0]),
            "ref_contour_top_y": int(ref_contour_top[1]),
            "ref_contour_bottom_x": int(ref_contour_bottom[0]),
            "ref_contour_bottom_y": int(ref_contour_bottom[1]),

            "reference_pixel_height_for_300um": left_shape["reference_pixel_height"],
            "pixel_per_um": left_shape["pixel_per_um"],
            "um_per_pixel": left_shape["um_per_pixel"],
            "side_scale": SIDE_SCALE,

            "left_top_contour_id": left_top_contour["contour_id"],
            "left_bottom_contour_id": left_bottom_contour["contour_id"],
            "left_shape_left_top_x": int(left_shape["p1_left_top"][0]),
            "left_shape_left_top_y": int(left_shape["p1_left_top"][1]),
            "left_shape_left_bottom_x": int(left_shape["p2_left_bottom"][0]),
            "left_shape_left_bottom_y": int(left_shape["p2_left_bottom"][1]),
            "left_shape_right_bottom_x": int(left_shape["p3_right_bottom"][0]),
            "left_shape_right_bottom_y": int(left_shape["p3_right_bottom"][1]),
            "left_shape_right_top_x": int(left_shape["p4_right_top"][0]),
            "left_shape_right_top_y": int(left_shape["p4_right_top"][1]),
            "left_shape_center_x": left_shape["center"][0],
            "left_shape_center_y": left_shape["center"][1],
            "left_shape_left_edge_px": left_shape["left_edge_px"],
            "left_shape_left_edge_um": left_shape["left_edge_um"],
            "left_shape_width_px": left_shape["width_px"],
            "left_shape_width_um": left_shape["width_um"],

            "right_top_contour_id": right_top_contour["contour_id"],
            "right_bottom_contour_id": right_bottom_contour["contour_id"],
            "right_shape_left_top_x": int(right_shape["p1_left_top"][0]),
            "right_shape_left_top_y": int(right_shape["p1_left_top"][1]),
            "right_shape_left_bottom_x": int(right_shape["p2_left_bottom"][0]),
            "right_shape_left_bottom_y": int(right_shape["p2_left_bottom"][1]),
            "right_shape_right_bottom_x": int(right_shape["p3_right_bottom"][0]),
            "right_shape_right_bottom_y": int(right_shape["p3_right_bottom"][1]),
            "right_shape_right_top_x": int(right_shape["p4_right_top"][0]),
            "right_shape_right_top_y": int(right_shape["p4_right_top"][1]),
            "right_shape_center_x": right_shape["center"][0],
            "right_shape_center_y": right_shape["center"][1],
            "right_shape_left_edge_px": right_shape["left_edge_px"],
            "right_shape_left_edge_um": right_shape["left_edge_um"],
            "right_shape_width_px": right_shape["width_px"],
            "right_shape_width_um": right_shape["width_um"],
            "right_auto_bottom_dx_px": right_shape["auto_bottom_dx_px"],
            "right_auto_bottom_dx_um": right_shape["auto_bottom_dx_um"],

            "center_distance_px": float(center_distance_px),
            "center_distance_um": float(center_distance_um),
            "center_dx_um": float(dx_um),
            "center_dy_um": float(dy_um),
        }
        all_rows.append(row)

        print(
            f"Processed: {img_path.name} | final mode + diagonals done | "
            f"D = {center_distance_um:.2f}um | Dx = {dx_um:.2f}um | Dy = {dy_um:.2f}um"
        )

    except Exception as e:
        print(f"Processed failure: {img_path.name} | error: {e}")

if all_rows:
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nResults stored: {csv_path}")
else:
    print("\nNo CSV results.")