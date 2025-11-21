import argparse
import cv2
import numpy as np
import ezdxf

# --- CONFIG ------------------------------------------------------------------

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# Lower value = smaller warped image (better for screens)
PIXELS_PER_MM = 100

# Max size of debug windows on screen
MAX_DEBUG_WIDTH = 5000
MAX_DEBUG_HEIGHT = 5000


# --- UTILS -------------------------------------------------------------------

def show_debug(name, img, wait=True):
    """
    Show an image in a window, automatically scaled down to fit on screen.
    Press any key to continue if wait=True.
    """
    h, w = img.shape[:2]
    scale = min(MAX_DEBUG_WIDTH / w, MAX_DEBUG_HEIGHT / h, 1.0)

    if scale < 1.0:
        img_disp = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_disp = img

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_disp)

    if wait:
        cv2.waitKey(0)
        cv2.destroyWindow(name)


def order_points(pts):
    """
    Order 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# --- PAPER DETECTION + WARP --------------------------------------------------

def find_paper_quad(image, debug=False):
    """
    Find the largest quadrilateral in the image (assumed to be the A4 paper).
    Returns 4 points (float32).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    if debug:
        show_debug("DEBUG: 00_gray", gray)
        show_debug("DEBUG: 01_edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_quad = approx

    if best_quad is None:
        raise RuntimeError("Could not find A4 paper contour (quadrilateral) in image.")

    quad = best_quad.reshape(4, 2).astype("float32")

    if debug:
        vis = image.copy()
        cv2.drawContours(vis, [best_quad], -1, (0, 0, 255), 3)
        show_debug("DEBUG: 02_original_with_paper_quad", vis)

    return quad


def warp_to_a4(image, src_quad, debug=False):
    """
    Perspective-warp the detected paper to a flat A4 image.
    """
    width_px = int(A4_WIDTH_MM * PIXELS_PER_MM)
    height_px = int(A4_HEIGHT_MM * PIXELS_PER_MM)

    src = order_points(src_quad)
    dst = np.array(
        [
            [0, 0],
            [width_px - 1, 0],
            [width_px - 1, height_px - 1],
            [0, height_px - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width_px, height_px))

    if debug:
        show_debug("DEBUG: 03_warped_a4", warped)

    return warped


# --- THRESHOLD SLIDER --------------------------------------------------------

def interactive_threshold(a4_image, debug=False):
    """
    Show a window with a slider (trackbar) to choose the threshold value.
    Returns the chosen threshold (0–255).
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if debug:
        show_debug("DEBUG: 04_a4_gray", gray)

    window_name = "Threshold (press Enter/q/Esc when happy)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    cv2.createTrackbar("thresh", window_name, 127, 255, nothing)

    while True:
        t = cv2.getTrackbarPos("thresh", window_name)
        _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)

        # Scale for display
        h, w = thresh.shape[:2]
        scale = min(MAX_DEBUG_WIDTH / w, MAX_DEBUG_HEIGHT / h, 1.0)
        if scale < 1.0:
            disp = cv2.resize(thresh, (int(w * scale), int(h * scale)))
        else:
            disp = thresh

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 27, ord("q")):  # Enter, Esc, or q
            chosen = t
            break

    cv2.destroyWindow(window_name)

    if debug:
        # Gray window already closed by show_debug, nothing extra needed
        pass

    return chosen


# --- OBJECT CONTOUR EXTRACTION ----------------------------------------------

def extract_object_contour(a4_image, manual_threshold=None, debug=False):
    """
    On the top-down A4 image, find the main object contour.
    If manual_threshold is given, use that; otherwise use Otsu.
    Assumes object is darker than the white paper.
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if manual_threshold is None:
        # Automatic threshold (Otsu)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # Manual threshold from slider
        _, thresh = cv2.threshold(
            blur, manual_threshold, 255, cv2.THRESH_BINARY_INV
        )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if debug:
        show_debug("DEBUG: 05_thresh_raw", thresh)
        show_debug("DEBUG: 06_thresh_clean", thresh_clean)

    if not contours:
        raise RuntimeError("No contours found on A4 paper; is the object visible?")

    largest = max(contours, key=cv2.contourArea)

    if debug:
        # Contour on threshold image
        thresh_color = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(thresh_color, [largest], -1, (0, 0, 255), 2)
        show_debug("DEBUG: 07_thresh_with_largest_contour", thresh_color)

    # Approximate to reduce number of points
    peri = cv2.arcLength(largest, True)
    epsilon = 0.01 * peri
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if debug:
        a4_vis = a4_image.copy()
        cv2.drawContours(a4_vis, [approx], -1, (0, 0, 255), 2)
        show_debug("DEBUG: 08_a4_with_contour", a4_vis)

    return approx.reshape(-1, 2)


# --- COORDINATE CONVERSION + DXF --------------------------------------------

def contour_to_mm(contour_points, flip_y=True):
    """
    Convert contour points (in pixels) to millimeters.
    If flip_y=True, invert Y so DXF has an upwards Y axis.
    """
    mm_per_pixel = 1.0 / PIXELS_PER_MM
    pts_mm = contour_points.astype("float32") * mm_per_pixel

    if flip_y:
        pts_mm[:, 1] = A4_HEIGHT_MM - pts_mm[:, 1]

    return pts_mm


def save_contour_as_dxf(points_mm, output_path):
    """
    Save the given Nx2 array of points (in mm) as a closed polyline in a DXF file.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    poly_points = [(float(x), float(y)) for x, y in points_mm]

    msp.add_lwpolyline(poly_points, close=True)
    doc.saveas(output_path)


# --- MAIN PIPELINE ----------------------------------------------------------

def process_image_to_dxf(input_image_path, output_dxf_path, use_slider=True, debug=False):
    image = cv2.imread(input_image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_image_path}")

    print(f"[INFO] Image shape: {image.shape}")

    # 1. Find and warp the A4 paper
    paper_quad = find_paper_quad(image, debug=debug)
    print(f"[INFO] Paper quad: {paper_quad}")

    a4_topdown = warp_to_a4(image, paper_quad, debug=debug)
    print(f"[INFO] Warped A4 shape: {a4_topdown.shape}")

    # 2. Get threshold (slider or auto) and extract contour
    if use_slider:
        thr = interactive_threshold(a4_topdown, debug=debug)
        print(f"[INFO] Using manual threshold: {thr}")
        object_contour_px = extract_object_contour(
            a4_topdown, manual_threshold=thr, debug=debug
        )
    else:
        print("[INFO] Using automatic Otsu threshold")
        object_contour_px = extract_object_contour(
            a4_topdown, manual_threshold=None, debug=debug
        )

    print(f"[INFO] Contour has {len(object_contour_px)} points")

    # 3. Convert to mm coordinates
    object_contour_mm = contour_to_mm(object_contour_px)

    # 4. Save as DXF
    save_contour_as_dxf(object_contour_mm, output_dxf_path)


# --- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert photo of object on A4 paper to DXF outline."
    )
    parser.add_argument("input_image", help="Path to the input photo (jpg/png)")
    parser.add_argument("output_dxf", help="Path to the output DXF file")
    parser.add_argument(
        "--no-slider",
        action="store_true",
        help="Disable interactive threshold slider and use automatic threshold instead.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug windows for each processing step.",
    )

    args = parser.parse_args()

    process_image_to_dxf(
        args.input_image,
        args.output_dxf,
        use_slider=not args.no_slider,
        debug=args.debug,
    )
    print(f"[INFO] DXF saved to: {args.output_dxf}")


if __name__ == "__main__":
    main()
