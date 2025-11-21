import argparse
import cv2
import numpy as np
import ezdxf

# --- CONFIG ------------------------------------------------------------------

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# Lower value = smaller warped image (better for screens)
PIXELS_PER_MM = 10

# Max size of debug windows on screen
MAX_DEBUG_WIDTH = 1200
MAX_DEBUG_HEIGHT = 800


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
    Detect the A4 sheet as the biggest bright rectangle.

    Steps:
      - grayscale + blur
      - Otsu threshold to get bright regions (paper)
      - cleanup with morphology
      - largest contour with A4-ish aspect ratio
    Returns 4 points (float32).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Paper is the brightest thing: THRESH_BINARY so paper becomes white (255)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological closing to solidify the paper region
    kernel = np.ones((7, 7), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        show_debug("DEBUG: 00_gray", gray)
        show_debug("DEBUG: 01_thresh_paper", thresh_clean)

    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contours found for paper")

    img_h, img_w = gray.shape
    img_area = img_w * img_h
    target_aspect = max(A4_HEIGHT_MM, A4_WIDTH_MM) / min(A4_HEIGHT_MM, A4_WIDTH_MM)

    best_quad = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.1 * img_area:
            # Too small to be the paper
            continue

        # Approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # If not 4 points, get a minimum-area rectangle instead
        if len(approx) != 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.int32)

        # Bounding box to compute aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / min(w, h)

        # Score: big area and aspect ratio near A4
        aspect_penalty = abs(ratio - target_aspect)
        score = (area / img_area) - 0.5 * aspect_penalty

        if score > best_score:
            best_score = score
            best_quad = approx

    if best_quad is None:
        raise RuntimeError("Could not find A4 paper contour.")

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


# --- INTERACTIVE OBJECT CONTOUR (SLIDER) ------------------------------------

def interactive_object_contour(a4_image, debug=False):
    """
    Use a slider to control the threshold used for object detection.
    Each slider move:
      - thresholds
      - cleans noise
      - finds largest contour
      - shows [binary mask | contour on A4]
    When user presses Enter / q / Esc, returns that contour (in pixels).
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    window_name = "Object threshold (Enter/q/Esc to accept)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    cv2.createTrackbar("thresh", window_name, 127, 255, nothing)

    chosen_t = 127
    chosen_contour = None
    final_contour_vis = None

    while True:
        t = cv2.getTrackbarPos("thresh", window_name)

        # Threshold & clean
        _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=10)

        # Find contours
        contours, _ = cv2.findContours(
            thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_vis = a4_image.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # Approximate contour – decrease epsilon if you want more points
            peri = cv2.arcLength(largest, True)
            epsilon = 0.005 * peri  # finer than 0.01
            approx = cv2.approxPolyDP(largest, epsilon, True)
            cv2.drawContours(contour_vis, [approx], -1, (0, 0, 255), 2)

            chosen_contour = approx.reshape(-1, 2)
            chosen_t = t
            final_contour_vis = contour_vis

        # Build side-by-side view: [binary | contour overlay]
        thresh_vis = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((thresh_vis, contour_vis))

        # Scale for window
        h, w = combined.shape[:2]
        scale = min(MAX_DEBUG_WIDTH / w, MAX_DEBUG_HEIGHT / h, 1.0)
        if scale < 1.0:
            disp = cv2.resize(combined, (int(w * scale), int(h * scale)))
        else:
            disp = combined

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 27, ord("q")):  # Enter, Esc, q
            break

    cv2.destroyWindow(window_name)

    if chosen_contour is None:
        raise RuntimeError("No contour found with chosen threshold")

    if debug and final_contour_vis is not None:
        show_debug("DEBUG: 08_a4_with_final_contour", final_contour_vis)

    print(f"[INFO] Final manual threshold: {chosen_t}")
    return chosen_contour


def auto_object_contour(a4_image, debug=False):
    """
    Automatic object contour detection (Otsu threshold), no slider.
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
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

    peri = cv2.arcLength(largest, True)
    epsilon = 0.005 * peri
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

    # 2. Object contour (slider or auto)
    if use_slider:
        print("[INFO] Using interactive threshold/contour slider")
        object_contour_px = interactive_object_contour(a4_topdown, debug=debug)
    else:
        print("[INFO] Using automatic Otsu threshold")
        object_contour_px = auto_object_contour(a4_topdown, debug=debug)

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
        help="Disable interactive slider and use automatic threshold instead.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug windows for intermediate steps (paper detection & contour).",
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
