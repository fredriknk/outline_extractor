import argparse
import cv2
import numpy as np
import ezdxf

# --- CONFIG ------------------------------------------------------------------

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# Working resolution for the warped A4 (pixels per mm)
PIXELS_PER_MM = 4

# Max size of debug / interactive windows on screen
MAX_DEBUG_WIDTH = 1800
MAX_DEBUG_HEIGHT = 1200


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
    Detect the A4 sheet as the biggest bright rectangle, using an interactive
    threshold slider.

    Each slider move:
      - thresholds the image (paper = white)
      - does morphology to solidify the paper
      - finds the largest A4-ish contour
      - shows [binary mask | original with red quad]

    When you press Enter / q / Esc, the last valid quad is returned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Start value from Otsu, but allow manual adjustment
    _, otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    start_t = 119#int(np.mean(blur[otsu == 255])) if np.any(otsu == 255) else 119
    start_t = max(0, min(255, start_t))

    window_name = "Paper threshold (Enter/q/Esc to accept)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    cv2.createTrackbar("thresh", window_name, start_t, 255, nothing)

    img_h, img_w = gray.shape
    img_area = img_w * img_h
    target_aspect = max(A4_HEIGHT_MM, A4_WIDTH_MM) / min(A4_HEIGHT_MM, A4_WIDTH_MM)

    chosen_quad = None
    chosen_t = start_t
    final_vis = None

    while True:
        t = cv2.getTrackbarPos("thresh", window_name)

        # Paper is bright → THRESH_BINARY (paper = white)
        _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

        # Morphological closing to solidify the paper region
        kernel = np.ones((7, 7), np.uint8)
        thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find external contours on the bright mask
        contours, _ = cv2.findContours(
            thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_quad = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.1 * img_area:
                # Too small to be the paper
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # If not 4 points, use min-area rectangle
            if len(approx) != 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                approx = box.reshape(-1, 1, 2).astype(np.int32)

            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue
            ratio = max(w, h) / min(w, h)

            aspect_penalty = abs(ratio - target_aspect)
            score = (area / img_area) - 0.5 * aspect_penalty

            if score > best_score:
                best_score = score
                best_quad = approx

        # Visualisation
        thresh_vis = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)
        contour_vis = image.copy()

        if best_quad is not None:
            cv2.drawContours(contour_vis, [best_quad], -1, (0, 0, 255), 3)
            chosen_quad = best_quad.reshape(4, 2).astype("float32")
            chosen_t = t
            final_vis = contour_vis

        combined = np.hstack((thresh_vis, contour_vis))

        # Scale for display
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

    if chosen_quad is None:
        raise RuntimeError("Could not find A4 paper contour with chosen threshold.")

    if debug and final_vis is not None:
        show_debug("DEBUG: 02_original_with_paper_quad", final_vis)

    print(f"[INFO] Final paper threshold: {chosen_t}")
    return chosen_quad


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


# --- INTERACTIVE OBJECT CONTOUR (ADVANCED SLIDERS) --------------------------
def interactive_object_contour(a4_image, debug=False):
    """
    Advanced interactive object detection with paint tools:

    Trackbars:
      - mode:
          0: dark object on light paper (binary INV)
          1: light object on dark background (binary)
          2: edge mode (Canny)
      - thresh: threshold (0–255) or Canny high threshold
      - morph: morphology strength (kernel size)
      - eps_x10000: contour approximation factor
            epsilon = eps_x10000 / 10000 * perimeter
      - brush: brush radius in pixels for painting
      - zoom: zoom factor * 0.1 (1.0–4.0), zooms around last clicked point

    Mouse:
      - Left click + drag  on LEFT (mask) pane  -> erase (remove from object)
      - Right click + drag on LEFT (mask) pane  -> add   (add to object)
      - Any click also recenters zoom on that point

    Keys:
      - Z or Ctrl+Z -> undo last paint stroke
      - R           -> reset all painting
      - Enter / q / Esc -> accept contour and return it
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    mask_h, mask_w = gray.shape

    # Masks for user painting
    removal_mask = np.zeros_like(gray, dtype=np.uint8)  # 255 = forced remove
    add_mask = np.zeros_like(gray, dtype=np.uint8)      # 255 = forced add

    # History for undo (stack of (removal_mask, add_mask))
    history = []

    window_name = "Object detect – paint (LMB erase, RMB add, Z undo, R reset, Enter/q/Esc)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    cv2.createTrackbar("mode",        window_name, 2, 2,   nothing)
    cv2.createTrackbar("thresh",      window_name, 120, 255, nothing)
    cv2.createTrackbar("morph",       window_name, 2, 10,  nothing)
    cv2.createTrackbar("eps_x10000",  window_name, 5, 50, nothing)  # finer control
    cv2.createTrackbar("brush",       window_name, 15, 80, nothing)   # brush radius
    cv2.createTrackbar("zoom",        window_name, 10, 40, nothing)   # 10→1.0, 40→4.0

    # Info shared with mouse callback
    display_info = {
        "scale": 1.0,
        "roi_x": 0,
        "roi_y": 0,
        "mask_w": mask_w,
        "mask_h": mask_h,
        "zoom_cx": mask_w,      # start near center of mask pane
        "zoom_cy": mask_h // 2,
    }

    def push_history():
        # Keep at most ~30 steps
        if len(history) >= 30:
            history.pop(0)
        history.append((removal_mask.copy(), add_mask.copy()))

    def on_mouse(event, x, y, flags, param):
        nonlocal removal_mask, add_mask, history
        # Map window coords -> combined image coords -> mask coords
        scale = max(display_info["scale"], 1e-6)
        rx = int(x / scale)
        ry = int(y / scale)
        gx = display_info["roi_x"] + rx
        gy = display_info["roi_y"] + ry

        mask_w = display_info["mask_w"]
        mask_h = display_info["mask_h"]

        # Combined image is [mask | contour] → width = 2*mask_w
        if gx < 0 or gy < 0 or gy >= mask_h or gx >= 2 * mask_w:
            return

        # Only paint in left (mask) pane
        if gx >= mask_w:
            return

        mx, my = gx, gy

        # Update zoom center on any button down
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            display_info["zoom_cx"] = mx
            display_info["zoom_cy"] = my

        brush_r = cv2.getTrackbarPos("brush", window_name)
        brush_r = max(1, brush_r)

        # Start stroke -> push history
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            push_history()

        # Continuous painting while dragging
        if (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)) or \
           event == cv2.EVENT_LBUTTONDOWN:
            # Erase: mark removal_mask, clear add_mask there
            cv2.circle(removal_mask, (mx, my), brush_r, 255, -1)
            cv2.circle(add_mask, (mx, my), brush_r, 0,   -1)

        elif (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON)) or \
             event == cv2.EVENT_RBUTTONDOWN:
            # Add: mark add_mask, clear removal_mask there
            cv2.circle(add_mask, (mx, my), brush_r, 255, -1)
            cv2.circle(removal_mask, (mx, my), brush_r, 0,   -1)

    cv2.setMouseCallback(window_name, on_mouse)

    chosen_contour = None
    final_vis = None

    while True:
        mode  = cv2.getTrackbarPos("mode",       window_name)
        t     = cv2.getTrackbarPos("thresh",     window_name)
        morph = cv2.getTrackbarPos("morph",      window_name)
        epsk  = cv2.getTrackbarPos("eps_x10000", window_name)
        zooms = cv2.getTrackbarPos("zoom",       window_name)

        morph = max(1, morph)
        epsk  = max(1, epsk)
        zoom_factor = max(1.0, zooms / 10.0)  # 10→1.0, 40→4.0

        # --- Build base mask from sliders ------------------------------------
        if mode in (0, 1):
            if mode == 0:
                # dark object on light background
                _, mask = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
            else:
                # light object on dark background
                _, mask = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

            ksize = 2 * morph + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # Canny edge mode
            high = max(10, t)
            low = high // 2
            edges = cv2.Canny(blur, low, high)

            ksize = 2 * morph + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=1)

        # Apply user painting
        mask[removal_mask > 0] = 0
        mask[add_mask > 0] = 255

        # --- Contour detection ----------------------------------------------
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_vis = a4_image.copy()
        if contours:
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            epsilon = (epsk / 10000.0) * peri
            approx = cv2.approxPolyDP(largest, epsilon, True)

            cv2.drawContours(contour_vis, [approx], -1, (0, 0, 255), 2)

            chosen_contour = approx.reshape(-1, 2)
            final_vis = contour_vis
        elif chosen_contour is not None:
            # No contour this frame, keep last good one visible
            cc = chosen_contour.reshape(-1, 1, 2)
            cv2.drawContours(contour_vis, [cc], -1, (0, 0, 255), 2)

        # --- Build side-by-side & apply zoom --------------------------------
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_vis,
            "LMB erase, RMB add, Z undo, R reset",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        combined = np.hstack((mask_vis, contour_vis))
        ch, cw = combined.shape[:2]

        # Determine ROI for zoom
        cx = display_info["zoom_cx"]
        cy = display_info["zoom_cy"]
        cx = int(np.clip(cx, 0, 2 * mask_w - 1))
        cy = int(np.clip(cy, 0, mask_h - 1))

        view_w = int(cw / zoom_factor)
        view_h = int(ch / zoom_factor)
        view_w = max(1, min(cw, view_w))
        view_h = max(1, min(ch, view_h))

        x0 = int(cx - view_w // 2)
        y0 = int(cy - view_h // 2)
        x0 = max(0, min(cw - view_w, x0))
        y0 = max(0, min(ch - view_h, y0))

        roi = combined[y0:y0 + view_h, x0:x0 + view_w]

        # Scale ROI to window
        scale = min(MAX_DEBUG_WIDTH / view_w, MAX_DEBUG_HEIGHT / view_h, 1.0)
        if scale < 1.0:
            disp = cv2.resize(roi, (int(view_w * scale), int(view_h * scale)))
        else:
            disp = roi

        # Update mapping for mouse callback
        display_info["scale"] = scale
        display_info["roi_x"] = x0
        display_info["roi_y"] = y0

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(30) & 0xFF

        # Accept
        if key in (13, 27, ord("q")):  # Enter, Esc, q
            break

        # Undo: Z or Ctrl+Z (26)
        if key in (ord("z"), ord("Z"), 26):
            if history:
                removal_mask, add_mask = history.pop()

        # Reset painting
        if key in (ord("r"), ord("R")):
            removal_mask[:] = 0
            add_mask[:] = 0
            history.clear()

    cv2.destroyWindow(window_name)

    if chosen_contour is None:
        raise RuntimeError("No contour found with chosen settings")

    if debug and final_vis is not None:
        show_debug("DEBUG: final_object_contour", final_vis)

    return chosen_contour




def auto_object_contour(a4_image, debug=False):
    """
    Simple automatic object contour detection (Otsu threshold), no sliders.
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

    # 1. Find and warp the A4 paper (with paper slider)
    paper_quad = find_paper_quad(image, debug=debug)
    print(f"[INFO] Paper quad: {paper_quad}")

    a4_topdown = warp_to_a4(image, paper_quad, debug=debug)
    print(f"[INFO] Warped A4 shape: {a4_topdown.shape}")

    # 2. Object contour (advanced slider or auto)
    if use_slider:
        print("[INFO] Using advanced interactive object detection")
        object_contour_px = interactive_object_contour(a4_topdown, debug=debug)
    else:
        print("[INFO] Using automatic object detection (Otsu)")
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
        help="Disable interactive object slider and use automatic detection instead.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug windows for intermediate steps.",
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
