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
    Interactive A4 sheet detection with multiple methods + manual corner editing.

    Trackbars:
      - mode:
          0: bright region (paper by brightness)
          1: edge mode (Canny)
          2: combined (bright & edge)
      - thresh: threshold / Canny high threshold (0–255)
      - morph: morphology strength (kernel size)
      - aspect: aspect-ratio weight (0–100); 0 = ignore A4 aspect, high = enforce

    Mouse (in interactive window):
      - LEFT CLICK on the RIGHT (original) pane:
          * first 4 clicks define the 4 corners
          * further clicks move the nearest existing corner
      - Corners are auto-ordered (top-left, top-right, bottom-right, bottom-left)
      - R / r -> reset manual corners

    Keys:
      - Enter / q / Esc -> accept current quad
          * if 4 manual corners exist: use them
          * else: use best automatic quad
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_h, img_w = gray.shape
    img_area = img_w * img_h
    target_aspect = max(A4_HEIGHT_MM, A4_WIDTH_MM) / min(A4_HEIGHT_MM, A4_WIDTH_MM)

    window_name = "Paper detect (mode, thresh, morph, aspect) – click corners, R reset, Enter/q/Esc"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    # mode 0..2
    cv2.createTrackbar("mode",   window_name, 0,   2,   nothing)
    cv2.createTrackbar("thresh", window_name, 180, 255, nothing)
    cv2.createTrackbar("morph",  window_name, 3,   15,  nothing)
    cv2.createTrackbar("aspect", window_name, 50,  100, nothing)  # aspect weight %

    # Manual corner editing state
    manual_corners = []  # list of [x, y] in original image coords
    # Info shared with mouse callback
    display_info = {
        "scale": 1.0,
        "mask_w": img_w,
        "mask_h": img_h,
    }

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_corners
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map from display coords -> combined coords -> original image coords
        scale = max(display_info["scale"], 1e-6)
        cx = int(x / scale)
        cy = int(y / scale)

        mask_w = display_info["mask_w"]
        mask_h = display_info["mask_h"]

        # Combined image is [mask | original_with_quad]
        if cx < 0 or cy < 0 or cy >= mask_h or cx >= 2 * mask_w:
            return

        # Only accept clicks in right pane (original)
        if cx < mask_w:
            return

        px = cx - mask_w
        py = cy

        # First 4 clicks: add corners
        if len(manual_corners) < 4:
            manual_corners.append([px, py])
        else:
            # More than 4 clicks: move nearest existing corner
            pts = np.array(manual_corners, dtype=np.float32)
            dists = np.sum((pts - np.array([px, py], dtype=np.float32)) ** 2, axis=1)
            idx = int(np.argmin(dists))
            manual_corners[idx] = [px, py]

    cv2.setMouseCallback(window_name, on_mouse)

    chosen_auto_quad = None
    final_vis = None

    while True:
        mode   = cv2.getTrackbarPos("mode",   window_name)
        t      = cv2.getTrackbarPos("thresh", window_name)
        morph  = cv2.getTrackbarPos("morph",  window_name)
        aspect = cv2.getTrackbarPos("aspect", window_name)

        morph = max(1, morph)
        aspect_weight = aspect / 100.0  # 0.0–1.0

        # --- Build base mask depending on mode ------------------------------
        # Base bright mask (paper bright)
        _, bright_mask = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
        ksize = 2 * morph + 1
        kernel = np.ones((ksize, ksize), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Edge mask (Canny)
        high = max(10, t)
        low = high // 2
        edges = cv2.Canny(blur, low, high)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)

        if mode == 0:
            mask = bright_mask
        elif mode == 1:
            mask = edge_mask
        else:
            # Combined: bright & edges
            mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(edge_mask))
            # if that fails (too strict), you can change to bitwise_or

        # --- Automatic quad from mask (if no full manual override) ----------
        best_quad = None
        if len(manual_corners) < 4:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_score = -1.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.1 * img_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) != 4:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    approx = box.reshape(-1, 1, 2).astype(np.int32)

                x, y, w, h = cv2.boundingRect(approx)
                if w == 0 or h == 0:
                    continue
                ratio = max(w, h) / min(w, h)
                aspect_penalty = abs(ratio - target_aspect)

                area_norm = area / float(img_area)
                score = area_norm - aspect_weight * aspect_penalty

                if score > best_score:
                    best_score = score
                    best_quad = approx

            if best_quad is not None:
                chosen_auto_quad = best_quad.reshape(4, 2).astype("float32")

        # --- Build visualisation -------------------------------------------
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_vis,
            "mode:0 bright,1 edge,2 combo | Click 4 corners in right pane, R reset",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        contour_vis = image.copy()

        # Decide which quad to display: manual first, otherwise auto
        if len(manual_corners) == 4:
            manual_pts = np.array(manual_corners, dtype=np.float32)
            quad = order_points(manual_pts)
            quad_int = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(contour_vis, [quad_int], -1, (0, 0, 255), 3)

            # Draw corner markers
            for (x, y) in quad:
                cv2.circle(contour_vis, (int(x), int(y)), 6, (0, 255, 0), -1)
        elif chosen_auto_quad is not None:
            quad = chosen_auto_quad
            quad_int = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(contour_vis, [quad_int], -1, (0, 0, 255), 3)
        else:
            quad = None  # nothing found yet

        # Also show manual corner clicks if <4
        if 0 < len(manual_corners) < 4:
            for (x, y) in manual_corners:
                cv2.circle(contour_vis, (int(x), int(y)), 6, (0, 255, 255), -1)

        combined = np.hstack((mask_vis, contour_vis))

        # Scale to fit window
        ch, cw = combined.shape[:2]
        scale = min(MAX_DEBUG_WIDTH / cw, MAX_DEBUG_HEIGHT / ch, 1.0)
        if scale < 1.0:
            disp = cv2.resize(combined, (int(cw * scale), int(ch * scale)))
        else:
            disp = combined

        display_info["scale"] = scale  # for mouse mapping

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(30) & 0xFF

        if key in (13, 27, ord("q")):  # Enter, Esc, q -> accept
            break

        if key in (ord("r"), ord("R")):
            manual_corners = []

    cv2.destroyWindow(window_name)

    # Decide final quad to return
    if len(manual_corners) == 4:
        manual_pts = np.array(manual_corners, dtype=np.float32)
        quad = order_points(manual_pts)
        if debug:
            print("[INFO] Using manual paper corners:", quad)
        return quad.astype("float32")

    if chosen_auto_quad is not None:
        if debug:
            print("[INFO] Using automatic paper quad:", chosen_auto_quad)
        return chosen_auto_quad.astype("float32")

    raise RuntimeError("Could not determine paper quad (no manual or automatic fit).")



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

def interactive_multi_tool_contours(a4_image, debug=False):
    """
    Multi-tool interactive object detector.

    Workflow:
      - Global sliders define a base mask of ALL tools.
      - You click on a tool in the RIGHT pane; the program finds the contour
        that contains that click and creates/activates a "tool".
      - Each tool has its own 'offset' (contour grown around it).
      - You can have multiple tools, switch between them, and each gets its own
        final contour that will be exported to DXF.

    Trackbars:
      - mode:
          0: dark objects on light paper    (binary INV)
          1: light objects on dark paper    (binary)
          2: edge mode (Canny)
      - thresh: threshold / Canny high threshold (0–255)
      - morph: morphology strength (kernel size)
      - eps_x10000: contour detail -> epsilon = eps_x10000 / 10000 * perimeter
      - offset: offset radius in pixels (dilates the tool mask)

    Mouse:
      - LEFT CLICK in RIGHT pane (original image):
          * if no tools OR 'new tool' mode is active:
               -> create a new tool centered at click
          * otherwise:
               -> switch active tool to the one whose center is nearest the click

    Keys:
      - n / N : "new tool" mode -> next click creates a new tool
      - 1..9  : switch active tool (1 = tool 1, etc., if it exists)
      - r / R : remove ACTIVE tool
      - Enter / q / Esc : finish and return contours for all tools
    """
    gray = cv2.cvtColor(a4_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    mask_h, mask_w = gray.shape

    window_name = "Multi-tool detect (mode/thresh/morph/eps/offset) – n=new, r=remove, Enter=q/Esc=done"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, MAX_DEBUG_WIDTH, MAX_DEBUG_HEIGHT)

    def nothing(x):
        pass

    cv2.createTrackbar("mode",        window_name, 0,   2,   nothing)
    cv2.createTrackbar("thresh",      window_name, 120, 255, nothing)
    cv2.createTrackbar("morph",       window_name, 2,   10,  nothing)
    cv2.createTrackbar("eps_x10000",  window_name, 50,  500, nothing)  # finer control
    cv2.createTrackbar("offset",      window_name, 0,   40,  nothing)  # offset radius (px)

    # tools: list of dicts
    # {
    #   "center": (x, y),
    #   "offset": int,
    #   "contour_px": Nx2 array (updated every frame)
    # }
    tools = []
    active_tool_idx = -1
    new_tool_mode = False

    # For mapping clicks from display -> image coordinates
    display_info = {
        "scale": 1.0,
        "mask_w": mask_w,
        "mask_h": mask_h,
    }

    def set_active_tool(idx):
        nonlocal active_tool_idx
        if 0 <= idx < len(tools):
            active_tool_idx = idx
            # Update offset slider to this tool's offset
            cv2.setTrackbarPos("offset", window_name, tools[idx]["offset"])

    def add_tool_at(px, py):
        nonlocal tools, active_tool_idx
        # Clamp to image bounds
        px = int(np.clip(px, 0, mask_w - 1))
        py = int(np.clip(py, 0, mask_h - 1))
        tool = {
            "center": (px, py),
            "offset": cv2.getTrackbarPos("offset", window_name),
            "contour_px": None,
        }
        tools.append(tool)
        set_active_tool(len(tools) - 1)

    def on_mouse(event, x, y, flags, param):
        nonlocal new_tool_mode, tools, active_tool_idx

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Map from display coords -> combined coords -> image coords
        scale = max(display_info["scale"], 1e-6)
        cx = int(x / scale)
        cy = int(y / scale)

        mask_w = display_info["mask_w"]
        mask_h = display_info["mask_h"]

        # Combined image is [mask | original]; width = 2 * mask_w
        if cx < 0 or cy < 0 or cy >= mask_h or cx >= 2 * mask_w:
            return

        # Only respond to clicks in the RIGHT pane (original image)
        if cx < mask_w:
            return

        px = cx - mask_w
        py = cy

        if new_tool_mode or len(tools) == 0:
            # Create a new tool centered at this point
            add_tool_at(px, py)
            new_tool_mode = False
        else:
            # Select nearest existing tool center to this click
            if not tools:
                return
            click_pt = np.array([px, py], dtype=np.float32)
            centers = np.array([t["center"] for t in tools], dtype=np.float32)
            dists = np.sum((centers - click_pt) ** 2, axis=1)
            idx = int(np.argmin(dists))
            set_active_tool(idx)

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        mode  = cv2.getTrackbarPos("mode",       window_name)
        t     = cv2.getTrackbarPos("thresh",     window_name)
        morph = cv2.getTrackbarPos("morph",      window_name)
        epsk  = cv2.getTrackbarPos("eps_x10000", window_name)
        off_slider = cv2.getTrackbarPos("offset", window_name)

        morph = max(1, morph)
        epsk  = max(1, epsk)

        # If there's an active tool, update its offset from slider
        if 0 <= active_tool_idx < len(tools):
            tools[active_tool_idx]["offset"] = off_slider

        # --- Build base mask for all tools ----------------------------------
        if mode in (0, 1):
            if mode == 0:
                # dark objects on light paper
                _, mask_all = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
            else:
                # light objects on dark paper
                _, mask_all = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

            ksize = 2 * morph + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # edge mode
            high = max(10, t)
            low = high // 2
            edges = cv2.Canny(blur, low, high)

            ksize = 2 * morph + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            mask_all = cv2.dilate(edges, kernel, iterations=1)

        # Find all contours on full mask
        all_contours, _ = cv2.findContours(
            mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # --- Per-tool contour computation -----------------------------------
        active_mask_tool = None  # binary mask of active tool (for left pane)

        for ti, tool in enumerate(tools):
            cx, cy = tool["center"]
            best_cnt = None
            best_dist = 1e12

            # 1) try contours that actually contain the center
            for cnt in all_contours:
                inside = cv2.pointPolygonTest(cnt, (cx, cy), False)
                if inside >= 0:
                    # prefer the one with largest area
                    area = cv2.contourArea(cnt)
                    d = -area  # bigger area = "closer"
                    if d < best_dist:
                        best_dist = d
                        best_cnt = cnt

            # 2) if none contain the point, pick closest centroid
            if best_cnt is None and all_contours:
                for cnt in all_contours:
                    M = cv2.moments(cnt)
                    if M["m00"] <= 0:
                        continue
                    cx_cnt = M["m10"] / M["m00"]
                    cy_cnt = M["m01"] / M["m00"]
                    d = (cx - cx_cnt) ** 2 + (cy - cy_cnt) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_cnt = cnt

            if best_cnt is None:
                tool["contour_px"] = None
                continue

            # Build tool-specific mask
            tool_mask = np.zeros_like(mask_all)
            cv2.drawContours(tool_mask, [best_cnt], -1, 255, -1)

            # Apply offset (dilate)
            off_px = int(tool["offset"])
            if off_px > 0:
                ksize = 2 * off_px + 1
                kernel_off = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                tool_mask = cv2.dilate(tool_mask, kernel_off, iterations=1)

            # Extract contour from offset mask
            tool_cnts, _ = cv2.findContours(
                tool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not tool_cnts:
                tool["contour_px"] = None
                continue

            largest = max(tool_cnts, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            epsilon = (epsk / 10000.0) * peri
            approx = cv2.approxPolyDP(largest, epsilon, True)
            tool["contour_px"] = approx.reshape(-1, 2)

            if ti == active_tool_idx:
                active_mask_tool = tool_mask

        # --- Build visualisation -------------------------------------------
        # Left pane: show active tool mask if exists, else full mask
        if active_mask_tool is not None:
            left_vis = cv2.cvtColor(active_mask_tool, cv2.COLOR_GRAY2BGR)
        else:
            left_vis = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            left_vis,
            "Click tool in right pane. n=new, r=remove, 1-9 select",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Right pane: original image with all tool contours
        right_vis = a4_image.copy()
        for idx, tool in enumerate(tools):
            color = (0, 255, 0)  # other tools = green
            if idx == active_tool_idx:
                color = (0, 0, 255)  # active tool = red

            if tool["contour_px"] is not None:
                cc = tool["contour_px"].astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(right_vis, [cc], -1, color, 2)

            # center marker
            cx, cy = tool["center"]
            cv2.circle(right_vis, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(
                right_vis,
                f"{idx+1}",
                (int(cx) + 6, int(cy) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            right_vis,
            f"Tools: {len(tools)}  Active: {active_tool_idx+1 if active_tool_idx>=0 else '-'}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined = np.hstack((left_vis, right_vis))

        ch, cw = combined.shape[:2]
        scale = min(MAX_DEBUG_WIDTH / cw, MAX_DEBUG_HEIGHT / ch, 1.0)
        if scale < 1.0:
            disp = cv2.resize(combined, (int(cw * scale), int(ch * scale)))
        else:
            disp = combined

        display_info["scale"] = scale  # for mouse mapping

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(30) & 0xFF

        # Finish
        if key in (13, 27, ord("q")):  # Enter, Esc, q
            break

        # New tool mode
        if key in (ord("n"), ord("N")):
            new_tool_mode = True

        # Remove active tool
        if key in (ord("r"), ord("R")):
            if 0 <= active_tool_idx < len(tools):
                tools.pop(active_tool_idx)
                if not tools:
                    active_tool_idx = -1
                else:
                    active_tool_idx = min(active_tool_idx, len(tools) - 1)
                    cv2.setTrackbarPos(
                        "offset", window_name,
                        tools[active_tool_idx]["offset"]
                    )

        # Number keys 1..9 to select tool
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(tools):
                set_active_tool(idx)

    cv2.destroyWindow(window_name)

    # Collect final contours
    contours_px = [
        t["contour_px"] for t in tools if t.get("contour_px") is not None
    ]

    if not contours_px:
        raise RuntimeError("No valid tool contours selected.")

    if debug:
        print(f"[INFO] Multi-tool: {len(contours_px)} contours")

    return contours_px


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


def save_contours_as_dxf(list_of_points_mm, output_path):
    """
    Save multiple tool outlines (each Nx2 in mm) into a single DXF file.
    Each tool becomes a polyline; optionally put them on separate layers.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for i, pts in enumerate(list_of_points_mm, start=1):
        poly_points = [(float(x), float(y)) for x, y in pts]
        layer_name = f"TOOL_{i}"
        if layer_name not in doc.layers:
            doc.layers.add(name=layer_name)
        msp.add_lwpolyline(poly_points, close=True, dxfattribs={"layer": layer_name})

    doc.saveas(output_path)


# --- MAIN PIPELINE ----------------------------------------------------------

def process_image_to_dxf(input_image_path, output_dxf_path, use_slider=True, debug=False):
    image = cv2.imread(input_image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_image_path}")

    print(f"[INFO] Image shape: {image.shape}")

    # 1. Find and warp the A4 paper (using your interactive find_paper_quad)
    paper_quad = find_paper_quad(image, debug=debug)
    print(f"[INFO] Paper quad: {paper_quad}")

    a4_topdown = warp_to_a4(image, paper_quad, debug=debug)
    print(f"[INFO] Warped A4 shape: {a4_topdown.shape}")

    # 2. Multi-tool object contour editor
    print("[INFO] Multi-tool editor: click each tool on the RIGHT pane, n=new tool")
    tool_contours_px = interactive_multi_tool_contours(a4_topdown, debug=debug)

    print(f"[INFO] Got {len(tool_contours_px)} tool contours")

    # 3. Convert all to mm
    tool_contours_mm = [contour_to_mm(c) for c in tool_contours_px]

    # 4. Save to DXF
    save_contours_as_dxf(tool_contours_mm, output_dxf_path)



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
