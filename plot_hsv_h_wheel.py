import argparse
from pathlib import Path

import cv2
import numpy as np


def build_hsv_hue_ring(size: int = 800, ring_width: int = 120) -> np.ndarray:
    """Create a BGR image of an HSV hue ring where H spans 0..180."""
    h, w = size, size
    cx, cy = w // 2, h // 2
    yy, xx = np.mgrid[0:h, 0:w]

    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy)

    outer_r = min(cx, cy) - 10
    inner_r = max(outer_r - ring_width, 1)
    ring_mask = (radius >= inner_r) & (radius <= outer_r)

    # Angle: 0..360, then map to OpenCV HSV hue range 0..180.
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    hue_float = angle / 2.0
    hue = np.clip(hue_float, 0, 179).astype(np.uint8)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 0
    hsv[..., 2] = 255

    hsv[..., 0][ring_mask] = hue[ring_mask]
    hsv[..., 1][ring_mask] = 255
    hsv[..., 2][ring_mask] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Draw boundaries for readability.
    cv2.circle(bgr, (cx, cy), int(outer_r), (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(bgr, (cx, cy), int(inner_r), (30, 30, 30), 1, cv2.LINE_AA)

    # Draw ticks and labels for H values in OpenCV range [0, 180].
    font = cv2.FONT_HERSHEY_SIMPLEX
    for h_tick in range(0, 181, 10):
        # H in [0, 180] maps to angle in [0, 360].
        angle_deg = h_tick * 2.0
        angle_rad = np.deg2rad(angle_deg)

        is_major = (h_tick % 30 == 0) or (h_tick == 180)
        tick_len = 18 if is_major else 10
        tick_color = (10, 10, 10) if is_major else (50, 50, 50)
        tick_thickness = 2 if is_major else 1

        r1 = outer_r + 2
        r2 = outer_r + 2 + tick_len
        x1 = int(cx + r1 * np.cos(angle_rad))
        y1 = int(cy + r1 * np.sin(angle_rad))
        x2 = int(cx + r2 * np.cos(angle_rad))
        y2 = int(cy + r2 * np.sin(angle_rad))
        cv2.line(bgr, (x1, y1), (x2, y2), tick_color, tick_thickness, cv2.LINE_AA)

        if is_major:
            text = str(h_tick)
            text_scale = 0.6
            text_thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, text_scale, text_thickness)
            r_text = outer_r + 34
            tx = int(cx + r_text * np.cos(angle_rad) - tw / 2)
            ty = int(cy + r_text * np.sin(angle_rad) + th / 2)
            cv2.putText(
                bgr,
                text,
                (tx, ty),
                font,
                text_scale,
                (20, 20, 20),
                text_thickness,
                cv2.LINE_AA,
            )

    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw HSV hue ring (H: 0..180)")
    parser.add_argument("--size", type=int, default=800, help="Output image size (square)")
    parser.add_argument("--ring-width", type=int, default=120, help="Ring thickness in pixels")
    parser.add_argument(
        "--output",
        type=str,
        default="show/hsv_h_0_180_ring.png",
        help="Output image path",
    )
    args = parser.parse_args()

    image = build_hsv_hue_ring(size=args.size, ring_width=args.ring_width)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved hue ring to: {output_path}")


if __name__ == "__main__":
    main()
