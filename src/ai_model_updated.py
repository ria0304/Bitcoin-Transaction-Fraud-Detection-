
# ======================
# AI GARMENT CLASSIFIER (CLIP)
# ======================
# Added to improve auto‑tagging accuracy and avoid everything being classified as "Dress".
# Requires: pip install torch torchvision transformers pillow

try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch

    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def classify_garment_with_clip(img_array):
        labels = [
            "t-shirt","shirt","dress","jacket","hoodie",
            "pants","jeans","shorts","skirt","sweater","coat"
        ]

        pil_image = Image.fromarray(img_array)

        inputs = _clip_processor(
            text=labels,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        outputs = _clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        idx = probs.argmax().item()
        return labels[idx]

except Exception as e:
    # If CLIP cannot load, fallback to None so the rest of the system still runs.
    def classify_garment_with_clip(img_array):
        return None


# ai_model.py - FULLY CORRECTED VERSION WITH ROBUST MASK EXTRACTION
import os
import json
import base64
import logging
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from datetime import datetime

# Import OpenCV and sklearn at module level
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available - image processing will be limited")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available - color clustering will use fallback")

# Import the new advanced matching engine
try:
    from ai_matcher import fashion_matcher
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("ai_matcher not found, using fallback matching")
    fashion_matcher = None

logger = logging.getLogger(__name__)


class LocalComputerVision:
    """
    Local Deep Learning & CV Engine.
    FIXED: Robust mask extraction, improved category detection, and proper variation handling.
    """

    def decode_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array with proper error handling."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available for image decoding")
            return np.zeros((256, 256, 3), dtype=np.uint8)

        try:
            # Handle data URI format
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            # Validate base64
            if not base64_str or len(base64_str) < 100:
                raise ValueError("Invalid base64 image data")

            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Image decode failed - cv2.imdecode returned None")

            # Validate image dimensions
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Image too small: {img.shape}")

            return img

        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def get_improved_mask(self, image: np.ndarray) -> np.ndarray:
        """
        ROBUST mask extraction using GrabCut with enhanced parameters.
        Returns binary mask (0 or 255) isolating the garment.
        """
        if not CV2_AVAILABLE:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ---- Generate initial candidate mask using multiple techniques ----
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Adaptive threshold
        thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)

        # Otsu threshold
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine
        combined = cv2.bitwise_or(edges_dilated, thresh_adapt)
        combined = cv2.bitwise_or(combined, thresh_otsu)

        # Clean up
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours, return a centered rectangle mask
        if not contours:
            logger.warning("No contours found in initial mask, using centered rectangle")
            mask = np.zeros_like(gray)
            cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 255, -1)
            return mask

        # Get the largest contour (assumed to be the garment)
        largest = max(contours, key=cv2.contourArea)
        x, y, wc, hc = cv2.boundingRect(largest)

        # Expand rectangle generously (30% on each side)
        expand_x = int(wc * 0.3)
        expand_y = int(hc * 0.3)
        x = max(0, x - expand_x)
        y = max(0, y - expand_y)
        wc = min(w - x, wc + 2 * expand_x)
        hc = min(h - y, hc + 2 * expand_y)

        # Ensure rectangle is at least 100x100
        if wc < 100 or hc < 100:
            logger.warning(f"Bounding rectangle too small ({wc}x{hc}), using centered fallback")
            mask = np.zeros_like(gray)
            cv2.rectangle(mask, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 255, -1)
            return mask

        rect = (x, y, wc, hc)

        # ---- GrabCut ----
        mask_gc = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            # Run GrabCut with the rectangle (more iterations for better convergence)
            cv2.grabCut(image, mask_gc, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

            # Check if we got a reasonable foreground
            fg_pixels = np.sum((mask_gc == 1) | (mask_gc == 3))
            if fg_pixels < 5000:  # Arbitrary threshold, adjust if needed
                logger.warning(f"GrabCut produced too few foreground pixels ({fg_pixels}), using fallback mask")
                # Fallback: fill the largest contour
                fallback_mask = np.zeros_like(gray)
                cv2.drawContours(fallback_mask, [largest], -1, 255, -1)
                return fallback_mask

            # Convert to binary mask
            mask_final = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')

            # Clean up small holes
            kernel = np.ones((5, 5), np.uint8)
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)

            return mask_final

        except Exception as e:
            logger.warning(f"GrabCut failed: {e}, using fallback mask")
            # Fallback: fill the largest contour
            fallback_mask = np.zeros_like(gray)
            cv2.drawContours(fallback_mask, [largest], -1, 255, -1)
            return fallback_mask

    def analyze_shape(self, image: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        FIX 2: IMPROVED SHAPE ANALYSIS with better distinction between skirts, dresses, bottoms.
        Returns (category, garment_mask)
        """
        if not CV2_AVAILABLE:
            return "Top", np.zeros((256, 256), dtype=np.uint8)

        H, W = image.shape[:2]

        # Get improved mask
        garment_mask = self.get_improved_mask(image)

        # Find largest contour from the mask
        contours, _ = cv2.findContours(garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Top", garment_mask

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Normalized values
        garment_top = y / H
        garment_bottom = (y + h) / H
        garment_height_ratio = h / H
        aspect_ratio = w / h

        # Compute upper and lower region occupancy
        top_half = garment_mask[y:y + h // 2, x:x + w]
        bottom_half = garment_mask[y + h // 2:y + h, x:x + w]

        top_fill = np.count_nonzero(top_half) / (top_half.size + 1e-5)
        bottom_fill = np.count_nonzero(bottom_half) / (bottom_half.size + 1e-5)

        # Detect waist narrowing (for skirts vs dresses)
        waist_narrow = False
        if h > 100:
            # Check width at 30% and 70% of height
            waist_y = y + int(h * 0.3)
            hip_y = y + int(h * 0.7)
            if waist_y < y + h and hip_y < y + h:
                waist_row = garment_mask[waist_y, x:x + w]
                hip_row = garment_mask[hip_y, x:x + w]
                waist_width = np.count_nonzero(waist_row)
                hip_width = np.count_nonzero(hip_row)
                if waist_width > 0 and hip_width > waist_width * 1.3:
                    waist_narrow = True

        # Leg gap detection (for pants vs skirts)
        leg_gap_detected = False
        has_two_legs = False
        if h > 100:
            leg_gap_count = 0
            two_legs_count = 0
            for slice_pos in [0.6, 0.65, 0.7, 0.75, 0.8]:
                slice_y = min(y + int(h * slice_pos), H - 1)
                if slice_y >= H:
                    continue
                row = garment_mask[slice_y, x:min(x + w, W)]
                if len(row) < 30:
                    continue

                # Split into thirds
                third = len(row) // 3
                if third < 5:
                    continue

                left = row[:third]
                center = row[third:2 * third]
                right = row[2 * third:]

                left_fill = np.count_nonzero(left) / len(left)
                center_fill = np.count_nonzero(center) / len(center)
                right_fill = np.count_nonzero(right) / len(right)

                # Leg gap pattern: left and right filled, center empty
                if left_fill > 0.2 and right_fill > 0.2 and center_fill < 0.1:
                    leg_gap_count += 1

                # Two separate legs pattern
                if left_fill > 0.25 and right_fill > 0.25 and center_fill < 0.1:
                    two_legs_count += 1

            if leg_gap_count >= 3:
                leg_gap_detected = True
            if two_legs_count >= 3:
                has_two_legs = True

        # Check for shorts (short length, covers lower body)
        is_short = False
        if garment_top > 0.3 and garment_bottom > 0.5 and garment_height_ratio < 0.4:
            is_short = True

        # DECISION TREE
        # 1. FULL BODY GARMENTS (Dresses, Jumpsuits)
        if garment_height_ratio > 0.6 and garment_top < 0.2:
            # It's a full-length garment
            if leg_gap_detected or has_two_legs:
                category = "Jumpsuit"
            else:
                # Check if it's actually a skirt (skirt would have top_fill low, bottom_fill high)
                if top_fill < 0.3 and bottom_fill > 0.5:
                    category = "Skirt"
                else:
                    category = "Dress"

        # 2. BOTTOMS (Pants, Skirt, Shorts)
        elif garment_top > 0.3 and garment_bottom > 0.5:
            # Starts below waist, covers lower body
            if leg_gap_detected or has_two_legs:
                category = "Pants"
            elif is_short:
                category = "Shorts"
            else:
                # Check if it flares out (skirt characteristic)
                if waist_narrow or (top_fill < bottom_fill * 0.7):
                    category = "Skirt"
                else:
                    category = "Pants"

        # 3. TOPS
        elif garment_top < 0.3 and garment_bottom < 0.7:
            category = "Top"

        # 4. AMBIGUOUS - use multiple signals
        else:
            if garment_height_ratio > 0.5:
                if leg_gap_detected:
                    category = "Pants"
                elif top_fill > 0.3 and bottom_fill > 0.3:
                    if top_fill < 0.3 and bottom_fill > 0.5:
                        category = "Skirt"
                    else:
                        category = "Dress"
                else:
                    category = "Top"
            else:
                category = "Top"

        # Final refinements
        if category == "Pants" and garment_height_ratio < 0.4:
            category = "Shorts"

        return category, garment_mask

    def get_dominant_color(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        Returns (Hex Color, Color Name, RGB Tuple) using KMeans.
        FIX: Uses improved mask to exclude background.
        """
        if not CV2_AVAILABLE or not SKLEARN_AVAILABLE:
            return '#808080', 'Gray', (128, 128, 128)

        h, w, _ = image.shape

        # If mask is provided, use it to sample only garment pixels
        if mask is not None and mask.size > 0:
            # Ensure mask is correct size
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Apply mask to get only garment pixels
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Get non-zero pixels (where mask is white)
            pixels = masked_image.reshape(-1, 3)
            pixels = pixels[np.any(pixels != 0, axis=1)]  # Remove black background

            # If too few pixels from mask, fall back to center crop
            if len(pixels) < 100:
                logger.warning("Mask produced too few pixels, using center crop fallback")
                start_x, start_y = int(w * 0.25), int(h * 0.25)
                end_x, end_y = int(w * 0.75), int(h * 0.75)
                crop = image[start_y:end_y, start_x:end_x]
                img_small = cv2.resize(crop, (128, 128))
                img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
                pixels = img_rgb.reshape(-1, 3)
        else:
            # Fallback to center crop if no mask
            start_x, start_y = int(w * 0.25), int(h * 0.25)
            end_x, end_y = int(w * 0.75), int(h * 0.75)
            crop = image[start_y:end_y, start_x:end_x]
            img_small = cv2.resize(crop, (128, 128))
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            pixels = img_rgb.reshape(-1, 3)

        if len(pixels) < 50:  # Need enough pixels
            return '#808080', 'Gray', (128, 128, 128)

        # Filter out very bright pixels that might be background
        mask_filter = np.any(pixels < 240, axis=1)
        filtered_pixels = pixels[mask_filter]

        # If we lost too many pixels, use original pixels
        if len(filtered_pixels) < len(pixels) * 0.2:
            data = pixels
        else:
            data = filtered_pixels

        try:
            # Use more clusters for better color detection
            n_clusters = min(5, len(data) // 20)
            if n_clusters < 2:
                n_clusters = 2

            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42, max_iter=100)
            kmeans.fit(data)

            counts = np.bincount(kmeans.labels_)
            dominant_idx = np.argmax(counts)
            r, g, b = kmeans.cluster_centers_[dominant_idx].astype(int)
        except Exception as e:
            logger.warning(f"KMeans failed: {e}, using mean fallback")
            r, g, b = np.mean(data, axis=0).astype(int)

        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        # Color name matching with expanded palette
        colors = {
            'Black': (15, 15, 15), 'White': (250, 250, 250), 'Off-White': (248, 248, 255),
            'Gray': (128, 128, 128), 'Charcoal': (54, 69, 79), 'Silver': (192, 192, 192),
            'Slate': (112, 128, 144), 'Taupe': (135, 124, 113),
            'Beige': (245, 245, 220), 'Cream': (255, 253, 208), 'Camel': (193, 154, 107),
            'Khaki': (240, 230, 140), 'Brown': (100, 50, 20), 'Coffee': (111, 78, 55),
            'Tan': (210, 180, 140), 'Rust': (183, 65, 14),
            'Red': (220, 20, 60), 'Burgundy': (128, 0, 32),
            'Pink': (255, 192, 203), 'Hot Pink': (255, 105, 180),
            'Coral': (255, 127, 80), 'Peach': (255, 218, 185),
            'Orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'Gold': (212, 175, 55),
            'Green': (0, 128, 0), 'Emerald': (80, 200, 120), 'Forest Green': (34, 139, 34),
            'Olive': (85, 107, 47), 'Sage': (156, 175, 136), 'Lime': (50, 205, 50),
            'Mint': (162, 228, 184), 'Teal': (0, 128, 128),
            'Blue': (0, 0, 255), 'Navy': (0, 0, 128), 'Royal Blue': (65, 105, 225),
            'Denim': (70, 130, 180), 'Light Denim': (135, 206, 250), 'Sky Blue': (135, 206, 235),
            'Ice Blue': (200, 230, 240), 'Cyan': (0, 255, 255), 'Turquoise': (64, 224, 208),
            'Purple': (128, 0, 128), 'Lavender': (230, 230, 250), 'Lilac': (200, 162, 200),
            'Mauve': (224, 176, 255), 'Plum': (142, 69, 133),
            'Light Blue': (173, 216, 230), 'Indigo': (75, 0, 130), 'Midnight Navy': (25, 25, 112)
        }

        min_dist = float('inf')
        initial_name = "Unknown"
        for cname, crgb in colors.items():
            dist = np.sqrt((r - crgb[0]) ** 2 + (g - crgb[1]) ** 2 + (b - crgb[2]) ** 2)
            if dist < min_dist:
                min_dist = dist
                initial_name = cname

        # HSV refinement for better color classification
        try:
            hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]
            final_name = initial_name

            if initial_name in ["Gray", "Silver", "White", "Beige"]:
                if b > r + 8 and b > g + 5 and v > 160:
                    final_name = "Ice Blue"
                elif g > r + 8 and g > b + 8 and v > 180:
                    final_name = "Mint"
                elif b > g + 8 and r > g + 8 and v > 160:
                    final_name = "Lavender"

            if final_name == "Black" and b > r + 10 and b > g + 10:
                final_name = "Navy"
            if final_name == "Navy" and s < 90:
                final_name = "Denim"
        except Exception as e:
            logger.warning(f"HSV refinement failed: {e}")
            final_name = initial_name

        return hex_color, final_name, (r, g, b)

    def analyze_texture_properties(self, image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """Analyze texture with optional mask for garment-only analysis."""
        if not CV2_AVAILABLE:
            return {"variance": 0.0, "brightness": 128.0}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Use mask if provided to analyze only garment region
        if mask is not None and mask.size > 0:
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            # Get bounding box of mask for region of interest
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w_roi, h_roi = cv2.boundingRect(coords)
                if w_roi > 10 and h_roi > 10:
                    center = masked_gray[y:y + h_roi, x:x + w_roi]
                else:
                    center = gray[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
            else:
                center = gray[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        else:
            center = gray[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]

        if center.size == 0:
            center = gray

        blurred = cv2.GaussianBlur(center, (3, 3), 0)
        laplacian_var = cv2.Laplacian(blurred, cv2.CV_64F).var()
        brightness = np.mean(center)

        return {"variance": laplacian_var, "brightness": brightness}


class FabricClassifier:
    """
    Enhanced Logic Inference Engine (LIE) for Fabric Detection.
    """
    @staticmethod
    def classify(variance: float, brightness: float, color: str, category: str) -> str:

        # 1. IMMEDIATE DENIM OVERRIDE - IMPROVED DETECTION
        denim_colors = ["Denim", "Light Denim", "Navy", "Blue", "Charcoal", "Ice Blue", "Gray", "Black",
                       "Light Blue", "Royal Blue", "Sky Blue", "Slate", "Indigo", "Midnight Navy"]

        denim_categories = ["Pants", "Shorts", "Jacket", "Skirt", "Dress", "Jumpsuit", "Top"]

        # Improved denim detection: check both color AND texture variance
        if category in denim_categories and color in denim_colors:
            # Denim has characteristic medium-high texture variance (150-800)
            # Too smooth (<100) or too rough (>1000) is probably not denim
            if 150 < variance < 800:
                return "Denim"
            # If color is specifically "Denim" or "Light Denim", trust it even with variance edge cases
            if color in ["Denim", "Light Denim"]:
                return "Denim"

        # --- ACCESSORIES & JEWELRY ---
        if category in ["Necklace", "Ring", "Earrings", "Watch", "Jewellery"]:
            if color in ["Gold", "Yellow", "Orange", "Beige", "Cream"]:
                return "Gold"
            if color in ["Silver", "Gray", "White", "Platinum", "Ash"]:
                return "Silver"
            if category == "Watch" and color in ["Black", "Brown", "Tan"]:
                return "Leather Strap"
            return "Metal"

        if category == "Bag":
            if color in ["Brown", "Tan", "Black", "Camel", "Cognac", "Red"]:
                return "Leather"
            if variance > 300:
                return "Canvas"
            return "Synthetic"

        # --- BOTTOM FABRICS ---
        if category in ["Pants", "Shorts"]:
            if variance > 500:
                return "Wool"
            if 200 < variance < 500:
                return "Cotton"
            return "Polyester"

        if category == "Skirt":
            if variance > 800:
                return "Wool"
            # TIGHTENED SATIN THRESHOLD - only extremely smooth fabrics
            if variance < 30 and brightness > 120:
                return "Satin"
            if 300 < variance < 700:
                return "Cotton"
            return "Polyester"

        # --- TOP FABRICS ---
        if category == "Top":
            if variance > 800:
                return "Wool" if brightness < 150 else "Cotton"
            # TIGHTENED SATIN THRESHOLD - only extremely smooth fabrics
            if variance < 30 and brightness > 120:
                return "Satin"
            if 300 < variance < 700 and color in ["White", "Beige", "Cream", "Olive"]:
                return "Linen"
            return "Cotton"

        # --- DRESS FABRICS ---
        if category == "Dress":
            if variance > 800:
                if color in ["Red", "Burgundy", "Navy", "Black", "Green"] and brightness < 150:
                    return "Velvet"
                return "Wool"
            # TIGHTENED SATIN THRESHOLD
            if variance < 30 and brightness > 120:
                return "Satin"
            if 300 < variance < 700:
                return "Linen"
            return "Cotton"

        # --- JUMPSUIT FABRICS ---
        if category == "Jumpsuit":
            # TIGHTENED SATIN THRESHOLD
            if variance < 30:
                return "Satin"
            if variance > 500:
                return "Wool"
            return "Cotton"

        return "Cotton"  # Default


class StyleProfile:
    """
    Manages user style preferences and profiles.
    """

    @staticmethod
    def get_default_profile() -> Dict[str, Any]:
        """Return a default style profile"""
        return {
            "everyday_look": "Casual Comfort",
            "style_archetype": "Casual",
            "style_vibes": ["Casual"],
            "preferred_colors": ["Black", "White", "Gray"],
            "color_categories": ["Monochrome"],
            "silhouette_preference": "Draped & Flowing",
            "avoid_colors": [],
            "preferred_fabrics": ["Cotton"],
            "comfort_level": 5,
            "style_keywords": ["comfortable", "versatile"],
            "body_type": "Unknown",
            "occasion_preferences": {
                "casual": "Relaxed and comfortable",
                "work": "Smart casual",
                "formal": "Classic and elegant",
                "party": "Bold and stylish"
            }
        }

    @staticmethod
    def extract_from_questionnaire(answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert questionnaire answers to style profile.
        """

        profile = StyleProfile.get_default_profile()

        # --- 1. Process Everyday Look selections ---
        everyday_looks = answers.get("everyday_look", [])
        if isinstance(everyday_looks, str):
            everyday_looks = [everyday_looks]

        # Map selected looks to style archetypes
        look_mapping = {
            "Minimalist & Clean": {
                "archetype": "Minimalist",
                "keywords": ["clean-lined", "understated", "monochromatic", "streamlined"],
                "colors": ["Monochrome"],
                "silhouette": "Tailored & Structured"
            },
            "Bold & Experimental": {
                "archetype": "Avant-Garde",
                "keywords": ["bold", "experimental", "statement", "unique", "artistic"],
                "colors": ["Bright", "Contrast"],
                "silhouette": "Architectural"
            },
            "Classic & Sophisticated": {
                "archetype": "Classic",
                "keywords": ["timeless", "elegant", "polished", "refined", "sophisticated"],
                "colors": ["Monochrome", "Neutrals"],
                "silhouette": "Tailored & Structured"
            },
            "Bohemian & Relaxed": {
                "archetype": "Boho",
                "keywords": ["flowy", "relaxed", "earthy", "layered", "free-spirited"],
                "colors": ["Earthy", "Warm"],
                "silhouette": "Draped & Flowing"
            },
            "Streetwear & Edgy": {
                "archetype": "Streetwear",
                "keywords": ["edgy", "urban", "oversized", "graphic", "casual-cool"],
                "colors": ["Monochrome", "Bold"],
                "silhouette": "Oversized & Relaxed"
            }
        }

        # Store selected vibes
        profile["style_vibes"] = everyday_looks

        # Process each selected look
        all_keywords = []
        all_colors = []

        for look in everyday_looks:
            if look in look_mapping:
                mapped = look_mapping[look]
                all_keywords.extend(mapped["keywords"])
                if mapped["colors"]:
                    all_colors.extend(mapped["colors"])

        # Set primary archetype
        if everyday_looks:
            primary_look = everyday_looks[0]
            profile["style_archetype"] = look_mapping.get(primary_look, {}).get("archetype", "Casual")
            profile["everyday_look"] = primary_look

        # Remove duplicate keywords
        profile["style_keywords"] = list(dict.fromkeys(all_keywords))

        # --- 2. Process Color Preference ---
        color_pref = answers.get("color_preference", "")
        profile["color_preference"] = color_pref

        # Parse color preference
        if "Monochrome" in color_pref:
            profile["color_categories"] = ["Monochrome"]
            profile["preferred_colors"] = ["Black", "White", "Gray", "Charcoal", "Silver"]
        elif "Pastels" in color_pref:
            profile["color_categories"] = ["Pastels"]
            profile["preferred_colors"] = ["Blush", "Lavender", "Mint", "Baby Blue", "Cream", "Powder Pink"]
        elif "Earthy" in color_pref or "Warm" in color_pref:
            profile["color_categories"] = ["Earthy"]
            profile["preferred_colors"] = ["Olive", "Brown", "Tan", "Rust", "Sage", "Camel", "Terracotta"]
        elif "Bright" in color_pref:
            profile["color_categories"] = ["Bright"]
            profile["preferred_colors"] = ["Red", "Royal Blue", "Emerald", "Yellow", "Coral", "Fuchsia"]
        else:
            if "Black/White" in color_pref:
                profile["preferred_colors"].extend(["Black", "White"])
            if "Pastels" in color_pref:
                profile["preferred_colors"].extend(["Blush", "Lavender", "Mint"])

        # --- 3. Process Silhouette Preference ---
        silhouette = answers.get("silhouette", "Draped & Flowing")
        profile["silhouette_preference"] = silhouette

        # Add silhouette to keywords
        if silhouette == "Tailored & Structured":
            profile["style_keywords"].extend(["tailored", "structured", "sharp"])
        elif silhouette == "Draped & Flowing":
            profile["style_keywords"].extend(["flowy", "draped", "soft"])
        elif silhouette == "Oversized & Relaxed":
            profile["style_keywords"].extend(["oversized", "relaxed", "comfortable"])

        # Remove duplicates
        profile["style_keywords"] = list(dict.fromkeys(profile["style_keywords"]))

        return profile


class FashionAIModel:
    vision = LocalComputerVision()
    classifier = FabricClassifier()

    # Store user profiles
    _user_profiles: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    async def set_user_style_profile(user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set or update a user's style profile from questionnaire answers.
        """
        if "answers" in profile_data:
            profile = StyleProfile.extract_from_questionnaire(profile_data["answers"])
        else:
            profile = {**StyleProfile.get_default_profile(), **profile_data}

        FashionAIModel._user_profiles[user_id] = profile

        logger.info(f"Style profile set for user {user_id}: {profile['style_archetype']}")

        return {
            "success": True,
            "profile": profile,
            "message": f"Style DNA mapped to {profile['style_archetype']}"
        }

    @staticmethod
    def get_user_style_profile(user_id: str) -> Dict[str, Any]:
        """
        Get a user's style profile.
        """
        return FashionAIModel._user_profiles.get(user_id, StyleProfile.get_default_profile())

    @staticmethod
    async def autotag_garment(image_data: str) -> Dict[str, Any]:
        """
        FIXED: Uses improved mask for color and texture, and better category detection.
        """
        try:
            # Validate input
            if not image_data or not isinstance(image_data, str):
                raise ValueError("Invalid image_data: must be non-empty string")

            # Decode image
            img = FashionAIModel.vision.decode_image(image_data)

            # Check if image loaded successfully
            if img is None or img.size == 0 or np.all(img == 0):
                raise ValueError("Failed to decode image or image is empty")

            # Get improved category and mask
            category, garment_mask = FashionAIModel.vision.analyze_shape(img)

            # Use mask for accurate color detection
            hex_color, color_name, rgb = FashionAIModel.vision.get_dominant_color(img, mask=garment_mask)

            # Use mask for texture analysis
            texture = FashionAIModel.vision.analyze_texture_properties(img, mask=garment_mask)

            # Classify fabric using improved logic
            fabric = FashionAIModel.classifier.classify(
                variance=texture['variance'],
                brightness=texture['brightness'],
                color=color_name,
                category=category
            )

            # Smart category refinement
            final_category = category
            final_name_parts = []

            # Add fabric if distinctive
            if fabric not in ["Cotton", "Polyester", "Unknown"]:
                final_name_parts.append(fabric)

            # Add color
            final_name_parts.append(color_name)

            # Add specific category name with special handling for Denim
            if fabric == "Denim" and category in ["Pants", "Shorts", "Skirt", "Jacket", "Dress", "Jumpsuit", "Top"]:
                denim_names = {
                    "Pants": "Jeans",
                    "Shorts": "Denim Shorts",
                    "Skirt": "Denim Skirt",
                    "Jacket": "Denim Jacket",
                    "Dress": "Denim Dress",
                    "Jumpsuit": "Denim Jumpsuit",
                    "Top": "Denim Top"
                }
                final_name_parts.append(denim_names.get(category, category))
            else:
                final_name_parts.append(category)

            final_name = " ".join(final_name_parts)

            # Complementary color matching
            best_color = "White"
            if fashion_matcher:
                try:
                    candidates = ['White', 'Black', 'Denim', 'Navy', 'Beige', 'Gray', 'Camel']
                    best_score = 0

                    dummy_input = {
                        'color': color_name,
                        'category': final_category,
                        'fabric': fabric
                    }

                    for c in candidates:
                        if final_category in ['Dress', 'Jumpsuit']:
                            target_cat = 'Accessory'
                        elif final_category in ['Top']:
                            target_cat = 'Pants'
                        else:
                            target_cat = 'Top'

                        dummy_partner = {
                            'color': c,
                            'category': target_cat,
                            'fabric': 'Cotton'
                        }

                        res = fashion_matcher.match_items(dummy_input, dummy_partner)
                        if res and res.get('compatibility_score', 0) > best_score:
                            best_score = res['compatibility_score']
                            best_color = c

                except Exception as e:
                    logger.warning(f"Color matching failed: {e}")
                    # Fallback to basic color theory
                    best_color = FashionAIModel._fallback_color_match(color_name)

            return {
                "success": True,
                "name": final_name,
                "category": final_category,
                "fabric": fabric,
                "color": color_name,
                "hex_color": hex_color,
                "rgb": rgb,
                "best_color": best_color,
                "details": f"AI Scan: {fabric} {category} | Color: {color_name} ({hex_color})",
                "confidence": 0.96,
                "texture_variance": round(texture['variance'], 2),
                "brightness": round(texture['brightness'], 2)
            }

        except ValueError as ve:
            logger.error(f"Validation error in autotag_garment: {ve}")
            return {
                "success": False,
                "error": str(ve),
                "name": "Unknown Item",
                "category": "Unknown",
                "fabric": "Unknown",
                "color": "Unknown",
                "best_color": "White",
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Unexpected error in autotag_garment: {e}")
            return {
                "success": False,
                "error": "Internal processing error",
                "name": "Scanned Item",
                "category": "Top",
                "fabric": "Cotton",
                "color": "Multi",
                "best_color": "White",
                "confidence": 0.5
            }

    @staticmethod
    def _fallback_color_match(base_color: str) -> str:
        """Fallback color matching when fashion_matcher is unavailable."""
        color_complements = {
            'Black': 'White',
            'White': 'Black',
            'Navy': 'Beige',
            'Blue': 'White',
            'Denim': 'White',
            'Gray': 'Black',
            'Red': 'Black',
            'Green': 'Beige',
            'Yellow': 'Navy',
            'Pink': 'Gray',
            'Purple': 'Black',
            'Orange': 'Navy',
            'Brown': 'Cream',
            'Beige': 'Navy'
        }
        return color_complements.get(base_color, 'White')

    @staticmethod
    async def get_outfit_suggestion(image_data: str, variation: int = 0, user_id: str = None, season: str = "summer") -> Dict[str, Any]:
        """
        FIX 3: Variation now produces truly different suggestions.
        """
        try:
            img = FashionAIModel.vision.decode_image(image_data)

            # Get category and mask
            category, _ = FashionAIModel.vision.analyze_shape(img)

            # Get color
            hex_color, color, rgb = FashionAIModel.vision.get_dominant_color(img)

            texture = FashionAIModel.vision.analyze_texture_properties(img)
            fabric = FashionAIModel.classifier.classify(
                variance=texture['variance'],
                brightness=texture['brightness'],
                color=color,
                category=category
            )

            # Get user's Style DNA profile
            user_profile = FashionAIModel.get_user_style_profile(user_id) if user_id else StyleProfile.get_default_profile()

            # FIX: Use variation to seed random differently
            # Combine image hash, variation, and current time to ensure different results
            base_seed = hash(image_data[:100]) + variation * 1000 + int(datetime.utcnow().timestamp() % 100)
            random.seed(base_seed)

            # Determine vibe based on style DNA and variation
            style_archetype = user_profile.get("style_archetype", "Casual")
            style_keywords = user_profile.get("style_keywords", [])
            style_vibes = user_profile.get("style_vibes", [])

            vibe = style_archetype

            if style_vibes and len(style_vibes) > 1:
                vibe_idx = variation % len(style_vibes)
                selected_look = style_vibes[vibe_idx]

                look_to_archetype = {
                    "Minimalist & Clean": "Minimalist",
                    "Bold & Experimental": "Avant-Garde",
                    "Classic & Sophisticated": "Classic",
                    "Bohemian & Relaxed": "Boho",
                    "Streetwear & Edgy": "Streetwear"
                }
                vibe = look_to_archetype.get(selected_look, style_archetype)

            logger.info(f"Using Style DNA: {vibe} for user {user_id}, season: {season}, variation: {variation}")

            # Match Logic for complementary color
            input_item = {'color': color, 'category': category, 'fabric': fabric}

            preferred_colors = user_profile.get("preferred_colors", [])
            color_categories = user_profile.get("color_categories", [])

            color_expansion = {
                "Monochrome": ["Black", "White", "Gray", "Charcoal", "Silver"],
                "Pastels": ["Blush", "Lavender", "Mint", "Baby Blue", "Cream"],
                "Bright": ["Red", "Royal Blue", "Emerald", "Yellow", "Coral"],
                "Earthy": ["Olive", "Brown", "Tan", "Rust", "Sage", "Camel"],
                "Neutrals": ["Beige", "Navy", "Camel", "Gray", "White"]
            }

            candidates = []
            for cat in color_categories:
                if cat in color_expansion:
                    candidates.extend(color_expansion[cat])

            candidates.extend(preferred_colors)
            candidates.extend(['Black', 'White', 'Navy', 'Beige', 'Denim', 'Gray'])

            avoid_colors = user_profile.get("avoid_colors", [])
            candidates = [c for c in candidates if c not in avoid_colors]

            # Remove duplicates
            seen = set()
            candidates = [x for x in candidates if not (x in seen or seen.add(x))]

            random.shuffle(candidates)

            best_match_color = candidates[0] if candidates else "White"
            best_score = -1

            if fashion_matcher:
                for cand_color in candidates:
                    if category in ['Pants', 'Shorts', 'Skirt']:
                        target_cat = 'Top'
                    elif category in ['Top']:
                        target_cat = 'Pants'
                    else:
                        target_cat = 'Accessory'

                    dummy_partner = {'color': cand_color, 'category': target_cat, 'fabric': 'Cotton'}
                    result = fashion_matcher.match_items(input_item, dummy_partner)

                    if result:
                        preference_boost = 20 if cand_color in preferred_colors else 0
                        # Add randomness to score
                        randomized_score = result.get('compatibility_score', 0) + (random.random() * 20) + preference_boost

                        if randomized_score > best_score:
                            best_score = randomized_score
                            best_match_color = cand_color

            # --- AI-GENERATED SUGGESTIONS ---
            detected_item = f"{color} {fabric} {category}"

            silhouette = user_profile.get("silhouette_preference", "Draped & Flowing")

            # Pass variation to suggestion generators
            shoe_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="shoes",
                base_item=detected_item,
                vibe=vibe,
                style_keywords=style_keywords,
                silhouette=silhouette,
                color_context=f"complementing {best_match_color}",
                season=season,
                variation=variation
            )

            jewelry_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="jewelry/accessory",
                base_item=detected_item,
                vibe=vibe,
                style_keywords=style_keywords,
                silhouette=silhouette,
                color_context=f"with {best_match_color} accents",
                season=season,
                variation=variation
            )

            bag_suggestion = await FashionAIModel._generate_style_dna_suggestion(
                item_type="bag",
                base_item=detected_item,
                vibe=vibe,
                style_keywords=style_keywords,
                silhouette=silhouette,
                color_context=f"in {best_match_color}",
                season=season,
                variation=variation
            )

            # Construct match description
            if category in ['Pants', 'Shorts', 'Skirt']:
                match_piece_str = f"{best_match_color} Top"
            elif category == 'Top':
                match_piece_str = f"{best_match_color} Bottom"
            elif category in ['Dress', 'Jumpsuit']:
                match_piece_str = ""
            else:
                match_piece_str = f"{best_match_color} Matching Piece"

            # Generate styling tips
            styling_tips = await FashionAIModel._generate_style_dna_tips(
                detected_item, vibe, style_keywords, best_match_color, season
            )

            # Create style DNA message
            style_dna_message = f"Based on your {style_archetype} Style DNA"
            if style_vibes:
                style_dna_message = f"Based on your {', '.join(style_vibes)} Style DNA"

            # Add season emoji
            season_emoji = {
                "summer": "☀️",
                "winter": "❄️",
                "spring": "🌸",
                "fall": "🍂",
                "all-season": "🌤️"
            }.get(season, "☀️")

            return {
                "style_dna": style_dna_message,
                "season": f"{season_emoji} {season.capitalize()}",
                "vibe": vibe,
                "identified_item": detected_item,
                "match_piece": match_piece_str,
                "jewelry": jewelry_suggestion,
                "shoes": shoe_suggestion,
                "bag": bag_suggestion,
                "best_color": best_match_color,
                "styling_tips": styling_tips,
                "silhouette": silhouette
            }

        except Exception as e:
            logger.error(f"Suggestion failed: {e}")
            return {
                "style_dna": "Based on your Casual Style DNA",
                "season": "☀️ Summer",
                "vibe": "Casual",
                "identified_item": "Unknown Item",
                "match_piece": "",
                "jewelry": "Simple accessory",
                "shoes": "Classic sneakers",
                "bag": "Versatile tote",
                "best_color": "White",
                "styling_tips": "Keep it simple and comfortable",
                "silhouette": "Relaxed"
            }

    @staticmethod
    async def _generate_style_dna_suggestion(item_type: str, base_item: str, vibe: str,
                                             style_keywords: List[str], silhouette: str,
                                             color_context: str, season: str = "summer",
                                             variation: int = 0) -> str:
        """
        Generate suggestions that follow the user's Style DNA and are seasonally appropriate.
        FIXED: Variation now affects the selection.
        """
        import random

        # Seed with variation to ensure different outputs
        random.seed(hash(f"{base_item}{vibe}{season}{variation}") % 2**32)

        # STYLE DNA SPECIFIC LIBRARIES

        # MINIMALIST Style DNA - SUMMER version
        minimalist_items_summer = {
            "shoes": [
                "White leather sneakers",
                "Minimalist leather sandals",
                "Streamlined mules",
                "Understated ballet flats",
                "Clean espadrilles",
                "Simple slide sandals",
                "Breathable mesh sneakers",
                "Beige canvas sneakers",
                "Black leather loafers",
                "Nude block heels"
            ],
            "jewelry/accessory": [
                "Thin gold band",
                "Small silver studs",
                "Geometric pendant necklace",
                "Minimalist watch with mesh band",
                "Delicate chain bracelet",
                "Single hoop earrings",
                "Simple cuff bracelet",
                "Tiny diamond necklace",
                "Pearl studs",
                "Thin silver bangle"
            ],
            "bag": [
                "Structured leather tote",
                "Clean canvas shopper",
                "Minimalist crossbody bag",
                "Straw beach bag",
                "Streamlined backpack",
                "Understated shoulder bag",
                "Architectural clutch",
                "Bamboo handle bag",
                "Nylon belt bag",
                "Clear acrylic purse"
            ]
        }

        # MINIMALIST Style DNA - WINTER version
        minimalist_items_winter = {
            "shoes": [
                "Black chelsea boots",
                "Minimalist leather loafers",
                "Clean derby shoes",
                "Streamlined ankle boots",
                "Simple leather lace-ups",
                "Understated heeled boots",
                "Minimalist winter sneakers",
                "Sleek knee-high boots",
                "Black leather oxfords",
                "Minimalist combat boots"
            ],
            "jewelry/accessory": [
                "Thin gold band",
                "Small silver studs",
                "Geometric pendant necklace",
                "Minimalist leather watch",
                "Delicate chain bracelet",
                "Single hoop earrings",
                "Simple cuff bracelet",
                "Wool scarf in neutral tone",
                "Leather gloves",
                "Cashmere beanie"
            ],
            "bag": [
                "Structured leather tote",
                "Clean canvas shopper",
                "Minimalist crossbody bag",
                "Sleek leather backpack",
                "Understated shoulder bag",
                "Architectural saddle bag",
                "Leather doctor bag",
                "Minimalist belt bag",
                "Felt carryall",
                "Structured satchel"
            ]
        }

        # BOHO Style DNA - SUMMER version
        boho_items_summer = {
            "shoes": [
                "Woven espadrilles",
                "Leather gladiator sandals",
                "Embroidered mules",
                "Beaded slides",
                "Natural fiber wedges",
                "Crochet slip-ons",
                "Fringed sandals",
                "Suede ankle boots",
                "Tassel loafers",
                "Macrame sandals"
            ],
            "jewelry/accessory": [
                "Turquoise pendant necklace",
                "Layered beaded bracelets",
                "Feather earrings",
                "Stacked silver rings",
                "Boho charm bracelet",
                "Macrame choker",
                "Crystal statement necklace",
                "Shell anklet",
                "Wooden bangles",
                "Dreamcatcher earrings"
            ],
            "bag": [
                "Woven straw tote",
                "Embroidered hobo bag",
                "Beaded clutch",
                "Macrame shoulder bag",
                "Fringed crossbody bag",
                "Tasseled leather satchel",
                "Suede bucket bag",
                "Patchwork backpack",
                "Tribal print bag",
                "Boho fringe bag"
            ]
        }

        # BOHO Style DNA - WINTER version
        boho_items_winter = {
            "shoes": [
                "Suede ankle boots",
                "Fringed western boots",
                "Embroidered booties",
                "Tasseled loafers",
                "Beaded winter boots",
                "Crochet-lined boots",
                "Suede knee-high boots",
                "Fringed combat boots",
                "Embroidered snow boots",
                "Leather moccasins"
            ],
            "jewelry/accessory": [
                "Turquoise pendant necklace",
                "Layered beaded bracelets",
                "Feather earrings",
                "Stacked silver rings",
                "Boho charm bracelet",
                "Crystal statement necklace",
                "Woven scarf",
                "Fringed shawl",
                "Tassel beanie",
                "Embroidered gloves"
            ],
            "bag": [
                "Fringed crossbody bag",
                "Embroidered hobo bag",
                "Tasseled leather satchel",
                "Suede bucket bag",
                "Beaded shoulder bag",
                "Patchwork tote",
                "Fringed backpack",
                "Wool tapestry bag",
                "Boho saddle bag",
                "Embroidered duffel"
            ]
        }

        # STREETWEAR Style DNA - SUMMER version
        streetwear_items_summer = {
            "shoes": [
                "Air Force 1s",
                "Dunk Low SB",
                "Tech runner sneakers",
                "Box fresh retro runners",
                "Breathable mesh sneakers",
                "Low-top skate shoes",
                "Slide sandals with socks",
                "Yeezy foam runners",
                "New Balance 550s",
                "Jordan 1 Lows"
            ],
            "jewelry/accessory": [
                "Chunky silver chain",
                "Gold hoop earrings",
                "Layered pendant necklace",
                "Diamond stud earrings",
                "Statement watch",
                "Silver cuff bracelet",
                "Chain belt",
                "Bucket hat",
                "Baseball cap",
                "Crossbody phone case"
            ],
            "bag": [
                "Crossbody fanny pack",
                "Nylon utility bag",
                "Mini backpack",
                "Chest rig",
                "Tech fabric tote",
                "Logo belt bag",
                "Graffiti print backpack",
                "Transparent bag",
                "Puffer bag",
                "Webbing belt bag"
            ]
        }

        # STREETWEAR Style DNA - WINTER version
        streetwear_items_winter = {
            "shoes": [
                "Chunky dad sneakers",
                "Jordan 1 Highs",
                "Yeezy 700s",
                "Dunk High SB",
                "Tech runner sneakers",
                "Air Force 1s high",
                "Puffy winter sneakers",
                "Timberland boots",
                "Puma cell sneakers",
                "Balenciaga runners"
            ],
            "jewelry/accessory": [
                "Chunky silver chain",
                "Gold hoop earrings",
                "Layered pendant necklace",
                "Diamond stud earrings",
                "Statement watch",
                "Silver cuff bracelet",
                "Chain belt",
                "Beanie with logo",
                "Technical gloves",
                "Puffer scarf"
            ],
            "bag": [
                "Crossbody fanny pack",
                "Nylon utility bag",
                "Mini backpack",
                "Chest rig",
                "Tech fabric tote",
                "Logo belt bag",
                "Graffiti print backpack",
                "Puffer bag",
                "Technical backpack",
                "Webbing crossbody"
            ]
        }

        # CLASSIC Style DNA - SUMMER version
        classic_items_summer = {
            "shoes": [
                "Leather loafers",
                "Ballet flats",
                "Derby shoes",
                "Elegant sandals",
                "Penny loafers",
                "Espadrilles",
                "Classic pumps",
                "Slingback heels",
                "Spectator shoes",
                "Mary Janes"
            ],
            "jewelry/accessory": [
                "Pearl stud earrings",
                "Tennis bracelet",
                "Gold pendant necklace",
                "Classic watch",
                "Signet ring",
                "Silk scarf",
                "Cameo brooch",
                "Gold hoop earrings",
                "Charm bracelet",
                "Locket necklace"
            ],
            "bag": [
                "Structured leather tote",
                "Classic flap bag",
                "Top handle satchel",
                "Leather doctor bag",
                "Elegant shoulder bag",
                "Frame clutch",
                "Woven leather bag",
                "Chain strap bag",
                "Bowler bag",
                "Classic duffle"
            ]
        }

        # CLASSIC Style DNA - WINTER version
        classic_items_winter = {
            "shoes": [
                "Cap toe oxfords",
                "Leather loafers",
                "Classic pumps",
                "Derby shoes",
                "Elegant ankle boots",
                "Penny loafers",
                "Knee-high boots",
                "Chukka boots",
                "Wingtip oxfords",
                "Chelsea boots"
            ],
            "jewelry/accessory": [
                "Pearl stud earrings",
                "Tennis bracelet",
                "Gold pendant necklace",
                "Classic watch",
                "Signet ring",
                "Silk scarf",
                "Cameo brooch",
                "Cashmere scarf",
                "Leather gloves",
                "Wool beret"
            ],
            "bag": [
                "Structured leather tote",
                "Classic flap bag",
                "Top handle satchel",
                "Leather doctor bag",
                "Elegant shoulder bag",
                "Frame clutch",
                "Bucket bag",
                "Saddle bag",
                "Leather backpack",
                "Classic holdall"
            ]
        }

        # AVANT-GARDE Style DNA - SUMMER version
        avantgarde_items_summer = {
            "shoes": [
                "Architectural heel sandals",
                "Asymmetric pumps",
                "Sculptural sandals",
                "Geometric mules",
                "Abstract design sneakers",
                "Cut-out booties",
                "Deconstructed slides",
                "Platform creepers",
                "Spiral heels",
                "Modular sandals"
            ],
            "jewelry/accessory": [
                "Geometric statement necklace",
                "Asymmetric earrings",
                "Sculptural cuff",
                "Abstract brooch",
                "Architectural ring",
                "Deconstructed chain",
                "Mixed metal pieces",
                "Modular earrings",
                "Kinetic jewelry",
                "Resin statement piece"
            ],
            "bag": [
                "Sculptural structured bag",
                "Asymmetric clutch",
                "Geometric shoulder bag",
                "Deconstructed tote",
                "Architectural backpack",
                "Abstract print bag",
                "Mixed material hobo",
                "Modular bag system",
                "Origami fold bag",
                "Clear structural purse"
            ]
        }

        # AVANT-GARDE Style DNA - WINTER version
        avantgarde_items_winter = {
            "shoes": [
                "Architectural heel boots",
                "Asymmetric pumps",
                "Sculptural boots",
                "Deconstructed oxfords",
                "Platform creepers",
                "Geometric booties",
                "Abstract design winter boots",
                "Modular boot system",
                "Spiral heel booties",
                "Cut-out combat boots"
            ],
            "jewelry/accessory": [
                "Geometric statement necklace",
                "Asymmetric earrings",
                "Sculptural cuff",
                "Abstract brooch",
                "Architectural ring",
                "Deconstructed chain",
                "Mixed metal pieces",
                "Modular necklace",
                "Kinetic bracelet",
                "Resin cuff"
            ],
            "bag": [
                "Sculptural structured bag",
                "Asymmetric clutch",
                "Geometric shoulder bag",
                "Deconstructed tote",
                "Architectural backpack",
                "Abstract print bag",
                "Mixed material hobo",
                "Modular bag",
                "Origami fold tote",
                "Sculptural saddle"
            ]
        }

        # Map vibe to the appropriate style library based on season
        if season.lower() in ["summer", "spring"]:
            style_libraries = {
                "Minimalist": minimalist_items_summer,
                "Boho": boho_items_summer,
                "Streetwear": streetwear_items_summer,
                "Classic": classic_items_summer,
                "Avant-Garde": avantgarde_items_summer,
                "Casual": minimalist_items_summer,
                "Elegant": classic_items_summer,
                "Edgy": streetwear_items_summer,
                "Vintage": boho_items_summer,
                "Bohemian": boho_items_summer,
                "Chic": classic_items_summer,
                "Glam": classic_items_summer,
                "Retro": boho_items_summer,
                "Sporty": streetwear_items_summer,
                "Grunge": streetwear_items_summer,
                "Preppy": classic_items_summer,
                "Romantic": boho_items_summer
            }
        else:  # winter or fall
            style_libraries = {
                "Minimalist": minimalist_items_winter,
                "Boho": boho_items_winter,
                "Streetwear": streetwear_items_winter,
                "Classic": classic_items_winter,
                "Avant-Garde": avantgarde_items_winter,
                "Casual": minimalist_items_winter,
                "Elegant": classic_items_winter,
                "Edgy": streetwear_items_winter,
                "Vintage": boho_items_winter,
                "Bohemian": boho_items_winter,
                "Chic": classic_items_winter,
                "Glam": classic_items_winter,
                "Retro": boho_items_winter,
                "Sporty": streetwear_items_winter,
                "Grunge": streetwear_items_winter,
                "Preppy": classic_items_winter,
                "Romantic": boho_items_winter
            }

        # Get the right style library based on vibe
        default_lib = minimalist_items_summer if season in ["summer", "spring"] else minimalist_items_winter
        style_library = style_libraries.get(vibe, default_lib)

        # Get items for the requested type
        items_for_type = style_library.get(item_type, style_library.get("shoes", default_lib["shoes"]))

        # If we have style keywords, use them to filter and prioritize
        if style_keywords:
            keyword_matches = []
            other_items = []

            for item in items_for_type:
                item_lower = item.lower()
                if any(keyword.lower() in item_lower for keyword in style_keywords):
                    keyword_matches.append(item)
                else:
                    other_items.append(item)

            if keyword_matches:
                # Interleave matches and others to maintain variety
                items_for_type = []
                for i in range(max(len(keyword_matches), len(other_items))):
                    if i < len(keyword_matches):
                        items_for_type.append(keyword_matches[i])
                    if i < len(other_items):
                        items_for_type.append(other_items[i])
            else:
                items_for_type = items_for_type

        # Ensure we have enough items
        if not items_for_type:
            items_for_type = default_lib.get(item_type, default_lib["shoes"])

        # FIX: Use variation to pick different items each time
        if len(items_for_type) > 1:
            # Use variation to select an index, ensuring different results
            idx = (variation + hash(item_type) % 100) % len(items_for_type)
            suggestion = items_for_type[idx]
        else:
            suggestion = items_for_type[0]

        # Add color context to some items based on variation
        color_match = best_match_color_from_context(color_context)
        if (variation + hash(suggestion)) % 3 == 0:  # 1/3 chance, but deterministic based on variation
            suggestion = f"{suggestion} in {color_match}"

        return suggestion

    @staticmethod
    async def _generate_style_dna_tips(base_item: str, vibe: str, style_keywords: List[str], accent_color: str, season: str = "summer") -> str:
        """
        Generate styling tips based on Style DNA and season.
        """
        style_tips = {
            "Minimalist": {
                "summer": [
                    "Keep accessories minimal - let the clean lines speak for themselves.",
                    "Choose breathable linens and cottons for hot days.",
                    "A monochromatic palette in light colors keeps you cool.",
                    "Opt for sleeveless silhouettes that maintain clean lines.",
                    "Negative space and lighter fabrics are your summer friends."
                ],
                "winter": [
                    "Focus on quality wool and cashmere for warmth without bulk.",
                    "Layer thoughtfully - a fine gauge turtleneck under a structured coat.",
                    "Stick to a monochromatic palette in deeper tones for winter.",
                    "Choose one statement piece and keep everything else understated.",
                    "Invest in a well-tailored wool coat - it elevates everything."
                ]
            },
            "Boho": {
                "summer": [
                    "Layer lightweight textures like crochet, lace, and gauze.",
                    "Mix earthy tones with pops of turquoise or coral for summer.",
                    "Flowy maxi dresses in breathable fabrics keep you cool.",
                    "Natural materials like wood and shell add authentic bohemian charm.",
                    "Embrace bare shoulders and open backs in the heat."
                ],
                "winter": [
                    "Layer chunky knits over flowing skirts with tights.",
                    "Mix warm textures like suede, wool, and velvet.",
                    "Deep jewel tones and earthy shades work beautifully for winter.",
                    "Fringed boots and layered scarves add boho warmth.",
                    "A poncho or fringed shawl completes the winter boho look."
                ]
            },
            "Streetwear": {
                "summer": [
                    "Play with proportions - oversized tees with bike shorts.",
                    "Fresh white sneakers complete any summer streetwear look.",
                    "Lightweight technical fabrics keep you cool.",
                    "Bucket hats and fanny packs add summer street cred.",
                    "Layer with mesh or technical vests for dimension."
                ],
                "winter": [
                    "Oversized puffers are both warm and stylish.",
                    "Layer hoodies under denim jackets for warmth.",
                    "Chunky sneakers work with thick socks.",
                    "Beanies and technical fabrics are winter essentials.",
                    "Cargo pants with thermal lining keep you cozy."
                ]
            },
            "Classic": {
                "summer": [
                    "Invest in quality linen and cotton basics.",
                    "A well-tailored white shirt elevates any summer outfit.",
                    "Pearls and gold tones add timeless elegance to summer looks.",
                    "Stick to a neutral palette with navy or blush accents.",
                    "Classic doesn't mean boring - play with textures like seersucker."
                ],
                "winter": [
                    "A cashmere sweater in a neutral tone is worth the investment.",
                    "Well-tailored wool trousers create clean lines.",
                    "Pearls peeking out from a turtleneck add elegance.",
                    "A camel hair coat is the ultimate classic winter piece.",
                    "Leather gloves and a silk scarf complete the polished look."
                ]
            },
            "Avant-Garde": {
                "summer": [
                    "Let your outfit be a conversation starter even in summer.",
                    "Mix unexpected lightweight textures like mesh and organza.",
                    "Asymmetric cuts create visual interest without extra layers.",
                    "Don't follow trends - set them, even in the heat.",
                    "Architectural accessories in lighter materials complete the look."
                ],
                "winter": [
                    "Sculptural coats become the focal point.",
                    "Layer deconstructed pieces for warmth and interest.",
                    "Bold silhouettes stand out against winter neutrals.",
                    "Mixed textures like wool and leather create dimension.",
                    "Statement outerwear is worth the investment."
                ]
            },
            "Casual": {
                "summer": [
                    "Comfort is key - choose soft, breathable cotton and linen.",
                    "Well-fitted basics create an effortlessly cool summer look.",
                    "Add interest with texture rather than pattern.",
                    "A great pair of white sneakers grounds any casual summer outfit.",
                    "Layer with a lightweight denim jacket for cooler evenings."
                ],
                "winter": [
                    "Comfort is key - choose soft sweaters and warm layers.",
                    "Well-fitted basics under cardigans create effortless style.",
                    "Add interest with textured knits.",
                    "A great pair of boots grounds any casual winter outfit.",
                    "Layer with a puffer vest for warmth without bulk."
                ]
            }
        }

        # Get tips for this vibe and season
        vibe_tips_dict = style_tips.get(vibe, style_tips["Casual"])
        season_tips = vibe_tips_dict.get(season.lower(), vibe_tips_dict.get("summer",
            ["Style is personal - wear what makes you feel confident!"]))

        # Add color-specific tip
        color_tip = f"The {accent_color} accents will tie your whole {season} look together."

        all_tips = season_tips + [color_tip]
        return random.choice(all_tips)

    @staticmethod
    async def generate_outfits_from_wardrobe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates outfit vibes using the Advanced Engine."""
        if not fashion_matcher:
            return []

        outfits = []
        styles = ['casual', 'business', 'streetwear']

        for style in styles:
            outfit = fashion_matcher.create_complete_outfit(items, style=style)
            if outfit and 'items' in outfit:
                outfits.append({
                    "name": outfit['name'],
                    "vibe": outfit['vibe'],
                    "item_ids": outfit['item_ids']
                })

        return outfits

    @staticmethod
    def get_evolution_data(items: List[Dict[str, Any]], history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
        """Analyzes wardrobe health/gaps."""
        if not fashion_matcher:
            return {
                "timeline": [],
                "insights": {
                    "dominant_style": "Casual",
                    "style_change": "0%",
                    "color_preferences": [],
                    "style_confidence": 50,
                    "wardrobe_size": len(items),
                    "recommendations": []
                }
            }

        analysis = fashion_matcher.analyze_wardrobe_gaps(items)
        timeline = []

        def format_date(iso_str):
            try:
                dt = datetime.fromisoformat(iso_str)
                return dt.strftime("%b %d")
            except:
                return "Unknown"

        for i, entry in enumerate(history):
            style_list = []
            try:
                style_list = json.loads(entry.get('styles', '[]'))
            except:
                pass

            primary_style = style_list[0] if style_list else entry.get('archetype', 'Mapped')

            mood = "Exploring"
            if 'Minimalist' in primary_style: mood = "Clean & Sharp"
            elif 'Streetwear' in primary_style: mood = "Bold & Expressive"
            elif 'Vintage' in primary_style: mood = "Nostalgic"

            timeline.append({
                "period": format_date(entry.get('created_at', '')),
                "stage": primary_style,
                "style": " & ".join(style_list[:2]),
                "color": "Personalized",
                "mood": mood,
                "progress": entry.get('comfort_level', 50),
                "items": len(items),
                "key_item": "DNA Profile",
                "is_current": i == 0
            })

        return {
            "timeline": timeline,
            "insights": {
                "dominant_style": analysis.get('dominant_style', 'Casual'),
                "style_change": f"{analysis.get('wardrobe_health_score', 50)}%",
                "color_preferences": analysis.get('color_preferences', [])[:5],
                "style_confidence": analysis.get('wardrobe_health_score', 50),
                "wardrobe_size": analysis.get('total_items', len(items)),
                "recommendations": analysis.get('recommendations', [])
            }
        }

    @staticmethod
    def curate_trip(city: str, duration: int, vibe: str) -> Dict[str, Any]:
        """Curates a packing list for a trip."""
        CITY_SHOPPING_GUIDE = {
            "Delhi": {
                "markets": ["Khan Market", "Sarojini Nagar", "Dilli Haat", "Select Citywalk"],
                "gems": ["Hauz Khas Village", "Shahpur Jat", "Meherchand Market", "Santushti Complex"]
            },
            "Mumbai": {
                "markets": ["Colaba Causeway", "Linking Road", "Hill Road", "Palladium Mall"],
                "gems": ["Kala Ghoda Boutiques", "Chor Bazaar (Vintage)", "Bandra 190", "Le Mill"]
            },
            "London": {
                "markets": ["Oxford Street", "Regent Street", "Camden Market", "Covent Garden"],
                "gems": ["Carnaby Street", "Seven Dials", "Shoreditch Boxpark", "Brick Lane Vintage"]
            },
            "Paris": {
                "markets": ["Champs-Élysées", "Galeries Lafayette", "Le Marais", "Rue de Rivoli"],
                "gems": ["Canal Saint-Martin", "Passage des Panoramas", "Rue Saint-Honoré", "Montmartre Boutiques"]
            },
            "New York": {
                "markets": ["Fifth Avenue", "SoHo Broadway", "Herald Square", "Chelsea Market"],
                "gems": ["Williamsburg Vintage", "Meatpacking District", "Nolita Boutiques", "Greenwich Village"]
            },
            "Tokyo": {
                "markets": ["Ginza District", "Shibuya 109", "Omotesando Hills", "Shinjuku Station"],
                "gems": ["Shimokitazawa (Vintage)", "Harajuku Takeshita St", "Daikanyama T-Site", "Cat Street"]
            },
            "Milan": {
                "markets": ["Galleria Vittorio Emanuele", "Corso Buenos Aires", "Via Monte Napoleone", "La Rinascente"],
                "gems": ["Brera District", "Navigli Vintage", "Corso di Porta Ticinese", "10 Corso Como"]
            },
            "Dubai": {
                "markets": ["The Dubai Mall", "Mall of the Emirates", "City Walk", "Gold Souk"],
                "gems": ["Alserkal Avenue", "Boxpark", "Design District (d3)", "Global Village"]
            }
        }

        target_city = city.title().strip()
        shopping_data = CITY_SHOPPING_GUIDE.get(target_city)

        if not shopping_data:
            for key in CITY_SHOPPING_GUIDE:
                if key in target_city or target_city in key:
                    shopping_data = CITY_SHOPPING_GUIDE[key]
                    break

        if not shopping_data:
            shopping_data = {
                "markets": [f"{city} City Center", "Main Street Promenade", "Central Plaza Mall", "Old Town Market"],
                "gems": [f"{city} Arts District", "Heritage Quarter", "local Boutiques Lane", "Crafts Bazaar"]
            }

        tops_count = max(2, duration)
        bottoms_count = max(1, duration // 2 + 1)
        socks_count = duration + 1

        packing_list = [
            f"{tops_count}x Breathable Tops",
            f"{bottoms_count}x Bottoms (Versatile)",
            f"{socks_count}x Underwear & Socks",
            "1x Light Jacket / Layer",
            "1x Comfortable Walking Shoes",
            "1x Evening Outfit",
            "Sleepwear Set",
            "Toiletries Kit",
            "Power Bank & Chargers",
            "Sunglasses & Accessories"
        ]

        if vibe == 'beach':
            packing_list.extend(["2x Swimwear", "Flip Flops / Sandals", "Beach Towel", "Sunscreen (SPF 50)"])
        elif vibe == 'mountain':
            packing_list.extend(["Thermal Layers", "Hiking Boots", "Wool Beanie", "Rain Jacket"])
        elif vibe == 'city':
            packing_list.extend(["Daypack / Tote", "Smart Casual Shoes", "Compact Umbrella"])

        return {
            "city": city,
            "days": duration,
            "weather_summary": f"Seasonally mild ({random.randint(18, 28)}°C)",
            "clothes_count": len(packing_list),
            "packing_list": packing_list,
            "must_visit": [{"name": m, "description": "Popular shopping destination", "type": "Market"} for m in shopping_data['markets'][:3]],
            "hidden_gems": [{"name": g, "description": "Curated local finds", "type": "Boutique"} for g in shopping_data['gems'][:3]]
        }

    @staticmethod
    async def audit_brand(brand: str) -> Dict[str, Any]:
        """Audits a brand's sustainability."""
        b = brand.lower()
        known_scores = {
            "patagonia": {"total": 92, "eco": 95, "labor": 90, "trans": 91,
                          "summary": "Industry leader in environmental responsibility and supply chain transparency."},
            "reformation": {"total": 85, "eco": 88, "labor": 80, "trans": 87,
                            "summary": "Strong focus on sustainable materials and carbon neutrality."},
            "zara": {"total": 45, "eco": 40, "labor": 50, "trans": 45,
                     "summary": "Fast fashion model raises concerns about waste and labor conditions."},
            "h&m": {"total": 52, "eco": 55, "labor": 50, "trans": 50,
                    "summary": "Has sustainability initiatives but volume is high."},
            "shein": {"total": 15, "eco": 10, "labor": 20, "trans": 15,
                      "summary": "Ultra-fast fashion with significant environmental and ethical concerns."},
            "everlane": {"total": 78, "eco": 75, "labor": 80, "trans": 80,
                         "summary": "Built on 'Radical Transparency' regarding costs and factories."},
            "levi's": {"total": 65, "eco": 70, "labor": 60, "trans": 65,
                       "summary": "Good water-saving initiatives, improving transparency."},
            "nike": {"total": 60, "eco": 65, "labor": 55, "trans": 60,
                     "summary": "Mixed performance; strong innovation but massive scale challenges."},
            "gucci": {"total": 70, "eco": 72, "labor": 75, "trans": 65,
                      "summary": "Luxury sector leader in going carbon neutral."},
            "uniqlo": {"total": 55, "eco": 50, "labor": 60, "trans": 55,
                       "summary": "Focus on durability, but transparency could improve."}
        }

        if b in known_scores:
            s = known_scores[b]
            return {
                "brand": brand,
                "total_score": s['total'],
                "summary": s['summary'],
                "eco_score": s['eco'],
                "labor_score": s['labor'],
                "trans_score": s['trans'],
                "sources": [{"uri": "#", "title": f"{brand} Sustainability Report"}]
            }

        seed = sum(ord(c) for c in b)
        random.seed(seed)

        base_score = random.randint(30, 80)
        eco = max(0, min(100, base_score + random.randint(-10, 10)))
        labor = max(0, min(100, base_score + random.randint(-10, 10)))
        trans = max(0, min(100, base_score + random.randint(-10, 10)))
        total = (eco + labor + trans) // 3

        summary = "AI Estimate: Moderate sustainability performance based on sector averages."
        if total > 70:
            summary = "AI Estimate: Likely has good sustainability practices."
        elif total < 40:
            summary = "AI Estimate: Potential risks in supply chain transparency."

        return {
            "brand": brand,
            "total_score": total,
            "summary": summary,
            "eco_score": eco,
            "labor_score": labor,
            "trans_score": trans,
            "sources": []
        }

    @staticmethod
    def weather_styling(city: str) -> Dict[str, Any]:
        """Provides weather-based styling advice."""
        return {
            "temp": 22,
            "condition": "Sunny",
            "outfit": {"top": "T-Shirt", "bottom": "Jeans", "layer": "None", "shoes": "Sneakers"},
            "advice": "Great weather for a casual day out."
        }


# Helper functions
def best_match_color_from_context(color_context: str) -> str:
    """Extract color from context string"""
    colors = ['Black', 'White', 'Navy', 'Beige', 'Denim', 'Gray', 'Olive', 'Camel', 'Red', 'Silver',
              'Burgundy', 'Emerald', 'Mustard', 'Coral', 'Lavender', 'Rust', 'Sage', 'Blush']
    for color in colors:
        if color.lower() in color_context.lower():
            return color
    return "complementary"


def item_to_dict(item_str: str) -> Dict[str, Any]:
    """Convert item string to dictionary for matching"""
    parts = item_str.split()
    color = parts[0] if parts else "Unknown"

    if "Top" in item_str:
        category = "Top"
    elif "Pants" in item_str or "Jeans" in item_str:
        category = "Pants"
    elif "Shorts" in item_str:
        category = "Shorts"
    elif "Skirt" in item_str:
        category = "Skirt"
    elif "Dress" in item_str:
        category = "Dress"
    elif "Jumpsuit" in item_str:
        category = "Jumpsuit"
    else:
        category = "Unknown"

    return {'color': color, 'category': category, 'fabric': 'Unknown'}


def suggestion_to_dict(suggestion: str, color: str, item_type: str) -> Dict[str, Any]:
    """Convert suggestion to dictionary for matching"""
    category_map = {
        "shoes": "Shoes",
        "jewelry/accessory": "Accessory",
        "bag": "Bag",
        "top": "Top",
        "bottom": "Pants",
        "outerwear": "Outerwear",
        "jacket/blazer": "Jacket"
    }

    category = category_map.get(item_type, "Unknown")
    return {'color': color, 'category': category, 'fabric': 'Unknown'}
