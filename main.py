import cv2
import numpy as np
import requests
import re
import math
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import easyocr
from Levenshtein import distance as lev_distance

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])


# =========================================
# ADVANCED CV2 PREPROCESSING
# =========================================
def preprocess_image(image_path: str) -> tuple:
    """
    Advanced preprocessing with multiple CV2 techniques
    Returns: (processed_img, original_img)
    """
    img = cv2.imread(image_path)
    original = img.copy()
    
    # Step 1: Resize for better processing
    height, width = img.shape[:2]
    if width < 1000:
        scale = 1000 / width
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Bilateral filter (preserves edges, reduces noise)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Step 5: Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Opening: removes small noise
    opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Closing: fills small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Step 6: Edge detection
    edges = cv2.Canny(closed, 50, 150)
    
    # Step 7: Dilation to connect text regions
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=2)
    
    # Step 8: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 9: Combine adaptive threshold with dilated edges
    combined = cv2.bitwise_or(adaptive_thresh, dilated_edges)
    
    # Step 10: Erosion to separate touching characters
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final = cv2.erode(combined, kernel_erode, iterations=1)
    
    return final, original


def extract_text_regions(image_path: str) -> list:
    """
    Extract contours and find text regions using CV2
    Returns list of cropped text regions with detected characters
    """
    processed, original = preprocess_image(image_path)
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    text_regions = []
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio and size (text characteristics)
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        
        # Text regions typically have specific proportions
        if 0.2 < aspect_ratio < 15 and 100 < area < 200000:
            # Extract region
            region = original[y:y+h, x:x+w]
            text_regions.append({
                "crop": region,
                "bbox": (x, y, w, h),
                "area": area
            })
    
    # Sort by area (larger = likely more important)
    text_regions.sort(key=lambda r: r["area"], reverse=True)
    
    return text_regions[:10]


def extract_text_features(image: np.ndarray) -> list:
    """
    Extract character blobs using CV2 blob detection
    Returns list of detected text blobs
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Blob detection parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by area
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000
    
    # Filter by circularity (text blobs have lower circularity)
    params.filterByCircularity = False
    
    # Filter by color
    params.filterByColor = False
    params.blobColor = 255  # White blobs
    
    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(binary)
    
    return keypoints


def ocr_easyocr(image_path: str) -> list:
    """
    EasyOCR-based text extraction with CV2 preprocessing
    Returns actual readable text
    """
    img = cv2.imread(image_path)
    original = img.copy()
    
    candidates = []
    
    # Read text with EasyOCR
    try:
        results = reader.readtext(img)
        
        # Extract all detected text with confidence
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            
            # Only keep high-confidence detections
            if confidence > 0.4 and len(text.strip()) > 1:
                candidates.append(text.strip())
    except Exception as e:
        print(f"⚠️ EasyOCR error: {e}")
    
    # Also try on preprocessed image
    processed, _ = preprocess_image(image_path)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    try:
        results_processed = reader.readtext(processed_rgb)
        
        for detection in results_processed:
            text = detection[1]
            confidence = detection[2]
            
            if confidence > 0.4 and len(text.strip()) > 1:
                candidates.append(text.strip())
    except Exception as e:
        print(f"⚠️ EasyOCR preprocessed error: {e}")
    
    # Try with different contrast enhancement
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    try:
        results_enhanced = reader.readtext(enhanced_rgb)
        
        for detection in results_enhanced:
            text = detection[1]
            confidence = detection[2]
            
            if confidence > 0.4 and len(text.strip()) > 1:
                candidates.append(text.strip())
    except Exception as e:
        print(f"⚠️ EasyOCR enhanced error: {e}")
    
    return candidates


def extract_shop_name(image_path: str) -> list:
    """
    Extract shop names using EasyOCR + CV2 preprocessing
    Cleans up OCR artifacts and combines fragments
    """
    # Get actual text from EasyOCR
    candidates = ocr_easyocr(image_path)
    
    if not candidates:
        print("❌ No text detected by EasyOCR")
        return []
    
    # Clean up OCR artifacts
    cleaned_candidates = []
    for text in candidates:
        # Remove common OCR artifacts
        text = text.strip()
        text = text.rstrip('"\'.,;:!?')  # Remove trailing punctuation
        text = text.lstrip('"\'.,;:!?')   # Remove leading punctuation
        text = re.sub(r'["|""]', '', text)  # Remove quote marks
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        
        # Keep only meaningful text (at least 2 chars, not all numbers/symbols)
        if len(text) >= 2 and any(c.isalpha() for c in text):
            cleaned_candidates.append(text)
    
    print(f"📝 Raw OCR: {candidates}")
    print(f"🧹 Cleaned: {cleaned_candidates}")
    
    # Try to merge fragments that might be parts of same word
    # E.g., "DA" + "Mart" could be "DAMart" 
    merged_candidates = merge_text_fragments(cleaned_candidates)
    
    # Regex patterns for shop names
    shop_patterns = []
    full_text = " ".join(merged_candidates)
    
    # Pattern 1: Uppercase phrases (LOYAL WORLD RETAIL MART)
    uppercase = re.findall(r"[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*", full_text)
    shop_patterns.extend(uppercase)
    
    # Pattern 2: Mixed case (DMart, DaMart, DAMart)
    mixedcase = re.findall(r"[A-Z][a-z]*[A-Z][a-z]*", full_text)
    shop_patterns.extend(mixedcase)
    
    # Combine all candidates
    all_names = list(set(merged_candidates + shop_patterns))
    
    # Filter: remove very short strings, keep those with substance
    all_names = [n.strip() for n in all_names if len(n.strip()) >= 2]
    
    # Remove duplicates (case-insensitive) AND similar variants
    unique_names = []
    seen = set()
    for name in all_names:
        name_lower = name.lower()
        
        # Skip if already seen exact
        if name_lower in seen:
            continue
        
        # Skip if it's a fragment of something already added
        is_fragment = False
        for existing in unique_names:
            existing_lower = existing.lower()
            # If current is substring of existing or vice versa, keep the longer one
            if name_lower in existing_lower or existing_lower in name_lower:
                if len(name) > len(existing):
                    unique_names.remove(existing)
                else:
                    is_fragment = True
                    break
        
        if not is_fragment:
            unique_names.append(name)
            seen.add(name_lower)
    
    # Sort by length (longer = more specific)
    unique_names.sort(key=len, reverse=True)
    
    print(f"🏪 Final candidates: {unique_names[:10]}")
    return unique_names[:10]


def merge_text_fragments(fragments: list) -> list:
    """
    Try to merge fragments that are likely parts of the same word
    E.g., ["DA", "Mart"] -> ["DAMart", "DA", "Mart"]
    """
    merged = fragments.copy()
    
    # For each pair of consecutive fragments, try merging
    for i in range(len(fragments) - 1):
        frag1 = fragments[i]
        frag2 = fragments[i + 1]
        
        # If both are short and together form a common word pattern
        if len(frag1) <= 3 and len(frag2) <= 6:
            merged_word = frag1 + frag2
            # Check if this looks like a shop name (has vowels, reasonable length)
            if len(merged_word) >= 3 and len(merged_word) <= 20:
                merged.append(merged_word)
        
        # Also try with space
        if len(frag1) >= 2 and len(frag2) >= 2:
            spaced = frag1 + " " + frag2
            if len(spaced) >= 4 and len(spaced) <= 25:
                merged.append(spaced)
    
    return merged


# =========================================
# DISTANCE CALCULATION & WEIGHTING
# =========================================
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters"""
    R = 6371000
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def distance_score(distance_m: float, max_distance: float = 500) -> float:
    """Score based on distance (0-1)"""
    if distance_m > max_distance:
        return 0.1
    return math.exp(-distance_m / 200)


# =========================================
# SMART FUZZY MATCHING
# =========================================
def fuzzy_match_name(detected: str, osm_name: str) -> float:
    """Smart matching with typo tolerance"""
    d_lower = detected.lower().strip()
    o_lower = osm_name.lower().strip()
    
    # Strategy 1: Exact match (huge bonus)
    if d_lower == o_lower:
        return 1.0
    
    # Strategy 2: One is substring of the other
    if d_lower in o_lower or o_lower in d_lower:
        overlap_len = len(d_lower) if d_lower in o_lower else len(o_lower)
        longer_len = max(len(d_lower), len(o_lower))
        substring_score = 0.85 + (0.15 * (overlap_len / longer_len))
        return min(substring_score, 1.0)
    
    # Strategy 3: Levenshtein distance (typo tolerance) - NEW!
    lev_dist = lev_distance(d_lower, o_lower)
    max_len = max(len(d_lower), len(o_lower))
    
    # If very few edits needed, it's probably the same thing
    if lev_dist <= 2:  # 1-2 character differences
        lev_score = 0.90
    elif lev_dist <= max_len * 0.3:  # Up to 30% different
        lev_score = 0.80 - (lev_dist / max_len * 0.1)
    else:
        lev_score = 0
    
    # Strategy 4: Token set ratio (word order)
    token_score = fuzz.token_set_ratio(d_lower, o_lower) / 100.0
    
    # Strategy 5: Partial ratio (substrings)
    partial_score = fuzz.partial_ratio(d_lower, o_lower) / 100.0
    
    # Strategy 6: Common words
    d_words = set(d_lower.split())
    o_words = set(o_lower.split())
    common_words = d_words & o_words
    
    word_bonus = 0
    if common_words:
        word_bonus = len(common_words) / max(len(d_words), len(o_words)) * 0.2
    
    # REWEIGHTED: Levenshtein (typo catching) is now important!
    final_score = (
        lev_score * 0.40 +           # TYPO MATCHING
        token_score * 0.25 +         # Word order
        partial_score * 0.25 +       # Substring matching
        word_bonus * 0.10            # Common words
    )
    
    return min(final_score, 1.0)


# =========================================
# OSM QUERY
# =========================================
def query_osm(lat: float, lon: float, radius: float = 800, shop_type: str = "all"):
    """Fetch nearby places from OSM - can search for ANY type of shop"""
    
    # Define different search strategies based on detected text
    queries = {
        "all": f"""
        [out:json];
        (
          node(around:{radius},{lat},{lon})["name"];
          way(around:{radius},{lat},{lon})["name"];
          node(around:{radius},{lat},{lon})["amenity"];
          way(around:{radius},{lat},{lon})["amenity"];
          node(around:{radius},{lat},{lon})["shop"];
          way(around:{radius},{lat},{lon})["shop"];
          node(around:{radius},{lat},{lon})["healthcare"];
          way(around:{radius},{lat},{lon})["healthcare"];
        );
        out tags geom;
        """,
        
        "retail": f"""
        [out:json];
        (
          node(around:{radius},{lat},{lon})["shop"];
          way(around:{radius},{lat},{lon})["shop"];
          node(around:{radius},{lat},{lon})["amenity"~"supermarket|convenience|mall|marketplace"];
          way(around:{radius},{lat},{lon})["amenity"~"supermarket|convenience|mall|marketplace"];
        );
        out tags geom;
        """,
        
        "healthcare": f"""
        [out:json];
        (
          node(around:{radius},{lat},{lon})[healthcare];
          way(around:{radius},{lat},{lon})[healthcare];
          node(around:{radius},{lat},{lon})["amenity"~"clinic|doctors|hospital|pharmacy"];
          way(around:{radius},{lat},{lon})["amenity"~"clinic|doctors|hospital|pharmacy"];
        );
        out tags geom;
        """,
        
        "food": f"""
        [out:json];
        (
          node(around:{radius},{lat},{lon})["amenity"~"restaurant|cafe|bar|fast_food|food_court"];
          way(around:{radius},{lat},{lon})["amenity"~"restaurant|cafe|bar|fast_food|food_court"];
          node(around:{radius},{lat},{lon})["shop"~"bakery|butcher|supermarket|convenience"];
          way(around:{radius},{lat},{lon})["shop"~"bakery|butcher|supermarket|convenience"];
        );
        out tags geom;
        """
    }
    
    query = queries.get(shop_type, queries["all"])
    
    url = "https://overpass-api.de/api/interpreter"
    try:
        response = requests.post(url, data=query.encode("utf-8"), timeout=15)
        return response.json().get("elements", [])
    except Exception as e:
        print(f"❌ OSM Query Failed: {e}")
        return []


def detect_shop_type(text_candidates: list) -> str:
    """
    Detect what type of shop based on OCR'd text
    Returns: "retail", "healthcare", "food", or "all"
    """
    text_lower = " ".join(text_candidates).lower()
    
    # Healthcare keywords
    healthcare_keywords = ["clinic", "hospital", "doctor", "pharmacy", "medical", "health", "dental", "ivf", "care", "diagnostic"]
    
    # Retail keywords
    retail_keywords = ["mart", "store", "shop", "retail", "dmart", "bigbasket", "retail", "supermarket", "mall", "center"]
    
    # Food keywords
    food_keywords = ["restaurant", "cafe", "coffee", "food", "pizza", "burger", "bakery", "kitchen", "hotel"]
    
    healthcare_count = sum(1 for kw in healthcare_keywords if kw in text_lower)
    retail_count = sum(1 for kw in retail_keywords if kw in text_lower)
    food_count = sum(1 for kw in food_keywords if kw in text_lower)
    
    if healthcare_count > retail_count and healthcare_count > food_count:
        return "healthcare"
    elif retail_count > food_count:
        return "retail"
    elif food_count > 0:
        return "food"
    else:
        return "all"


# =========================================
# SCORING FUNCTION
# =========================================
def calculate_match_score(detected: str, osm_element: dict, user_lat: float, user_lon: float) -> tuple:
    """Comprehensive scoring - NAME MATCH IS CRITICAL"""
    osm_name = osm_element["tags"].get("name", "")
    
    if not osm_name:
        return (0, 999999, "")
    
    if "center" in osm_element:
        osm_lat, osm_lon = osm_element["center"]["lat"], osm_element["center"]["lon"]
    elif "lat" in osm_element:
        osm_lat, osm_lon = osm_element["lat"], osm_element["lon"]
    else:
        return (0, 999999, "")
    
    # Name matching is CRITICAL - must be strong
    name_score = fuzzy_match_name(detected, osm_name)
    
    distance_m = haversine_distance(user_lat, user_lon, osm_lat, osm_lon)
    distance_score_val = distance_score(distance_m)
    
    # Only apply bonuses if name score is decent
    type_bonus = 0
    keyword_bonus = 0
    
    if name_score > 0.4:  # Only apply bonuses for reasonable matches
        healthcare_type = osm_element["tags"].get("healthcare", "").lower()
        amenity = osm_element["tags"].get("amenity", "").lower()
        shop = osm_element["tags"].get("shop", "").lower()
        
        if "clinic" in amenity or "clinic" in healthcare_type:
            type_bonus = 0.08
        if "polyclinic" in healthcare_type:
            type_bonus = 0.12
        if "supermarket" in shop or "supermarket" in amenity:
            type_bonus = 0.08
        
        keywords = ["dmart", "mart", "medical", "diagnostic", "doctor", "anand"]
        for kw in keywords:
            if kw in osm_name.lower() and kw in detected.lower():
                keyword_bonus = 0.15
                break
    
    # REWEIGHTED: Name is now 65% (was 35%), distance is 25% (was 55%)
    final_score = (
        name_score * 0.65 +       # NAME IS PRIMARY
        distance_score_val * 0.25 +  # Distance is secondary
        type_bonus * 0.05 +
        keyword_bonus * 0.05
    )
    
    return final_score, distance_m, osm_name


# =========================================
# MAIN PIPELINE
# =========================================
def shop_insights(image_path: str, lat: float, lon: float):
    """Main function with EasyOCR + flexible shop type detection"""
    
    # Step 1: Extract candidates
    detected_candidates = extract_shop_name(image_path)
    if not detected_candidates:
        print("❌ No text detected")
        return
    
    # Step 2: Detect shop type from text
    shop_type = detect_shop_type(detected_candidates)
    print(f"\n🏷️ Detected shop type: {shop_type}")
    
    # Step 3: Query OSM with appropriate filter
    print(f"\n🔍 Searching for {shop_type} shops within 800m of ({lat}, {lon})...")
    elements = query_osm(lat, lon, radius=800, shop_type=shop_type)
    
    if not elements:
        print("❌ No OSM data found nearby.")
        return
    
    print(f"📍 Found {len(elements)} places in OSM")
    
    # Step 3: Score all combinations
    best_matches = []
    
    for detected in detected_candidates:
        for element in elements:
            score, distance_m, osm_name = calculate_match_score(detected, element, lat, lon)
            
            if score > 0.3:
                best_matches.append({
                    "score": score,
                    "detected": detected,
                    "osm_name": osm_name,
                    "distance_m": distance_m,
                    "element": element
                })
    
    if not best_matches:
        print("No credible matches found (threshold: 0.3)")
        return
    
    best_matches.sort(key=lambda x: x["score"], reverse=True)
    
    # Show results
    print("\n" + "="*60)
    print("🏆 TOP MATCHES")
    print("="*60)
    
    for i, match in enumerate(best_matches[:3], 1):
        print(f"\n#{i} MATCH (Score: {match['score']:.3f})")
        print(f"  Detected:  {match['detected']}")
        print(f"  OSM Name:  {match['osm_name']}")
        print(f"  Distance:  {match['distance_m']:.0f}m")
        print(f"  ---")
        
        for k, v in match["element"]["tags"].items():
            print(f"  {k}: {v}")
    
    winner = best_matches[0]
    print("\n" + "="*60)
    print(f"BEST MATCH: {winner['osm_name']}")
    print(f"   Score: {winner['score']:.3f} | Distance: {winner['distance_m']:.0f}m")
    print("="*60)


if __name__ == "__main__":
    shop_insights("sample.jpg", 12.936713789462031, 77.7487147310)
    #shop_insights("akk.jpg", 12.9556, 77.7286)

# Install: pip install opencv-python fuzzywuzzy python-Levenshtein requests

# Pure CV2 OCR - no external ML models needed!
