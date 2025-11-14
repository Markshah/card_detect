#!/usr/bin/env python3
"""Test classification system to diagnose issues"""
import os
import cv2
import numpy as np
from rank_suit import classify_fullcard_anyrot, _load_templates

# Load env
try:
    from dotenv import load_dotenv
    if os.path.exists("env"):
        load_dotenv("env")
except:
    pass

print("=" * 60)
print("CLASSIFICATION DIAGNOSTIC TEST")
print("=" * 60)

# 1. Check templates
print("\n1. Checking templates...")
templates = _load_templates()
print(f"   ✓ Loaded {len(templates)} templates")

if not templates:
    print("   ✗ ERROR: No templates loaded!")
    exit(1)

# Check template sizes
sizes = set()
for code, templ in templates[:10]:
    sizes.add(templ.shape)
print(f"   Template sizes: {sizes}")

# Group by card
from collections import defaultdict
by_card = defaultdict(list)
for code, templ in templates:
    by_card[code].append(templ.shape)
print(f"   Cards with templates: {len(by_card)}")
print(f"   Cards with multiple templates: {sum(1 for v in by_card.values() if len(v) > 1)}")

# 2. Check thresholds
print("\n2. Checking thresholds...")
fullcard_thresh = float(os.getenv("FULLCARD_MIN_SCORE", "0.60"))
classify_thresh = float(os.getenv("CLASSIFY_THRESH", os.getenv("FULLCARD_MIN_SCORE", "0.55")))
print(f"   FULLCARD_MIN_SCORE: {fullcard_thresh}")
print(f"   CLASSIFY_THRESH: {classify_thresh}")
if abs(fullcard_thresh - classify_thresh) > 0.01:
    print(f"   ⚠ WARNING: Thresholds don't match!")

# 3. Test with actual templates
print("\n3. Testing classification with template images...")
templ_dir = os.getenv("CARD_FULL_TEMPL_DIR", "./templates")
test_cards = ["AS1", "KH1", "10D1", "2C1", "QS1"]  # Test a few cards

for card_name in test_cards:
    test_file = f"{templ_dir}/{card_name}.png"
    if os.path.exists(test_file):
        img = cv2.imread(test_file)
        if img is not None:
            result = classify_fullcard_anyrot(img)
            if len(result) == 3:
                code, score, rot = result
            else:
                code, score = result
                rot = None
            
            expected = card_name[:-1]  # Remove the "1"
            match = "✓" if code == expected else "✗"
            print(f"   {match} {card_name}: got {code}, score={score:.3f}, rot={rot}")
            if score < fullcard_thresh:
                print(f"      ⚠ Score below threshold {fullcard_thresh}!")
        else:
            print(f"   ✗ Could not read {test_file}")
    else:
        print(f"   - {test_file} not found")

# 4. Test with a synthetic card (white rectangle)
print("\n4. Testing with synthetic card (white rectangle)...")
synthetic = np.ones((700, 500, 3), dtype=np.uint8) * 255
result = classify_fullcard_anyrot(synthetic)
if len(result) == 3:
    code, score, rot = result
else:
    code, score = result
print(f"   Synthetic card: code={code}, score={score:.3f}")
if code is not None:
    print(f"   ⚠ WARNING: Synthetic card matched! This suggests threshold is too low.")

# 5. Test with a real template but slightly modified
print("\n5. Testing with modified template (brightness change)...")
if test_cards:
    test_file = f"{templ_dir}/{test_cards[0]}.png"
    if os.path.exists(test_file):
        img = cv2.imread(test_file)
        if img is not None:
            # Make it slightly brighter
            modified = cv2.add(img, 30)
            result = classify_fullcard_anyrot(modified)
            if len(result) == 3:
                code, score, rot = result
            else:
                code, score = result
            expected = test_cards[0][:-1]
            print(f"   Modified {test_cards[0]}: got {code}, score={score:.3f}")
            if code != expected:
                print(f"   ⚠ WARNING: Modified template didn't match correctly!")

# 6. Check warp size
print("\n6. Checking warp size configuration...")
warp_w = int(os.getenv("WARP_W", "400"))
warp_h = int(os.getenv("WARP_H", "560"))
print(f"   WARP_W: {warp_w}")
print(f"   WARP_H: {warp_h}")

# Get template size
if templates:
    templ_h, templ_w = templates[0][1].shape
    print(f"   Template size: {templ_w}x{templ_h} (width x height)")
    if warp_w != templ_w or warp_h != templ_h:
        print(f"   ⚠ WARNING: Warp size ({warp_w}x{warp_h}) doesn't match template size ({templ_w}x{templ_h})!")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

