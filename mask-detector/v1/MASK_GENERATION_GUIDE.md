# Watermark Mask Generation Guide

## 🎯 Overview

This guide explains how to detect watermarks and generate precise masks for watermark removal using IOPaint.

## 🚀 Quick Start

### One-Step Detection + Mask Generation

```bash
# Detect watermarks and generate mask in one command (RECOMMENDED)
python detect_watermark.py -i sample.jpg \
  --generate-mask \
  --mask-preview \
  --visualize
```

This will create:
- `sample_detected.jpg` - Visualization showing all detected watermarks
- `sample_mask.png` - Binary mask for watermark removal
- `sample_mask_preview.jpg` - Overlay showing what will be masked

## 📖 Step-by-Step Workflow

### Step 1: Detect Watermarks

```bash
# Auto-detect with visualization
python detect_watermark.py -i sample.jpg --visualize
```

**What you'll see:**
- Colored bounding boxes for each detection method
- Precise contours following text shapes
- Statistics and coordinates

### Step 2: Generate Mask

#### Option A: Combined Mask (Recommended)

Generate one mask covering all detected watermarks:

```bash
python detect_watermark.py -i sample.jpg \
  --generate-mask \
  --mask-preview \
  --mask-method contour
```

**Parameters:**
- `--mask-method contour`: Follows text contours precisely (default: `auto`)
- `--mask-preview`: Shows overlay visualization
- `--mask-output`: Custom output path (default: `sample_mask.png`)

#### Option B: Separate Masks

Generate individual mask for each detected region:

```bash
python detect_watermark.py -i sample.jpg \
  --generate-mask \
  --separate-masks \
  --mask-output ./masks
```

This creates:
```
masks/
  ├── sample_mask_1_text.png
  ├── sample_mask_2_rotated.png
  ├── sample_mask_3_mser.png
  └── ...
```

### Step 3: Remove Watermarks

```bash
# Set environment variable (macOS requirement)
export KMP_DUPLICATE_LIB_OK=TRUE

# Remove watermarks using generated mask
iopaint run --model=lama --device=cpu \
  --image=sample.jpg \
  --mask=sample_mask.png \
  --output=./output
```

## 🎨 Mask Generation Methods

### 1. `auto` (Default)
Automatically selects the best method based on detection type:
- Text/Rotated/MSER → uses `contour` (precise)
- Others → uses `bbox` (fast)

### 2. `contour` / `text_trace`
**Most Precise** - Follows exact text contours

**How it works:**
- Multiple thresholding strategies (Otsu, Adaptive, Multi-level)
- Edge detection
- Contour tracing
- Morphological operations to connect text strokes

**Best for:**
- Text watermarks
- Semi-transparent watermarks
- Diagonal/rotated text
- When you want to avoid masking too much area

**Example:**
```bash
python detect_watermark.py -i sample.jpg \
  --generate-mask --mask-method contour --mask-preview
```

### 3. `bbox`
**Fast & Simple** - Rectangular mask

**Best for:**
- Logo watermarks
- Simple rectangular watermarks
- When speed is important
- When watermark is in a dedicated area

**Example:**
```bash
python detect_watermark.py -i sample.jpg \
  --generate-mask --mask-method bbox
```

## 📊 Understanding Mask Preview

The mask preview overlay shows:
- **Red areas**: What will be masked (removed)
- **Green contours**: Precise mask boundaries
- **Statistics**: Coverage percentage, pixel count

Example output:
```
Mask Coverage: 3.45%
Mask Pixels: 54,320
Image Size: 1920x1080
```

## 💡 Advanced Usage

### Interactive Mode + Mask Generation

For difficult-to-detect watermarks:

```bash
# 1. Manually select watermark region
python detect_watermark.py -i sample.jpg --interactive

# 2. Generate mask from selection
python detect_watermark.py -i sample.jpg \
  --interactive \
  --generate-mask \
  --mask-preview
```

### Batch Processing

```bash
# Process all images in a directory
for img in images/*.jpg; do
  python detect_watermark.py -i "$img" \
    --generate-mask \
    --mask-output "masks/$(basename $img .jpg)_mask.png"
done
```

### Custom Mask Post-Processing

```python
import cv2
import numpy as np

# Load generated mask
mask = cv2.imread('sample_mask.png', 0)

# Dilate to expand mask slightly
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(mask, kernel, iterations=2)

# Save modified mask
cv2.imwrite('sample_mask_dilated.png', mask_dilated)
```

## 🔧 Troubleshooting

### Mask is too large (covers too much)

**Solution 1:** Use `contour` method instead of `bbox`
```bash
--mask-method contour
```

**Solution 2:** Edit mask manually in image editor
- Open `sample_mask.png` in GIMP/Photoshop
- Erase unwanted areas (paint with black)
- Save as PNG

### Mask doesn't cover entire watermark

**Solution 1:** Mask is automatically dilated by 5 pixels. Increase it in code:
```python
# In generate_precise_mask(), change:
dilation_size: int = 5  # → 10
```

**Solution 2:** Post-process mask with additional dilation
```bash
# Use ImageMagick
convert sample_mask.png -morphology Dilate Disk:5 sample_mask_expanded.png
```

### Detection missed some watermarks

**Solutions:**
1. Use interactive mode to manually select
2. Check if watermark has enough contrast
3. Try adjusting detection parameters in code
4. Combine multiple detection runs

### Mask has noise/holes

The mask generation includes automatic noise removal, but if issues persist:

```python
import cv2

mask = cv2.imread('sample_mask.png', 0)

# Remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Fill holes
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imwrite('sample_mask_clean.png', mask)
```

## 📈 Performance Tips

1. **For large images**: Consider resizing before detection
   ```bash
   # Resize to max 2000px width
   convert sample.jpg -resize 2000x sample_small.jpg
   python detect_watermark.py -i sample_small.jpg --generate-mask
   ```

2. **For batch processing**: Disable preview to save time
   ```bash
   --generate-mask  # Without --mask-preview
   ```

3. **For high accuracy**: Use contour method with preview
   ```bash
   --mask-method contour --mask-preview
   ```

## 🎯 Best Practices

### For Semi-Transparent Watermarks
✅ Use `contour` method - follows text precisely
✅ Enable preview to verify coverage
✅ The enhanced preprocessing will help detection

### For Diagonal/Rotated Watermarks
✅ The detector covers -45° to +45° automatically
✅ Use `contour` method for best results
✅ Rotated text detection works on enhanced image

### For Repeated Pattern Watermarks
✅ Frequency analysis detects patterns automatically
✅ Combined mask covers all instances
✅ May generate large mask - consider separate masks

### For Multiple Watermarks
✅ Use combined mask for one-step removal
✅ Or use separate masks to selectively remove
✅ Merge separate masks if needed:
```bash
# Merge multiple masks
convert mask1.png mask2.png mask3.png \
  -compose lighten -composite merged_mask.png
```

## 🔗 Integration with IOPaint

### Basic Watermark Removal

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

iopaint run --model=lama --device=cpu \
  --image=sample.jpg \
  --mask=sample_mask.png \
  --output=./output
```

### Batch Watermark Removal

```bash
iopaint run --model=lama --device=cpu \
  --image=./images \
  --mask=./masks \
  --output=./output
```

### Using Different Models

```bash
# LaMa (default, best quality)
--model=lama

# LDM (good for large areas)
--model=ldm

# ZITS (fast)
--model=zits

# MAT (high quality)
--model=mat
```

## 📝 Complete Example

```bash
# 1. Detect and visualize
python detect_watermark.py -i photo.jpg --visualize

# 2. Generate mask with preview
python detect_watermark.py -i photo.jpg \
  --generate-mask \
  --mask-method contour \
  --mask-preview

# 3. Review generated files
ls -lh photo_*
# photo_detected.jpg      # Detection visualization
# photo_mask.png          # Binary mask
# photo_mask_preview.jpg  # Mask overlay

# 4. Remove watermark
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint run --model=lama --device=cpu \
  --image=photo.jpg \
  --mask=photo_mask.png \
  --output=./cleaned

# 5. Check result
open cleaned/photo.jpg
```

## 🆘 Getting Help

If you encounter issues:

1. **Check detection visualization** first
   ```bash
   python detect_watermark.py -i sample.jpg --visualize
   ```

2. **Try interactive mode** for manual control
   ```bash
   python detect_watermark.py -i sample.jpg --interactive
   ```

3. **Review mask preview** before removing
   ```bash
   --mask-preview
   ```

4. **Adjust mask method** if results aren't good
   - Too much masked → use `contour`
   - Too little masked → use `bbox` or increase dilation

5. **Check the guides**:
   - `WATERMARK_DETECTION_ENHANCED.md` - Detection improvements
   - `SMART_DETECTION_GUIDE.md` - Detection strategies
   - `WATERMARK_REMOVAL_GUIDE.md` - Removal workflows

## 🎉 Success Indicators

You know it's working well when:
- ✅ Mask preview shows watermark covered precisely
- ✅ Coverage percentage is reasonable (typically 1-10%)
- ✅ Contours follow text shapes, not rectangles
- ✅ No large areas of good content are masked
- ✅ Removal result looks natural with minimal artifacts

Happy watermark removing! 🚀

