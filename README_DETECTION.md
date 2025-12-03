# Watermark Detection - Quick Guide

## 🎯 Three Methods Available

### 1. **OCR-Based (RECOMMENDED)** ⭐
Best for text watermarks, uses pretrained model

```bash
# Install
pip install easyocr

# Use
python detect_with_ocr.py -i sample.jpg
```

**Pros:**
- ✅ Most accurate for text
- ✅ Auto-detects diagonal/rotated text
- ✅ Supports Chinese characters
- ✅ No parameter tuning needed

**Cons:**
- Slower (10-30 seconds first time)
- Needs ~100MB model download

---

### 2. **Diagonal Pattern Detector**
For repeated diagonal watermarks

```bash
python detect_diagonal_watermark.py -i sample.jpg
```

**Pros:**
- Fast
- Good for repeated patterns
- Auto-detects angle

**Cons:**
- May miss scattered watermarks

---

### 3. **Simple OpenCV Detector**
Lightweight, no dependencies

```bash
python detect_and_mask.py -i sample.jpg --sensitivity ultra
```

**Pros:**
- Fastest
- No extra dependencies

**Cons:**
- Less accurate for low-contrast watermarks

---

## 🚀 Quick Start (Recommended Path)

### Step 1: Try OCR Method First

```bash
# Install EasyOCR (one time)
pip install easyocr

# Detect watermarks
python detect_with_ocr.py -i images/sample.jpg

# Check preview
open images/sample_mask_preview.jpg
```

### Step 2: Remove Watermarks

```bash
# Start IOPaint server
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint start --model=lama --device=cpu --port=8080

# Or use CLI (if available)
iopaint run --model=lama --device=cpu \
  --image=images/sample.jpg \
  --mask=images/sample_mask.png \
  --output=./output
```

---

## 📊 Method Comparison

| Method | Accuracy | Speed | Dependencies | Best For |
|--------|----------|-------|--------------|----------|
| **OCR** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | easyocr | Text watermarks |
| **Diagonal** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | none | Repeated patterns |
| **Simple** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | none | Quick tests |

---

## 💡 Tips

### For Your Image (diagonal Chinese watermarks)

**Best approach:**
```bash
python detect_with_ocr.py -i images/sample.jpg --expand 8
```

This will:
1. Detect all Chinese text (including rotated)
2. Generate precise mask
3. Expand by 8px for complete coverage

### If OCR is too slow

```bash
python detect_diagonal_watermark.py -i images/sample.jpg
```

### Adjust Coverage

```bash
# More coverage (catches more watermarks)
python detect_with_ocr.py -i sample.jpg --expand 10

# Less coverage (more precise)
python detect_with_ocr.py -i sample.jpg --expand 3
```

---

## 🔧 Troubleshooting

### EasyOCR installation fails?

```bash
# Use conda
conda install -c conda-forge easyocr

# Or system dependencies
# Ubuntu/Debian:
sudo apt-get install python3-dev

# macOS:
brew install python
```

### Mask coverage too low?

Try increasing expansion:
```bash
python detect_with_ocr.py -i sample.jpg --expand 10
```

### Mask coverage too high?

Try:
1. Use OCR method (most precise)
2. Manually edit mask in image editor
3. Lower expansion value

---

## 📁 Files

- `detect_with_ocr.py` - OCR-based (recommended)
- `detect_diagonal_watermark.py` - Diagonal pattern detector
- `detect_and_mask.py` - Simple OpenCV detector
- `detect_watermark.py` - Advanced detector (many options)

---

## ✅ Expected Results

**Good mask characteristics:**
- Coverage: 2-15% typically
- Only watermark text marked (not background/people)
- All watermark instances detected
- Clean edges around text

**Your ideal mask** (like `masks/01.JPG`):
- Diagonal stripes of text
- ~10-20% coverage
- No background marked

---

## 🎉 Complete Example

```bash
# 1. Install (one time)
pip install easyocr

# 2. Detect watermarks
python detect_with_ocr.py -i images/sample.jpg

# 3. Check preview
open images/sample_mask_preview.jpg

# 4. If good, remove watermarks
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint start --model=lama --device=cpu --port=8080
# Then open http://localhost:8080 and upload image + mask

# 5. Done!
```

---

## 📖 More Info

- `MASK_GENERATION_GUIDE.md` - Detailed guide
- `SIMPLE_USAGE.md` - Simple detector usage
- `WATERMARK_REMOVAL_GUIDE.md` - Removal workflows

---

**Recommendation:** Start with `detect_with_ocr.py` for best results! 🚀

