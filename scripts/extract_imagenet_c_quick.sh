#!/bin/bash
# Quick extract for testing - only extracts noise and blur corruptions
set -e

IMAGENET_C_ROOT="/home/user/zhoutianjian/DataSets/ImageNet-C"
RAW_DIR="${IMAGENET_C_ROOT}/raw"
EXTRACTED_DIR="${IMAGENET_C_ROOT}/extracted"

echo "🗂️  Quick Extract: ImageNet-C Noise & Blur Corruptions"
echo "======================================================="
echo "Raw dir: ${RAW_DIR}"
echo "Extract to: ${EXTRACTED_DIR}"
echo ""

# Create extracted directory
mkdir -p "${EXTRACTED_DIR}"

cd "${RAW_DIR}"

# Extract noise corruptions (contains gaussian_noise, shot_noise, impulse_noise)
if [ ! -d "${EXTRACTED_DIR}/gaussian_noise" ] && [ -f "noise.tar.gz" ]; then
    echo "📦 Extracting noise.tar.gz (contains gaussian_noise)..."
    tar -xzf "noise.tar.gz" -C "${EXTRACTED_DIR}"
    echo "✅ Noise corruptions extracted"
else
    echo "✓ Noise corruptions already extracted"
fi

# Extract blur corruptions (contains defocus_blur, motion_blur, zoom_blur, glass_blur)  
if [ ! -d "${EXTRACTED_DIR}/defocus_blur" ] && [ -f "blur.tar.gz" ]; then
    echo "📦 Extracting blur.tar.gz (contains defocus_blur)..."
    tar -xzf "blur.tar.gz" -C "${EXTRACTED_DIR}"
    echo "✅ Blur corruptions extracted"
else
    echo "✓ Blur corruptions already extracted"
fi

echo ""
echo "📊 Checking extracted corruption types:"
ls -lh "${EXTRACTED_DIR}" 2>/dev/null || echo "No data extracted yet"

echo ""
echo "🔍 Checking gaussian_noise structure (if exists):"
if [ -d "${EXTRACTED_DIR}/gaussian_noise" ]; then
    echo "Severity levels:"
    ls "${EXTRACTED_DIR}/gaussian_noise/" 2>/dev/null || echo "Not found"
    
    if [ -d "${EXTRACTED_DIR}/gaussian_noise/1" ]; then
        echo ""
        echo "Sample images in severity 1:"
        find "${EXTRACTED_DIR}/gaussian_noise/1" -type f | head -5
    fi
fi

echo ""
echo "✅ Quick extraction complete!"

