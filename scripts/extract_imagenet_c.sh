#!/bin/bash
# Extract ImageNet-C dataset from tar.gz files
set -e

IMAGENET_C_ROOT="/home/user/zhoutianjian/DataSets/ImageNet-C"
RAW_DIR="${IMAGENET_C_ROOT}/raw"
EXTRACTED_DIR="${IMAGENET_C_ROOT}/extracted"

echo "🗂️  Extracting ImageNet-C Dataset"
echo "=================================="
echo "Raw dir: ${RAW_DIR}"
echo "Extract to: ${EXTRACTED_DIR}"
echo ""

# Create extracted directory
mkdir -p "${EXTRACTED_DIR}"

# Function to extract a tar.gz file
extract_tar() {
    local tar_file=$1
    local tar_name=$(basename "$tar_file" .tar.gz)
    
    echo "📦 Extracting ${tar_name}.tar.gz..."
    
    # Extract to extracted directory
    tar -xzf "$tar_file" -C "${EXTRACTED_DIR}" --checkpoint=10000 --checkpoint-action=dot
    
    echo ""
    echo "✅ ${tar_name} extracted"
}

# Extract each corruption category
cd "${RAW_DIR}"

# Only extract if not already extracted (check for any corruption type directory)
if [ ! -d "${EXTRACTED_DIR}/gaussian_noise" ] && [ ! -d "${EXTRACTED_DIR}/blur" ]; then
    echo "Starting extraction (this may take a while)..."
    echo ""
    
    # Extract noise corruptions (contains gaussian_noise, shot_noise, impulse_noise)
    if [ -f "noise.tar.gz" ]; then
        extract_tar "noise.tar.gz"
    fi
    
    # Extract blur corruptions (contains defocus_blur, motion_blur, zoom_blur, glass_blur)  
    if [ -f "blur.tar.gz" ]; then
        extract_tar "blur.tar.gz"
    fi
    
    # Extract weather corruptions (contains snow, frost, fog, brightness)
    if [ -f "weather.tar.gz" ]; then
        extract_tar "weather.tar.gz"
    fi
    
    # Extract digital corruptions (contains contrast, elastic_transform, pixelate, jpeg_compression)
    if [ -f "digital.tar.gz" ]; then
        extract_tar "digital.tar.gz"
    fi
    
    # Extract extra corruptions if present
    if [ -f "extra.tar.gz" ]; then
        extract_tar "extra.tar.gz"
    fi
    
    echo ""
    echo "🎉 All ImageNet-C data extracted successfully!"
else
    echo "✓ Data already extracted, skipping..."
fi

echo ""
echo "📊 Checking extracted data structure:"
ls -lh "${EXTRACTED_DIR}" | head -20

echo ""
echo "✅ ImageNet-C extraction complete!"

