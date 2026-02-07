"""
Convert all PNG plot files to PDF format for publication.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Directory containing PNG files
results_dir = Path('results')

# Find all PNG files
png_files = list(results_dir.glob('*.png'))

print(f"Found {len(png_files)} PNG files to convert:\n")

for png_file in png_files:
    # Read the PNG
    img = mpimg.imread(png_file)
    
    # Create figure with same aspect ratio
    dpi = 300
    height, width = img.shape[:2]
    figsize = width / dpi, height / dpi
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as PDF
    pdf_file = png_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✓ Converted: {png_file.name} → {pdf_file.name}")

print(f"\nAll PNG files converted to PDF in {results_dir}/")
