"""
Extract individual panels from multi-panel plots.
Splits existing multi-panel figures into separate files for use in 2-column papers.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import os

def extract_panels_from_image(image_path, num_rows, num_cols, output_prefix):
    """
    Extract individual panels from a multi-panel plot image.
    
    Args:
        image_path: Path to the multi-panel PNG/PDF
        num_rows: Number of rows in the subplot grid
        num_cols: Number of columns in the subplot grid
        output_prefix: Prefix for output files (e.g., 'diversity_panel')
    """
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    
    # Calculate panel dimensions
    panel_height = height // num_rows
    panel_width = width // num_cols
    
    panel_files = []
    panel_num = 1
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Extract panel region
            y_start = row * panel_height
            y_end = (row + 1) * panel_height
            x_start = col * panel_width
            x_end = (col + 1) * panel_width
            
            panel = img_array[y_start:y_end, x_start:x_end]
            
            # Save as PNG
            panel_img = Image.fromarray(panel)
            png_filename = f'results/{output_prefix}_{panel_num}.png'
            panel_img.save(png_filename, dpi=(300, 300))
            
            # Save as PDF
            pdf_filename = f'results/{output_prefix}_{panel_num}.pdf'
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(panel)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            panel_files.append((png_filename, pdf_filename))
            print(f"Created {output_prefix}_{panel_num}: {png_filename}, {pdf_filename}")
            panel_num += 1
    
    return panel_files

def main():
    """Extract panels from all multi-panel plots."""
    
    print("Extracting panels from multi-panel plots...")
    print("=" * 60)
    
    # 1. diversity_vs_neutrality.png has 4 panels (2x2 grid)
    print("\n1. Extracting from diversity_vs_neutrality.png (4 panels)...")
    extract_panels_from_image(
        'results/diversity_vs_neutrality.png',
        num_rows=2,
        num_cols=2,
        output_prefix='diversity_panel'
    )
    
    # 2. equilibrium_comparison.png has 2 panels (1x2 grid)
    print("\n2. Extracting from equilibrium_comparison.png (2 panels)...")
    extract_panels_from_image(
        'results/equilibrium_comparison.png',
        num_rows=1,
        num_cols=2,
        output_prefix='equilibrium_panel'
    )
    
    # 3. time_series.png has 3 panels (3x1 grid)
    print("\n3. Extracting from time_series.png (3 panels)...")
    extract_panels_from_image(
        'results/time_series.png',
        num_rows=3,
        num_cols=1,
        output_prefix='time_panel'
    )
    
    print("\n" + "=" * 60)
    print("Panel extraction complete!")
    print("\nCreated files:")
    print("  - diversity_panel_1.png/pdf (Hamming vs neutrality)")
    print("  - diversity_panel_2.png/pdf (Entropy vs neutrality)")
    print("  - diversity_panel_3.png/pdf (Unique genotypes vs neutrality)")
    print("  - diversity_panel_4.png/pdf (Fitness vs neutrality)")
    print("  - equilibrium_panel_1.png/pdf (Equilibrium comparison)")
    print("  - equilibrium_panel_2.png/pdf (Equilibrium change)")
    print("  - time_panel_1.png/pdf (Hamming temporal)")
    print("  - time_panel_2.png/pdf (Entropy temporal)")
    print("  - time_panel_3.png/pdf (Fitness temporal)")
    
    # Remove old fig1-fig9 files
    print("\n" + "=" * 60)
    print("Removing old fig1-fig9 files...")
    for i in range(1, 10):
        for ext in ['png', 'pdf']:
            old_file = f'results/fig{i}_{["diversity_vs_neutrality", "entropy_vs_neutrality", "unique_genotypes", "fitness_vs_neutrality", "hamming_temporal", "entropy_temporal", "fitness_temporal", "equilibrium_comparison", "equilibrium_change"][i-1]}.{ext}'
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"  Removed {old_file}")

if __name__ == '__main__':
    main()
