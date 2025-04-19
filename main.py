from image import (
    LocationService, SatelliteImagery, VegetationAnalyzer, ImageUtils,
    # For backward compatibility
    get_location, get_satellite_image, analyze_vegetation, save_image
)
import cv2
from PIL import Image
import numpy as np

def main():
    """Main application that uses the image.py functions to analyze vegetation"""
    print("🛰️ Satellite Vegetation Analysis 🌿")
    print("---------------------------------")
    
    # Get the user's location based on IP
    latitude, longitude, city, country = get_location()
    print(f"Analyzing area around {city}, {country}")
    
    # Download satellite imagery
    print("\nDownloading satellite image (~2km radius)...")
    img = get_satellite_image(latitude, longitude, zoom=15)  # Zoom 15 ≈ 2km radius view
    if img is None:
        print("❌ Could not retrieve satellite imagery")
        return
    
    # Save original image
    original_filename = save_image(img, latitude, longitude)
    
    # Analyze vegetation coverage with enhanced sensitivity
    print("\nAnalyzing vegetation coverage with enhanced sensitivity...")
    vegetation_percentage, vegetation_highlighted, visualization, analysis_results = analyze_vegetation(
        img,
        highlight_color=[0, 255, 0],  # Bright green in BGR
        highlight_intensity=0.7,      # Highlighting intensity
        threshold=110                 # Lower threshold to capture more vegetation
    )
    
    # Print vegetation analysis results
    print(f"\n🌿 Vegetation Coverage: {vegetation_percentage:.2f}%")
    
    # Save vegetation image
    vegetation_img = Image.fromarray(vegetation_highlighted[:, :, ::-1])  # BGR to RGB
    vegetation_filename = save_image(vegetation_img, latitude, longitude, suffix="_vegetation")
    
    # Save analysis visualization
    visualization_img = Image.fromarray(visualization)
    visualization_filename = save_image(visualization_img, latitude, longitude, suffix="_analysis")
    
    # Show images
    print("\nDisplaying analysis results... (press any key to advance)")
    cv2.imshow("Vegetation Highlighted", vegetation_highlighted)
    cv2.waitKey(0)
    
    # Convert visualization to BGR for OpenCV
    visualization_cv = visualization[:, :, ::-1].copy()
    cv2.imshow("Vegetation Analysis", visualization_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n📊 Analysis Summary:")
    print(f"  Location: {latitude:.6f}, {longitude:.6f} ({city}, {country})")
    print(f"  Vegetation Coverage: {vegetation_percentage:.2f}%")
    print(f"  GLI Threshold: {analysis_results['threshold_value']}")
    print(f"  Original Image: {original_filename}")
    print(f"  Highlighted Image: {vegetation_filename}")
    print(f"  Analysis Image: {visualization_filename}")

if __name__ == "__main__":
    main()
