import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import math
from datetime import datetime
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global configuration
MAPTILER_API_KEY = os.environ.get('MAPTILER_API_KEY')
GEOPY_USER_AGENT = os.environ.get('GEOPY_USER_AGENT')

# Module 1: Location Services
class LocationService:
    @staticmethod
    def get_location():
        """Get user's location based on IP address"""
        try:
            ip_location = requests.get("http://ip-api.com/json/").json()
            if ip_location["status"] == "success":
                latitude = ip_location["lat"]
                longitude = ip_location["lon"]
                city = ip_location["city"]
                country = ip_location["country"]
                print(f"📍 Your location: {latitude}, {longitude} ({city}, {country})")
                return latitude, longitude, city, country
            else:
                print("❌ Failed to get location information")
                return 0, 0, "Unknown", "Unknown"
        except Exception as e:
            print(f"❌ Error getting location: {e}")
            return 0, 0, "Unknown", "Unknown"

# Module 2: Satellite Imagery
class SatelliteImagery:
    @staticmethod
    def get_tile_coordinates(lat, lon, zoom):
        """Convert latitude and longitude to tile coordinates"""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
        return x, y

    @staticmethod
    def get_satellite_image(latitude, longitude, zoom=16):
        """Download satellite image for given coordinates
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            zoom: Zoom level (10 ≈ 30km diameter)
        Returns:
            PIL.Image or None if download failed
        """
        # Get tile coordinates
        tile_x, tile_y = SatelliteImagery.get_tile_coordinates(latitude, longitude, zoom)

        # Generate MapTiler satellite tile URL
        map_url = f"https://api.maptiler.com/maps/satellite/{zoom}/{tile_x}/{tile_y}.jpg?key={MAPTILER_API_KEY}"
        print(f"🌐 Map URL: {map_url}")

        # Download the image
        response = requests.get(map_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"❌ Failed to download image: HTTP {response.status_code}")
            print(f"  URL: {map_url}")
            print(f"  Response: {response.text}")
            return None

# Module 3: Vegetation Analysis
class VegetationAnalyzer:
    @staticmethod
    def analyze_vegetation(img, use_false_colors=False, highlight_color=[0, 255, 0], highlight_intensity=1.0, threshold=120):
        """Analyze vegetation coverage using Green Leaf Index (GLI) with enhanced sensitivity
        Args:
            img: PIL Image object
            use_false_colors: Whether to use enhanced false color technique (ignored, kept for compatibility)
            highlight_color: BGR color for highlighting vegetation [B,G,R]
            highlight_intensity: Intensity of highlighting (0.0-1.0)
            threshold: Threshold value for GLI (0-255) - lower values capture more vegetation
        Returns:
            float: Vegetation coverage percentage
            np.array: Image with vegetation highlighted
            dict: Dictionary containing separate visualization images
            dict: Vegetation analysis results
        """
        # Convert PIL image to OpenCV format (RGB to BGR)
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Convert to RGB for analysis
        img_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        
        # Enhancement: Apply slight Gaussian blur to reduce noise
        img_blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
        
        # Split channels and convert to float to avoid overflow
        R = img_blur[:, :, 0].astype(float)
        G = img_blur[:, :, 1].astype(float)
        B = img_blur[:, :, 2].astype(float)
        
        # Calculate Green Leaf Index (GLI)
        numerator = (2 * G - R - B)
        denominator = (2 * G + R + B)
        GLI = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        
        # Normalize GLI to 0-255 range for visualization and thresholding
        GLI_norm = cv2.normalize(GLI, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Enhancement: Apply adaptive thresholding to better handle different lighting conditions
        # First attempt with GLI threshold
        _, vegetation_mask = cv2.threshold(GLI_norm, threshold, 255, cv2.THRESH_BINARY)
        
        # Enhancement: Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Enhancement: Use additional color-based vegetation indices
        # Extract additional vegetation using HSV color space for very light green areas
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 10, 50])  # More permissive lower bound to catch light green
        upper_green = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks (GLI + HSV)
        combined_mask = cv2.bitwise_or(vegetation_mask, hsv_mask)
        
        # Create vegetation highlighted image
        vegetation_simple = img_rgb.copy()
        vegetation_simple[combined_mask == 0] = 0
        
        # Method 2: Blended highlighting with original image
        vegetation_highlighted = open_cv_image.copy()
        highlight_mask = combined_mask > 0
        
        # Apply highlight with intensity control
        for i in range(3):  # For each color channel
            # Get original channel values where mask is True
            if np.any(highlight_mask):  # Check if mask has any True values
                original_values = vegetation_highlighted[highlight_mask, i].astype(float)
                
                # Calculate blended values
                blended_values = ((1-highlight_intensity) * original_values + 
                                highlight_intensity * highlight_color[i]).astype(np.uint8)
                
                # Apply the blended values back to the image
                vegetation_highlighted[highlight_mask, i] = blended_values
        
        # Calculate vegetation coverage
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        green_pixels = cv2.countNonZero(combined_mask)
        vegetation_percentage = (green_pixels / total_pixels) * 100
        
        # Create separate visualization images
        visualization_images = VegetationAnalyzer.create_vegetation_visualization(img_rgb, GLI_norm, combined_mask, vegetation_simple)
        
        # Create analysis results dictionary
        analysis_results = {
            'gli': GLI,
            'gli_normalized': GLI_norm,
            'vegetation_mask': combined_mask,
            'vegetation_percentage': vegetation_percentage,
            'threshold_value': threshold
        }
        
        return vegetation_percentage, vegetation_highlighted, visualization_images, analysis_results

    @staticmethod
    def create_vegetation_visualization(original_rgb, gli_normalized, vegetation_mask, vegetation_rgb):
        """Create separate visualization images without headers
        
        Args:
            original_rgb: Original RGB image
            gli_normalized: Normalized GLI values
            vegetation_mask: Binary vegetation mask
            vegetation_rgb: RGB image with vegetation highlighted
            
        Returns:
            dict: Dictionary containing separate RGB images
        """
        # Process original image
        original_img = original_rgb.copy()
        
        # Process vegetation mask - convert to RGB for consistent handling
        mask_rgb = cv2.cvtColor(vegetation_mask, cv2.COLOR_GRAY2RGB)
        
        # Process highlighted vegetation
        highlighted_img = vegetation_rgb.copy()
        
        # Package images into a dictionary
        visualization_images = {
            'original': original_img,
            'mask': mask_rgb,
            'highlighted': highlighted_img
        }
        
        return visualization_images

# Module 4: Image Utilities
class ImageUtils:
    @staticmethod
    def save_image(img, latitude, longitude, suffix=""):
        """Save image to file with location and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"satellite_{latitude:.4f}_{longitude:.4f}{suffix}_{timestamp}.jpg"
        img.save(filename)
        print(f"✅ Image saved as {filename}")
        return filename

# For backward compatibility, maintain original function names as wrappers
def get_location():
    return LocationService.get_location()

def get_satellite_image(latitude, longitude, zoom=16):
    return SatelliteImagery.get_satellite_image(latitude, longitude, zoom)

def analyze_vegetation(img, use_false_colors=False, highlight_color=[0, 255, 0], highlight_intensity=1.0, threshold=120):
    return VegetationAnalyzer.analyze_vegetation(img, use_false_colors, highlight_color, highlight_intensity, threshold)

def save_image(img, latitude, longitude, suffix=""):
    return ImageUtils.save_image(img, latitude, longitude, suffix)
