import os
import uuid
import json
import tempfile
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from image import (
    LocationService, SatelliteImagery, VegetationAnalyzer, ImageUtils,
    get_location, get_satellite_image, analyze_vegetation
)
from PIL import Image
import cv2
from weather import get_combined_weather_data
from pollen import get_combined_pollen_data, get_latest_pollen_data, get_forecast_pollen_data
from llm_report import generate_agriculture_report, answer_agriculture_query

# Create Flask app
app = Flask(__name__)

# Configure app from environment variables
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'default-secret-key'),
    MAX_CONTENT_LENGTH=int(os.environ.get('MAX_CONTENT_LENGTH', 16777216)),
    MAX_AGE_HOURS=int(os.environ.get('MAX_AGE_HOURS', 1)),
    ALLOWED_EXTENSIONS=set(os.environ.get('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,gif').split(','))
)

# Configure temporary file storage
TEMP_FOLDER = os.environ.get('TEMP_FOLDER', tempfile.gettempdir())
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(TEMP_FOLDER, 'datacosmos'))
DATA_FOLDER = os.environ.get('DATA_FOLDER', os.path.join(UPLOAD_FOLDER, 'data'))  # For analysis data
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MAX_AGE_HOURS = 1  # Images older than this will be deleted

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ANALYSES'] = []  # Store recent analyses in memory

# Clean up old files function
def cleanup_old_files():
    """Remove image files older than MAX_AGE_HOURS"""
    now = datetime.now()
    count = 0
    
    for dirname in [app.config['UPLOAD_FOLDER'], app.config['DATA_FOLDER']]:
        for filename in os.listdir(dirname):
            filepath = os.path.join(dirname, filename)
            try:
                # Skip if not a file
                if not os.path.isfile(filepath):
                    continue
                    
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                age = now - file_time
                
                # If file is older than MAX_AGE_HOURS, delete it
                if age > timedelta(hours=MAX_AGE_HOURS):
                    os.remove(filepath)
                    count += 1
            except Exception as e:
                print(f"Error while cleaning up file {filepath}: {e}")
    
    # Also clean up old analyses from memory
    if len(app.config['ANALYSES']) > 0:
        new_analyses = []
        for analysis in app.config['ANALYSES']:
            created_time = datetime.fromisoformat(analysis['analysis_date'])
            age = now - created_time
            if age <= timedelta(hours=MAX_AGE_HOURS):
                new_analyses.append(analysis)
        
        app.config['ANALYSES'] = new_analyses
    
    return count

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to save temporary image
def save_temp_image(img, prefix="img", suffix=""):
    """Save PIL Image to temporary file and return filename"""
    # Generate a unique filename
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{unique_id}{suffix}_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the image
    img.save(filepath)
    
    return filename

# Helper function to extract RGB channel data for analysis
def extract_rgb_data(img_array, sample_size=10000):
    """Extract RGB data for analysis from an image array with sampling"""
    # Convert to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        if isinstance(img_array[0, 0, 0], np.uint8):
            # Already RGB format
            img_rgb = img_array
        else:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        # Not an RGB image
        return None
    
    # Sample pixels if the image is large
    h, w = img_rgb.shape[:2]
    total_pixels = h * w
    
    if total_pixels > sample_size:
        # Random sampling
        indices = np.random.choice(total_pixels, sample_size, replace=False)
        pixels = img_rgb.reshape(-1, 3)[indices]
    else:
        pixels = img_rgb.reshape(-1, 3)
    
    # Create DataFrame
    df = pd.DataFrame({
        'r': pixels[:, 0],
        'g': pixels[:, 1],
        'b': pixels[:, 2]
    })
    
    return df

# Helper function to generate historical data
def generate_historical_data(current_percentage):
    """Generate simulated historical vegetation data"""
    base = max(5, current_percentage * 0.7)  # Base value
    amplitude = current_percentage * 0.3  # Seasonal amplitude
    
    # Generate yearly cycle with some randomness
    data = []
    for i in range(12):
        # Seasonal pattern with peak in summer (months 4-8)
        season_factor = 0.5 + 0.5 * np.sin((i - 1) * np.pi / 6)
        value = base + amplitude * season_factor
        
        # Add some noise
        noise = random.uniform(-5, 5)
        value = max(0, min(100, value + noise))
        
        data.append(round(value, 2))
    
    return data

def get_fallback_weather_data():
    """Provide fallback weather data from Manama, Bahrain"""
    return [{
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Avg Temperature (°C)': 32,
        'Avg Humidity (%)': 74,
        'Total Precipitation (mm)': 0.2
    }]

@app.route('/')
def index():
    """Home page - only My Location option"""
    # Clean up old files on page load
    deleted_count = cleanup_old_files()
    if (deleted_count > 0):
        print(f"Cleaned up {deleted_count} old files")
        
    return render_template('index.html')

# Analyze by coordinates
@app.route('/analyze-coordinates', methods=['POST'])
def analyze_coordinates():
    try:
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        zoom = int(request.form.get('zoom', 15))
        
        # Try to get location info
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="biopixel")
            location = geolocator.reverse(f"{latitude}, {longitude}", language='en')
            if location and location.raw.get('address'):
                address = location.raw['address']
                city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                country = address.get('country', 'Unknown')
            else:
                city = "Unknown"
                country = "Unknown"
        except Exception as loc_error:
            print(f"Location error: {loc_error}")
            city = "Unknown"
            country = "Unknown"
        
        img = get_satellite_image(latitude, longitude, zoom=zoom)
        if img is None:
            return jsonify({'status':'error','message':'Could not retrieve satellite imagery'}), 500

        original_filename = save_temp_image(img, "original")
        vegetation_percentage, vegetation_highlighted, visualization_images, analysis_results = analyze_vegetation(
            img, highlight_color=[0,255,0], highlight_intensity=0.7, threshold=110
        )
        vegetation_img = Image.fromarray(vegetation_highlighted[:,:,::-1])
        vegetation_filename = save_temp_image(vegetation_img, "vegetation")
        
        # Save mask visualization image
        mask_rgb = visualization_images['mask']
        mask_img = Image.fromarray(mask_rgb)
        mask_filename = save_temp_image(mask_img, "mask")
        
        # Save original image from visualization
        viz_original = visualization_images['original']
        original_viz_img = Image.fromarray(viz_original)
        original_viz_filename = save_temp_image(original_viz_img, "viz_original")

        analysis_id = uuid.uuid4().hex
        
        # Extract GLI data for histogram
        gli_values = analysis_results['gli_normalized'].flatten()
        # Sample if large
        if len(gli_values) > 10000:
            indices = np.random.choice(len(gli_values), 10000, replace=False)
            gli_values = gli_values[indices]
        
        # Extract RGB data
        rgb_data = extract_rgb_data(np.array(img))
        
        # Sample GLI for heatmap
        gli_2d = analysis_results['gli_normalized']
        h, w = gli_2d.shape
        if h > 50 or w > 50:
            scale_factor = min(50/h, 50/w)
            new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
            heatmap_data = cv2.resize(gli_2d, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            heatmap_data = gli_2d
            
        # Generate historical data
        historical_data = generate_historical_data(vegetation_percentage)
        
        # Create analysis record for storage
        analysis_record = {
            'id': analysis_id,
            'analysis_date': datetime.now().isoformat(),
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'city': city,
                'country': country
            },
            'analysis_type': 'coordinates',
            'vegetation_percentage': vegetation_percentage,
            'threshold_value': analysis_results['threshold_value'],
            'images': {
                'original': original_filename,
                'vegetation': vegetation_filename,
                'mask': mask_filename,
                'viz_original': original_viz_filename
            },
            'histogram_data': gli_values.tolist(),
            'rgb_histogram': {
                'r_values': rgb_data['r'].tolist() if rgb_data is not None else [],
                'g_values': rgb_data['g'].tolist() if rgb_data is not None else [],
                'b_values': rgb_data['b'].tolist() if rgb_data is not None else []
            },
            'heatmap_data': heatmap_data.tolist(),
            'historical_data': historical_data
        }

        # Fetch and add weather data
        try:
            weather_rt, weather_combined = get_combined_weather_data(city, latitude, longitude)
            if weather_combined is None or (isinstance(weather_combined, pd.DataFrame) and weather_combined.empty):
                analysis_record['weather'] = get_fallback_weather_data()
            else:
                analysis_record['weather'] = weather_combined.fillna('').to_dict(orient='records')
        except Exception as e:
            print(f"Weather data error: {e}")
            analysis_record['weather'] = get_fallback_weather_data()

        # Fetch and add pollen data
        try:
            pollen_df = get_combined_pollen_data(city)
            latest_pollen = get_latest_pollen_data(city)
            forecast_pollen = get_forecast_pollen_data(city)
            
            if not pollen_df.empty:
                analysis_record['pollen'] = {
                    'combined': pollen_df.fillna('').to_dict(orient='records'),
                    'latest': latest_pollen if latest_pollen else [],
                    'forecast': forecast_pollen if forecast_pollen else []
                }
            else:
                analysis_record['pollen'] = {
                    'combined': [],
                    'latest': [],
                    'forecast': []
                }
        except Exception as e:
            print(f"Pollen data error: {e}")
            analysis_record['pollen'] = {
                'combined': [],
                'latest': [],
                'forecast': []
            }

        # Save analysis data to file
        data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
        with open(data_file, 'w') as f:
            json.dump(analysis_record, f)

        # Store in memory
        app.config['ANALYSES'].insert(0, analysis_record)
        if len(app.config['ANALYSES']) > 10:
            app.config['ANALYSES'] = app.config['ANALYSES'][:10]

        return jsonify({
            'status':'success',
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'city': city,
                'country': country
            },
            'vegetation_percentage': vegetation_percentage,
            'threshold_value': analysis_results['threshold_value'],
            'images': {
                'original': original_filename,
                'vegetation': vegetation_filename,
                'mask': mask_filename,
                'viz_original': original_viz_filename
            },
            'analysis_id': analysis_id,
            'weather': analysis_record['weather'],
            'pollen': analysis_record['pollen']
        })
    except Exception as e:
        print(f"Error in analyze_coordinates: {e}")
        return jsonify({'status':'error','message': str(e)}), 500

# Analyze uploaded image
@app.route('/analyze-upload', methods=['POST'])
def analyze_upload():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'status':'error','message':'No file uploaded'}), 400
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'status':'error','message':'File type not allowed'}), 400
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        img = Image.open(path)

        original_filename = save_temp_image(img, "original")
        vegetation_percentage, vegetation_highlighted, visualization_images, analysis_results = analyze_vegetation(
            img, highlight_color=[0,255,0], highlight_intensity=0.7, threshold=110
        )
        vegetation_img = Image.fromarray(vegetation_highlighted[:,:,::-1])
        vegetation_filename = save_temp_image(vegetation_img, "vegetation")
        
        # Save mask visualization image
        mask_rgb = visualization_images['mask']
        mask_img = Image.fromarray(mask_rgb)
        mask_filename = save_temp_image(mask_img, "mask")
        
        # Save original image from visualization
        viz_original = visualization_images['original']
        original_viz_img = Image.fromarray(viz_original)
        original_viz_filename = save_temp_image(original_viz_img, "viz_original")

        analysis_id = uuid.uuid4().hex
        
        # Extract GLI data for histogram
        gli_values = analysis_results['gli_normalized'].flatten()
        # Sample if large
        if len(gli_values) > 10000:
            indices = np.random.choice(len(gli_values), 10000, replace=False)
            gli_values = gli_values[indices]
        
        # Extract RGB data
        rgb_data = extract_rgb_data(np.array(img))
        
        # Sample GLI for heatmap
        gli_2d = analysis_results['gli_normalized']
        h, w = gli_2d.shape
        if h > 50 or w > 50:
            scale_factor = min(50/h, 50/w)
            new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
            heatmap_data = cv2.resize(gli_2d, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            heatmap_data = gli_2d
            
        # Generate historical data
        historical_data = generate_historical_data(vegetation_percentage)
        
        # Create analysis record for storage
        analysis_record = {
            'id': analysis_id,
            'analysis_date': datetime.now().isoformat(),
            'location': {
                'latitude': 0,
                'longitude': 0,
                'city': "Uploaded Image",
                'country': file.filename[:20] if file.filename else "Unknown"
            },
            'analysis_type': 'upload',
            'file_name': file.filename,
            'vegetation_percentage': vegetation_percentage,
            'threshold_value': analysis_results['threshold_value'],
            'images': {
                'original': original_filename,
                'vegetation': vegetation_filename,
                'mask': mask_filename,
                'viz_original': original_viz_filename
            },
            'histogram_data': gli_values.tolist(),
            'rgb_histogram': {
                'r_values': rgb_data['r'].tolist() if rgb_data is not None else [],
                'g_values': rgb_data['g'].tolist() if rgb_data is not None else [],
                'b_values': rgb_data['b'].tolist() if rgb_data is not None else []
            },
            'heatmap_data': heatmap_data.tolist(),
            'historical_data': historical_data
        }
        # No weather/pollen for uploads
        analysis_record['weather'] = []
        analysis_record['pollen'] = []
        # Save analysis data to file
        data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
        with open(data_file, 'w') as f:
            json.dump(analysis_record, f)
        
        # Store in memory
        app.config['ANALYSES'].insert(0, analysis_record)
        if len(app.config['ANALYSES']) > 10:
            app.config['ANALYSES'] = app.config['ANALYSES'][:10]

        return jsonify({
            'status':'success',
            'vegetation_percentage': vegetation_percentage,
            'threshold_value': analysis_results['threshold_value'],
            'images': {
                'original': original_filename,
                'vegetation': vegetation_filename,
                'mask': mask_filename,
                'viz_original': original_viz_filename
            },
            'analysis_id': analysis_id,
            'weather': [],
            'pollen': []
        })
    except Exception as e:
        print(f"Error in analyze_upload: {e}")
        return jsonify({'status':'error','message': str(e)}), 500

@app.route('/analysis/<analysis_id>')
def show_analysis(analysis_id):
    """Show detailed analysis page"""
    try:
        # Look for analysis in memory
        for analysis in app.config['ANALYSES']:
            if analysis['id'] == analysis_id:
                # Process pollen data to ensure it's in the correct format
                if isinstance(analysis.get('pollen'), dict):
                    pollen_data = analysis['pollen']
                else:
                    # For backward compatibility with old data format
                    pollen_data = {'combined': analysis.get('pollen', []), 'latest': [], 'forecast': []}
                
                return render_template(
                    'analysis.html', 
                    analysis_id=analysis_id,
                    weather=analysis.get('weather', []),
                    pollen=pollen_data,
                    latitude=analysis['location']['latitude'],
                    longitude=analysis['location']['longitude'],
                    city=analysis['location']['city'],
                    country=analysis['location']['country'],
                    analysis_date=datetime.fromisoformat(analysis['analysis_date']).strftime("%Y-%m-%d %H:%M:%S"),
                    vegetation_percentage=analysis['vegetation_percentage'],
                    histogram_data=json.dumps(analysis['histogram_data']),
                    rgb_histogram=json.dumps(analysis['rgb_histogram']),
                    heatmap_data=json.dumps(analysis['heatmap_data']),
                    historical_data=json.dumps(analysis['historical_data']),
                    images=analysis['images']
                )
                
        # If not found in memory, check files
        data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                analysis = json.load(f)
                
            return render_template(
                'analysis.html',
                analysis_id=analysis_id,
                weather=analysis.get('weather', []),
                pollen=analysis.get('pollen', []),
                latitude=analysis['location']['latitude'],
                longitude=analysis['location']['longitude'],
                city=analysis['location']['city'],
                country=analysis['location']['country'],
                analysis_date=datetime.fromisoformat(analysis['analysis_date']).strftime("%Y-%m-%d %H:%M:%S"),
                vegetation_percentage=analysis['vegetation_percentage'],
                histogram_data=json.dumps(analysis['histogram_data']),
                rgb_histogram=json.dumps(analysis['rgb_histogram']),
                heatmap_data=json.dumps(analysis['heatmap_data']),
                historical_data=json.dumps(analysis['historical_data']),
                images=analysis['images']
            )
        
        # Not found
        return "Analysis not found", 404
        
    except Exception as e:
        print(f"Error showing analysis: {e}")
        return f"Error loading analysis: {str(e)}", 500

@app.route('/recent-analyses')
@app.route('/analyses') # Adding route alias to match menu link
def recent_analyses():
    """Show recent analyses list"""
    return render_template('recent_analyses.html', analyses=app.config['ANALYSES'])

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve image files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generate-report/<analysis_id>')
def generate_report(analysis_id):
    """Generate or show an agricultural report for the analysis"""
    try:
        # First check if a report already exists
        report_file = os.path.join(app.config['DATA_FOLDER'], f"report_{analysis_id}.json")
        report_data = None
        if (os.path.exists(report_file)):
            with open(report_file, 'r') as f:
                report_data = json.load(f)
        
        # Get analysis data
        analysis_data = None
        # Check in memory first
        for analysis in app.config['ANALYSES']:
            if analysis['id'] == analysis_id:
                analysis_data = analysis
                break
        
        # If not in memory, check file
        if not analysis_data:
            data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                return "Analysis not found", 404
        
        # Get common template variables from analysis data
        template_vars = {
            'analysis_id': analysis_id,
            'city': analysis_data['location']['city'],
            'country': analysis_data['location']['country'],
            'latitude': analysis_data['location']['latitude'],
            'longitude': analysis_data['location']['longitude'],
            'vegetation_percentage': analysis_data['vegetation_percentage'],
            'images': analysis_data['images'],
            'weather': analysis_data.get('weather', []),
            'pollen': analysis_data.get('pollen', {})
        }
        
        # If report exists and is completed, just display it
        if report_data and report_data.get('status') == 'completed' and report_data.get('content'):
            return render_template(
                'report.html',
                report_date=report_data.get('date'),
                report=report_data.get('content'),
                process_status=report_data.get('status', 'completed'),
                process_log=report_data.get('process_log', []),
                **template_vars
            )
        
        # If report exists but is still in progress, show the loading page
        elif report_data:
            return render_template(
                'report.html',
                report_date=report_data.get('date'),
                report=None,
                process_status=report_data.get('status', 'initializing'),
                process_log=report_data.get('process_log', []),
                **template_vars
            )
        
        # Otherwise, create a new report file with initial status
        else:
            # We'll create an empty report file to indicate we're generating a report
            initial_report_data = {
                'date': datetime.now().isoformat(),
                'status': 'initializing',
                'content': None,
                'process_log': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'status': 'initializing',
                        'message': 'Starting report generation process'
                    }
                ]
            }
            
            with open(report_file, 'w') as f:
                json.dump(initial_report_data, f)
            
            # Show the loading page
            return render_template(
                'report.html',
                report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report=None,
                process_status='initializing',
                process_log=initial_report_data['process_log'],
                **template_vars
            )
        
    except Exception as e:
        print(f"Error in generate_report: {e}")
        return f"Error generating report: {str(e)}", 500

@app.route('/check-report/<analysis_id>')
def check_report(analysis_id):
    """Check if a report is ready and generate it if not"""
    try:
        report_file = os.path.join(app.config['DATA_FOLDER'], f"report_{analysis_id}.json")
        
        # If report doesn't exist at all, redirect to generate-report
        if not os.path.exists(report_file):
            return redirect(url_for('generate_report', analysis_id=analysis_id))
        
        # Read the current report file
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        # If the report is already generated, redirect to generate-report to display it
        if report_data.get('status') == 'completed' and report_data.get('content'):
            return redirect(url_for('generate_report', analysis_id=analysis_id))
        
        # Otherwise, generate the report now
        # Get analysis data
        analysis_data = None
        # Check in memory first
        for analysis in app.config['ANALYSES']:
            if analysis['id'] == analysis_id:
                analysis_data = analysis
                break
        
        # If not in memory, check file
        if not analysis_data:
            data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                return "Analysis not found", 404
        
        # Update the report status to processing
        process_log = report_data.get('process_log', [])
        process_log.append({
            'timestamp': datetime.now().isoformat(),
            'status': 'processing',
            'message': 'Analyzing vegetation and environmental data'
        })
        
        report_data['status'] = 'processing'
        report_data['process_log'] = process_log
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        # Update status to show we're preparing the report
        process_log.append({
            'timestamp': datetime.now().isoformat(),
            'status': 'preparing',
            'message': 'Preparing agricultural insights'
        })
        
        report_data['status'] = 'preparing'
        report_data['process_log'] = process_log
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        # Update status to show we're connecting to the LLM
        process_log.append({
            'timestamp': datetime.now().isoformat(),
            'status': 'connecting',
            'message': 'Connecting to AI for report generation'
        })
        
        report_data['status'] = 'connecting'
        report_data['process_log'] = process_log
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        # Generate the report content using LLM
        try:
            report_content = generate_agriculture_report(analysis_data)
            
            # Update to show we're finalizing the report
            process_log.append({
                'timestamp': datetime.now().isoformat(),
                'status': 'finalizing',
                'message': 'Finalizing agricultural report'
            })
            
            # Save the completed report
            process_log.append({
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'message': 'Report generation completed successfully'
            })
            
            report_data = {
                'date': datetime.now().isoformat(),
                'status': 'completed',
                'content': report_content,
                'process_log': process_log
            }
        except Exception as e:
            # If there's an error, update the status
            process_log.append({
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'Error generating report: {str(e)}'
            })
            
            report_data = {
                'date': datetime.now().isoformat(),
                'status': 'error',
                'content': f"<h2>Error Generating Report</h2><p>{str(e)}</p>",
                'process_log': process_log
            }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f)
        
        # Redirect to show the report
        return redirect(url_for('generate_report', analysis_id=analysis_id))
        
    except Exception as e:
        print(f"Error in check_report: {e}")
        return f"Error checking report: {str(e)}", 500

@app.route('/rag-bot/<analysis_id>', methods=['POST'])
def rag_bot(analysis_id):
    """Handle RAG bot queries about analysis data"""
    try:
        # Get the query from the request
        query = request.json.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided', 'status': 'error'}), 400
        
        # Get analysis data
        analysis_data = None
        # Check in memory first
        for analysis in app.config['ANALYSES']:
            if analysis['id'] == analysis_id:
                analysis_data = analysis
                break
        
        # If not in memory, check file
        if not analysis_data:
            data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                return jsonify({'error': 'Analysis not found', 'status': 'error'}), 404
        
        # Generate response using RAG bot
        from llm_report import answer_agriculture_query
        response = answer_agriculture_query(analysis_data, query)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in rag_bot: {e}")
        return jsonify({
            'error': f"Error processing query: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/chat-bot/<analysis_id>')
def chat_bot_page(analysis_id):
    """Render the dedicated chat bot page"""
    try:
        # Get analysis data
        analysis_data = None
        # Check in memory first
        for analysis in app.config['ANALYSES']:
            if analysis['id'] == analysis_id:
                analysis_data = analysis
                break
        
        # If not in memory, check file
        if not analysis_data:
            data_file = os.path.join(app.config['DATA_FOLDER'], f"analysis_{analysis_id}.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                return render_template('error.html', error='Analysis not found'), 404
        
        # Get location info for display
        location = analysis_data.get('location', {})
        city = location.get('city', 'Unknown')
        country = location.get('country', 'Unknown')
        
        # Render the chat bot template
        return render_template('chatbot.html', 
                              analysis_id=analysis_id,
                              city=city,
                              country=country,
                              vegetation_percentage=analysis_data.get('vegetation_percentage', 0))
    except Exception as e:
        print(f"Error in chat_bot_page: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/readme')
def readme_page():
    """Render the README documentation page"""
    return render_template('readme.html')

@app.route('/api/readme')
def get_readme():
    """Return the README.md content from the main directory"""
    try:
        # Get the absolute path to the main directory
        main_dir = os.path.dirname(os.path.abspath(__file__))
        readme_path = os.path.join(main_dir, 'README.md')
        
        # Check if README.md exists
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
                print(f"Successfully loaded README.md from {readme_path}")
                return readme_content
        
        # Try alternative locations if not found
        alternative_paths = [
            os.path.join(main_dir, 'readme.md'),         # Lowercase name
            os.path.join(main_dir, 'docs', 'README.md'), # Docs folder
            os.path.join(main_dir, '..', 'README.md')    # Parent directory
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                with open(alt_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                    print(f"Successfully loaded README.md from {alt_path}")
                    return readme_content
        
        # If README is not found anywhere, return a helpful message
        print("README.md file not found in any of the expected locations")
        return "# Documentation Not Found\n\nThe README.md file could not be found in the project directory. Please ensure a README.md file exists at the root of the project."
        
    except Exception as e:
        print(f"Error reading README: {str(e)}")
        return f"# Error Loading Documentation\n\nAn error occurred while trying to load the documentation: {str(e)}\n\nPlease contact the administrator."

if __name__ == '__main__':
    # Clean up on startup
    cleanup_old_files()
    # Use PORT env var with fallback to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', '0') == '1')