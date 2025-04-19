import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
token = os.environ.get('LLM_TOKEN')
endpoint = os.environ.get('LLM_ENDPOINT')
model = os.environ.get('LLM_MODEL')

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Agriculture-specific system prompt for report generation
AGRICULTURE_SYSTEM_PROMPT = """
You are an expert agricultural report generator. Using the provided satellite vegetation and climate data (including vegetation percentage, temperature, humidity, 
and precipitation), produce a clear, data-driven assessment of the area's agricultural potential. Your report must be enclosed within `<report></report>` tags and 
formatted in **Markdown** using headers (`##`, `###`), paragraphs (no bullet points), **bold** and *italic* text for emphasis, horizontal rules (`---`) to divide sections,
 and tables where appropriate. Include the following sections: an executive summary of the region's agricultural potential, an analysis of vegetation health and density, a 
 review of environmental conditions and their impact on agriculture, recommendations for suitable crops or agricultural activities, and guidance on land management and sustainability. 
 Write in a factual, formal tone, avoid speculation where data is missing, and ensure the report reads like a professional agricultural analysis.
"""

# RAG bot system prompt for interactive queries
RAG_SYSTEM_PROMPT = """
You are a friendly and helpful agricultural assistant named Biopixi. You have expertise in analyzing vegetation data, weather patterns, and farming recommendations,
but you present your knowledge in a conversational, engaging manner.

When chatting with users:
1. Be personable and use a warm, conversational tone - feel free to use casual language and occasional humor
2. Still reference data points from the analysis, but present them in an accessible way
3. Respond to small talk and non-agricultural questions naturally, but gently guide the conversation back to agricultural topics
4. Use varied response patterns and expressions to seem more dynamic and less robotic
5. Occasionally ask follow-up questions to create a more interactive experience
6. If the user seems confused or frustrated, adapt your tone to be more helpful and empathetic

While you should be conversational and friendly, ensure your agricultural advice remains accurate and helpful.
"""

def generate_agriculture_report(analysis_data):
    """
    Generate an agricultural report using LLM based on analysis data
    
    Args:
        analysis_data: Dictionary containing analysis results and environmental data
        
    Returns:
        HTML formatted report content as a string
    """
    try:
        # Extract relevant data for the report
        vegetation_percentage = analysis_data.get('vegetation_percentage', 'Unknown')
        location = analysis_data.get('location', {})
        city = location.get('city', 'Unknown')
        country = location.get('country', 'Unknown')
        latitude = location.get('latitude', 0)
        longitude = location.get('longitude', 0)
        
        # Get weather data if available
        weather_data = analysis_data.get('weather', [])
        weather_summary = {}
        if weather_data and len(weather_data) > 0:
            weather_summary = {
                'temperature': weather_data[0].get('Avg Temperature (°C)', 'Unknown'),
                'humidity': weather_data[0].get('Avg Humidity (%)', 'Unknown'),
                'precipitation': weather_data[0].get('Total Precipitation (mm)', 'Unknown'),
                'condition': weather_data[0].get('Condition', 'Unknown')
            }
        
        # Get pollen data if available
        pollen_data = analysis_data.get('pollen', {})
        pollen_risk = 'Unknown'
        if isinstance(pollen_data, dict) and pollen_data.get('combined') and len(pollen_data['combined']) > 0:
            pollen_risk = pollen_data['combined'][0].get('Risk', 'Unknown')
        elif isinstance(pollen_data, list) and len(pollen_data) > 0:
            pollen_risk = pollen_data[0].get('Risk', 'Unknown')
        
        # Create a data summary to send to the LLM
        data_summary = {
            'location': {
                'city': city,
                'country': country,
                'latitude': latitude,
                'longitude': longitude
            },
            'vegetation_percentage': vegetation_percentage,
            'weather': weather_summary,
            'pollen_risk': pollen_risk,
            'analysis_date': analysis_data.get('analysis_date', 'Unknown')
        }
        
        # Create the user prompt with data
        user_prompt = f"""
        Generate an agricultural assessment report based on the following data:
        
        Location: {city}, {country} ({latitude}, {longitude})
        Vegetation Coverage: {vegetation_percentage}%
        Weather Conditions:
        - Temperature: {weather_summary.get('temperature', 'Unknown')}°C
        - Humidity: {weather_summary.get('humidity', 'Unknown')}%
        - Precipitation: {weather_summary.get('precipitation', 'Unknown')} mm
        - General Condition: {weather_summary.get('condition', 'Unknown')}
        Pollen Risk Level: {pollen_risk}
        
        The vegetation percentage represents the density of plant coverage in the analyzed area from satellite imagery.
        Please provide a comprehensive agricultural assessment report focused on farming potential, suitable crops,
        and land management recommendations.
        """
        
        # Call the LLM API to generate the report
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": AGRICULTURE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            temperature=0.7,
            top_p=1,
            model=model
        )
        
        # Extract and return the generated report content
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating agricultural report: {e}")
        return f"""
        <h2>Error Generating Report</h2>
        <p>We encountered an error while generating the agricultural report: {str(e)}</p>
        <p>Please try again later or contact support if the problem persists.</p>
        """

def answer_agriculture_query(analysis_data, user_query):
    """
    Answer user queries about agricultural data using the RAG bot
    
    Args:
        analysis_data: Dictionary containing analysis results and environmental data
        user_query: String containing the user's question
        
    Returns:
        String containing the assistant's response
    """
    try:
        # Extract relevant data similar to report generation
        vegetation_percentage = analysis_data.get('vegetation_percentage', 'Unknown')
        location = analysis_data.get('location', {})
        city = location.get('city', 'Unknown')
        country = location.get('country', 'Unknown')
        latitude = location.get('latitude', 0)
        longitude = location.get('longitude', 0)
        
        # Get weather data if available
        weather_data = analysis_data.get('weather', [])
        weather_summary = {}
        if weather_data and len(weather_data) > 0:
            weather_summary = {
                'temperature': weather_data[0].get('Avg Temperature (°C)', 'Unknown'),
                'humidity': weather_data[0].get('Avg Humidity (%)', 'Unknown'),
                'precipitation': weather_data[0].get('Total Precipitation (mm)', 'Unknown'),
                'condition': weather_data[0].get('Condition', 'Unknown')
            }
        
        # Get pollen data if available
        pollen_data = analysis_data.get('pollen', {})
        pollen_risk = 'Unknown'
        if isinstance(pollen_data, dict) and pollen_data.get('combined') and len(pollen_data['combined']) > 0:
            pollen_risk = pollen_data['combined'][0].get('Risk', 'Unknown')
        elif isinstance(pollen_data, list) and len(pollen_data) > 0:
            pollen_risk = pollen_data[0].get('Risk', 'Unknown')
        
        # Get soil data if available
        soil_data = analysis_data.get('soil', {})
        soil_type = soil_data.get('type', 'Unknown')
        soil_ph = soil_data.get('ph', 'Unknown')
        soil_moisture = soil_data.get('moisture', 'Unknown')
        
        # Create the context for the RAG bot
        context = f"""
        ANALYSIS DATA:
        Location: {city}, {country} ({latitude}, {longitude})
        Vegetation Coverage: {vegetation_percentage}%
        Weather Conditions:
          - Temperature: {weather_summary.get('temperature', 'Unknown')}
          - Humidity: {weather_summary.get('humidity', 'Unknown')}
          - Precipitation: {weather_summary.get('precipitation', 'Unknown')}
          - Condition: {weather_summary.get('condition', 'Unknown')}
        Pollen Risk: {pollen_risk}
        Soil Information:
          - Type: {soil_type}
          - pH: {soil_ph}
          - Moisture: {soil_moisture}
        """
        
        # Adding agricultural context for better responses
        if vegetation_percentage is not None and vegetation_percentage != 'Unknown':
            veg_float = float(vegetation_percentage)
            if veg_float > 70:
                context += "\nThe area has very high vegetation coverage, indicating fertile land suitable for diverse crops."
            elif veg_float > 40:
                context += "\nThe area has moderate vegetation coverage, suitable for many common crops with proper management."
            elif veg_float > 20:
                context += "\nThe area has low vegetation coverage, may require additional irrigation and soil amendments."
            else:
                context += "\nThe area has very low vegetation coverage, suggesting challenging conditions for traditional agriculture."
        
        # Call the LLM API to generate a conversational response
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": RAG_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"CONTEXT: {context}\n\nUSER QUERY: {user_query}"
                }
            ],
            temperature=0.8,  # Slightly higher temperature for more varied responses
            top_p=0.9,
            model=model
        )
        
        # Return the generated response
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error in answer_agriculture_query: {str(e)}")
        return "Oops! I ran into a small hiccup while thinking about that. Could you try asking me again in a different way? I'm eager to help with your agricultural questions!"

# The generate_agriculture_response function is no longer needed since we're using the LLM API directly
# but we'll keep it for backwards compatibility in case any other code calls it
def generate_agriculture_response(prompt, vegetation_percentage, weather, location):
    """Legacy function for backwards compatibility"""
    try:
        # Use the LLM API instead of template-based responses
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": RAG_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            top_p=0.9,
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Hey there! Looking at the data for {location.get('city', 'this area')}, I can tell you the vegetation coverage is {vegetation_percentage}%. The weather seems to be around {weather.get('temperature', 'N/A')}°C with {weather.get('humidity', 'N/A')}% humidity. What specific agricultural information are you curious about?"