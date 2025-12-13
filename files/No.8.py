import requests
import geopandas as gpd
import pandas as pd
import folium
import xml.etree.ElementTree as ET
import branca.colormap as cm
import webbrowser
import os
from shapely.geometry import box  # Required for creating the mask

# Helper function เพื่อดึงข้อความจาก XML element อย่างปลอดภัย
def get_xml_text(element, tag, default_val="N/A"):
    """Safely retrieves the text content of a sub-element."""
    node = element.find(tag)
    return node.text.strip() if node is not None and node.text else default_val

# --- 1. FETCH & PARSE DATA (ฉบับสมบูรณ์) ---
def get_weather_data_extended():
    # หมายเหตุ: uid และ ukey นี้เป็น placeholder หากใช้ API จริง ต้องเปลี่ยนเป็น Key ของคุณ
    url = "https://data.tmd.go.th/api/WeatherForecast7Days/v2/?uid=api&ukey=api12345"
    records = []
    
    try:
        print("Fetching XML data...")
        response = requests.get(url, timeout=15)
        response.raise_for_status() 
        root = ET.fromstring(response.content)
        
        for province in root.findall('./Provinces/Province'):
            name_en = get_xml_text(province, 'ProvinceNameEnglish', default_val=None)
            forecast = province.find('SevenDaysForecast')
            
            if name_en:
                # ดึงค่าอุณหภูมิสูงสุด
                max_temp_str = get_xml_text(forecast, 'MaximumTemperature', default_val="0")
                
                try:
                    max_temp = float(max_temp_str)
                except ValueError:
                    max_temp = 0.0 # หากแปลงไม่ได้ ให้เป็น 0

                records.append({
                    'province_en': name_en,
                    'date': get_xml_text(forecast, 'ForecastDate'),
                    'max_temp': max_temp,
                    'min_temp': get_xml_text(forecast, 'MinimumTemperature'),
                    'wind_speed': get_xml_text(forecast, 'WindSpeed'),
                    'desc': get_xml_text(forecast, 'DescriptionEnglish')
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            # การแก้ไขชื่อจังหวัดเพื่อให้ตรงกับไฟล์ GeoJSON ที่ใช้
            df['province_en'] = df['province_en'].replace({
                'Bangkok': 'Bangkok Metropolis',
                'Nakhon Ratchasima': 'Nakhon Ratchasima'
            })
        return df
    except requests.exceptions.RequestException as e:
        print(f"Network or API Error: {e}. Please check your internet connection or API URL/Key.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Parsing/Data Error: {e}")
        return pd.DataFrame()

# --- 2. CREATE MASKED MAP (ยังคงใช้ได้ดี) ---
def create_interactive_map(weather_df):
    map_url = "https://raw.githubusercontent.com/cvibhagool/thailand-map/master/thailand-provinces.geojson"
    output_file = "thailand_weather_cropped.html"
    
    if weather_df.empty:
        print("No weather data found to plot.")
        return

    try:
        print("Downloading map geometry and merging data...")
        # อ่านไฟล์ GeoJSON ด้วย geopandas
        gdf = gpd.read_file(map_url)
        # ผสานข้อมูลสภาพอากาศเข้ากับข้อมูลภูมิศาสตร์
        merged = gdf.merge(weather_df, left_on='NAME_1', right_on='province_en', how='left')
        merged['max_temp'] = merged['max_temp'].fillna(0)
        
        # กำหนดช่วงสีแบบ Dynamic
        valid_temps = merged[merged['max_temp'] > 0]['max_temp']
        if not valid_temps.empty:
            min_scale = valid_temps.min()
            max_scale = valid_temps.max()
        else:
            min_scale, max_scale = 20, 40

        print(f"Color Scale: {min_scale}°C - {max_scale}°C")

        # --- SETUP MAP ---
        esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}'
        esri_attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'

        m = folium.Map(
            location=[13.0, 101.0], 
            zoom_start=6, 
            tiles=esri_url, 
            attr=esri_attr
        )
        
        # --- NEW STEP: CREATE THE "INVERSE MASK" ---
        print("Generating Inverse Mask to hide external base map...")
        # 1. รวมโพลีกอนจังหวัดทั้งหมดให้เป็นโพลีกอนเดียวของประเทศไทย
        thailand_outline = gdf.dissolve().geometry[0]
        
        # 2. สร้างกรอบสี่เหลี่ยมที่ครอบคลุมทั้งโลก
        world_box = box(-180, -90, 180, 90)
        
        # 3. ลบพื้นที่ประเทศไทยออกจากกรอบโลก (ได้เป็น Mask รอบประเทศไทย)
        mask_geom = world_box.difference(thailand_outline)
        
        # 4. เพิ่ม Mask เป็น GeoJson Layer สีขาวทึบแสง (Opacity 1.0)
        folium.GeoJson(
            mask_geom,
            style_function=lambda x: {
                'fillColor': 'white', 
                'color': 'white',  # Border color
                'weight': 0,        # No border lines
                'fillOpacity': 1.0  # Opaque white
            },
            name="Inverse Mask"
        ).add_to(m)
        # --- END MASK STEP ---

        # Colormap
        colormap = cm.LinearColormap(
            # Blue (Cool) -> Red (Hot)
            colors=['#4575b4', '#91bfdb', '#fee090', '#fc8d59', '#d73027'],
            vmin=min_scale,
            vmax=max_scale,
            caption='Max Temperature (°C)'
        )
        colormap.add_to(m)

        # Style function สำหรับ Choropleth Layer
        def style_fn(feature):
            temp = feature['properties'].get('max_temp', 0)
            return {
                'fillColor': colormap(temp) if temp > 0 else '#d9d9d9', # สีเทาสำหรับข้อมูลที่ขาดหาย
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6 # กำหนดความโปร่งแสงให้มองเห็น Hillshade ด้านล่าง
            }

        # Add Data Layer (อยู่บนสุดของแผนที่)
        folium.GeoJson(
            merged,
            name='Weather Data',
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=['NAME_1'], aliases=['Province:']),
            popup=folium.GeoJsonPopup(
                fields=['NAME_1', 'date', 'max_temp', 'min_temp', 'wind_speed', 'desc'],
                aliases=['Province', 'Date', 'Max Temp (°C)', 'Min Temp (°C)', 'Wind', 'Weather'],
                localize=True
            )
        ).add_to(m)

        m.save(output_file)
        print(f"Success! Map saved as '{output_file}'")
        
        open_in_browser(output_file)

    except Exception as e:
        print(f"Mapping Error: {e}")

def open_in_browser(filename):
    try:
        file_path = os.path.abspath(filename)
        url = f'file://{file_path}'
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    df = get_weather_data_extended()
    create_interactive_map(df)