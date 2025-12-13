import requests
import pandas as pd
import xml.etree.ElementTree as ET
import webbrowser
import os
import folium
import branca.colormap as cm
from folium.plugins import MarkerCluster

# Helper function เพื่อดึงข้อความจาก XML element อย่างปลอดภัย
def get_xml_text(element, tag):
    """
    Safely retrieves the text content of a sub-element within an XML element.
    Returns None if the sub-element does not exist.
    """
    node = element.find(tag)
    return node.text if node is not None else None


def get_seismic_data():
    """
    Fetches daily seismic event data from the TMD API and converts it into a pandas DataFrame.
    """
    url = "http://data.tmd.go.th/api/DailySeismicEvent/v1/?uid=api&ukey=api12345"
    records = []

    try:
        # 1. Fetch data
        print("Fetching seismic data from TMD...")
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        
        # 2. Parse XML
        root = ET.fromstring(response.content)
        
        # 3. Extract records
        for event in root.findall('DailyEarthquakes'):
            try:
                # ใช้ get_xml_text ที่ถูกย้ายออกมาด้านนอก
                lat = get_xml_text(event, 'Latitude')
                lon = get_xml_text(event, 'Longitude')
                mag = get_xml_text(event, 'Magnitude')
                time_val = get_xml_text(event, 'DateTimeThai')
                region = get_xml_text(event, 'OriginThai')

                # ต้องแน่ใจว่าค่า Latitude, Longitude และ Magnitude มีอยู่ก่อนแปลงเป็น float
                if lat and lon and mag:
                    records.append({
                        'lat': float(lat),
                        'lon': float(lon),
                        'mag': float(mag),
                        'depth': get_xml_text(event, 'Depth') or "0",
                        'time': time_val or "Unknown Time",
                        'region': region or "Unknown Location"
                    })
            except (ValueError, TypeError) as ve: 
                # ข้ามเรคคอร์ดที่มีข้อมูลตัวเลขไม่ถูกต้อง
                # print(f"Skipping record due to invalid value format: {ve}")
                continue  

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data (Check URL/Network): {e}")
        return pd.DataFrame()
    except ET.ParseError:
        print("Error parsing XML content. The received data may not be valid XML.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


def create_seismic_map(df):
    """
    Creates an interactive Folium map showing earthquake locations with a MarkerCluster.
    """
    output_file = "seismic_map_clustered.html"
    
    if df.empty:
        print("No seismic data found to plot.")
        return

    print(f"Plotting {len(df)} seismic events...")

    # สร้างแผนที่เริ่มต้นที่พิกัดกลางของภูมิภาค
    m = folium.Map(
        location=[13.0, 101.0],  # ใกล้เคียงจุดศูนย์กลางของประเทศไทย
        zoom_start=5,  
        tiles='CartoDB dark_matter',
        attr='Seismic Data © TMD'
    )
    
    # --- CSS Fix สำหรับทำให้ Legend อ่านได้ชัดเจนบน Dark Map ---
    style_content = """
    <style>
        /* Force Legend Text to White with Black Outline */
        .legend, .legend text, .tick text, .caption {
            fill: #ffffff !important;       /* Changes SVG text color */
            color: #ffffff !important;      /* Changes HTML text color */
            font-weight: bold !important;
            font-size: 14px !important;
            
            /* Sharp Black Outline */
            text-shadow: 
                -1px -1px 0 #000, 
                 1px -1px 0 #000,
                -1px  1px 0 #000,
                 1px  1px 0 #000;
        }
        
        /* Make the legend background slightly semi-transparent dark */
        .legend {
            background-color: rgba(0, 0, 0, 0.3); 
            padding: 5px;
            border-radius: 5px;
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(style_content))

    # Color Scale (Green -> Yellow -> Red ตามความรุนแรง)
    colormap = cm.LinearColormap(
        colors=['#00ff00', '#ffff00', '#ff0000'],
        vmin=2.0, vmax=6.0, # กำหนดขอบเขตความรุนแรงที่สนใจ
        caption='Magnitude (Richter)'
    )
    colormap.add_to(m)

    # Marker Cluster (จัดกลุ่มหมุดเมื่อ Zoom Out)
    cluster_group = MarkerCluster(name="Seismic Clusters").add_to(m)

    # วนลูปเพื่อเพิ่ม marker ให้กับแผ่นดินไหวแต่ละจุด
    for _, row in df.iterrows():
        mag = row['mag']
        color = colormap(mag)
        
        # HTML สำหรับ Popup เมื่อคลิกที่ Marker
        popup_html = f"""
        <div style="font-family: Arial; width: 220px; color: black;">
            <h4 style="margin: 0 0 5px 0; color: {color}; text-shadow: 1px 1px 0 #fff;">Mag: {mag}</h4>
            <b>Loc:</b> {row['region']}<br>
            <b>Time:</b> {row['time']}<br>
            <b>Depth:</b> {row['depth']} km<br>
            <b>Coords:</b> {row['lat']}, {row['lon']}
        </div>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            # ขนาดขึ้นอยู่กับความรุนแรง
            radius=4 + (mag * 1.5), 
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Mag {mag}: {row['region']}"
        ).add_to(cluster_group)

    m.save(output_file)
    print(f"\nSuccess! Map saved as '{output_file}'")
    open_in_browser(output_file)

def open_in_browser(filename):
    """
    Opens the generated HTML map file in the default web browser.
    """
    try:
        path = os.path.abspath(filename)
        webbrowser.open(f'file://{path}')
    except Exception as e:
        print(f"Could not open map in browser: {e}")

if __name__ == "__main__":
    # --- ตรวจสอบให้แน่ใจว่าคุณได้รัน 'pip install folium branca requests pandas' แล้ว ---
    df = get_seismic_data()
    create_seismic_map(df)