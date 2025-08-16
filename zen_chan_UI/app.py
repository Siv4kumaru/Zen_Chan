from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import sys
import json
from big_boi import load_profile_data


app = Flask(__name__)
DB = "data.db"

@app.route('/')
def home():
    try:
        with open("last_profile.json", "r") as f:
            last_profile = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Handle empty file or missing file
        last_profile = {"profile": None, "name": None,"last_updated": None,"current": None}


    if last_profile.get("profile"):
        return render_template('index.html', profile_id=last_profile["profile"], profile_name=last_profile["name"], time_filter=last_profile["current"])
    profiles = get_chrome_profiles()
    return render_template('choose_profile.html', profiles=profiles)

@app.route('/api/edit_time_filter/<time_filter>')
def edit_time_filter(time_filter):
    try:
        with open("last_profile.json", "r") as f:
            last_profile = json.load(f)
        last_profile["current"] = time_filter
    except (json.JSONDecodeError, FileNotFoundError):
        last_profile = {"profile": None, "name": None, "last_updated": None, "current": time_filter}
    with open("last_profile.json", "w") as f:
        json.dump(last_profile, f)
    return jsonify(success=True)


@app.route('/bubble')
def bubble():
    return render_template('bubble.html')

@app.route('/category')
def category():
    return render_template('category_master.html')

@app.route('/dropdown')
def dropdown():
    return render_template('dropdown.html')

@app.route('/embedding')
def embedding():
    return render_template('embedding.html')

@app.route('/mood_distribution')
def mood_distribution():
    return render_template('mood_distribution.html')

@app.route('/mood')
def mood_master():
    return render_template('mood_master.html')

@app.route('/radar')
def radar():
    return render_template('radar_master.html')


@app.route('/get_data_embeddings')
def get_data_embeddings():
    filter_time = request.args.get('time', 'today')
    dimension = request.args.get('dim', '2d')

    query = """SELECT v.visit_datetime, v.title, v.domain, v.pre_labels,
                      v.x_2d, v.y_2d, v.x_3d, v.y_3d, v.z_3d,
                      v.visit_duration_sec 
               FROM visits v
               """
    params = []
    now = datetime.now()

    if filter_time == "today":
        query += " WHERE v.visit_datetime >= ?"
        params.append(now.replace(hour=0, minute=0, second=0, microsecond=0))
    elif filter_time == "week":
        start = now - timedelta(days=now.weekday())
        query += " WHERE v.visit_datetime >= ?"
        params.append(start.replace(hour=0, minute=0, second=0, microsecond=0))
    elif filter_time == "month":
        query += " WHERE v.visit_datetime >= date('now', 'start of month')"
    elif filter_time == "year":
        query += " WHERE v.visit_datetime >= date('now', 'start of year')"
    elif filter_time.isdigit():
        query += " WHERE strftime('%Y', v.visit_datetime) = ?"
        params.append(filter_time)

    with sqlite3.connect(DB) as conn:
        rows = conn.execute(query, params).fetchall()

    data = []
    for r in rows:
        point = {
            "datetime": r[0],
            "title": r[1],
            "domain": r[2],
            "pre_labels": r[3],
            "x": r[4] if dimension == "2d" else r[6],
            "y": r[5] if dimension == "2d" else r[7],
            "z": None if dimension == "2d" else r[8],
            # "color": r[9] or "#33ff33",  # fallback green if no color
            "duration": r[9] or 0
        }
        data.append(point)

    return jsonify(data)

@app.route('/api/radar_data')
def radar_data():
    """
    Provides JSON data for the radar graph.
    Accepts 'type' parameter ('pre_labels' or 'mood') and 'time_filter'.
    Defaults to 'mood' and 'all_time'.
    """
    chart_type = request.args.get('type', 'mood')
    time_filter = request.args.get('time_filter', 'all_time') # New parameter
    # remove if want control
    time_filter = "all_time"

    if chart_type not in ['pre_labels', 'mood']:
        return jsonify({"error": "Invalid chart type. Must be 'pre_labels' or 'mood'."}), 400
    if time_filter not in ['today', 'this_week', 'this_month', 'this_year', 'last_year', 'all_time']:
        return jsonify({"error": "Invalid time filter. Must be one of 'today', 'this_week', 'this_month', 'this_year', 'last_year', 'all_time'."}), 400

 
    

    # Base query parts
    select_clause = f"SELECT {chart_type}, sum(visit_duration_sec/3600.0) as avg_duration"
    from_clause = "FROM visits"
    
    # Always filter out NULL values for the selected chart_type
    where_parts = [f"{chart_type} IS NOT NULL"] 
    
    group_by_clause = f"GROUP BY {chart_type}"
    order_by_clause = f"ORDER BY {chart_type}"
    
    sql_params = [] # List to hold parameters for the SQL query

    # Add time filter conditions based on `time_filter`
    now = datetime.now()
    
    if time_filter == 'today':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        where_parts.append("visit_datetime >= ? AND visit_datetime < ?")
        sql_params.extend([start_date, end_date])
    elif time_filter == 'this_week':
        # Get the start of the current week (Monday)
        start_of_week = now - timedelta(days=now.weekday())
        start_date = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        where_parts.append("visit_datetime >= ?")
        sql_params.append(start_date)
    elif time_filter == 'this_month':
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        where_parts.append("visit_datetime >= ?")
        sql_params.append(start_date)
    elif time_filter == 'this_year':
        start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        where_parts.append("visit_datetime >= ?")
        sql_params.append(start_date)
    elif time_filter == 'last_year':
        start_of_last_year = now.replace(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        start_of_this_year = now.replace(year=now.year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        where_parts.append("visit_datetime >= ? AND visit_datetime < ?")
        sql_params.extend([start_of_last_year, start_of_this_year])
    # 'all_time' doesn't add any date conditions, so the list of where_parts remains as is.

    # Combine all WHERE clauses
    full_where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    # Construct the full SQL query
    query = f"{select_clause} {from_clause} {full_where_clause} {group_by_clause} {order_by_clause}"

    # print(f"Executing query: {query} with params: {sql_params}") # For debugging
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row  # <--- Add this line!
        cursor = conn.cursor()
        cursor.execute(query, sql_params) # Pass parameters separately to prevent SQL injection
        rows = cursor.fetchall()

    labels = []
    data = []

    for row in rows:
        labels.append(row[chart_type])
        data.append(round(row['avg_duration'],2))

    response_data = {
        "labels": labels,
        "datasets": [{
            "label": "Visit Duration (hours)",
            "data": data
        }]
    }

    return jsonify(response_data)
    

@app.route('/top_domains')
def top_domains():
    return render_template('top_domains.html')

@app.route('/daily_activity')
def daily_activity():
    return render_template('daily_activity.html')

@app.route('/mood_over_time')
def mood_over_time():
    return render_template('mood_over_time.html')

@app.route('/api/top_domains')
def top_domains_data():
    with sqlite3.connect(DB) as conn:
        query = """
            SELECT domain, SUM(visit_duration_sec)/3600  as total_duration
            FROM visits
            GROUP BY domain
            ORDER BY total_duration DESC
            LIMIT 6
        """
        rows = conn.execute(query).fetchall()
    labels = [row[0] for row in rows]
    data = [row[1] for row in rows]
    return jsonify({'labels': labels, 'data': data})

@app.route('/api/daily_activity')
def daily_activity_data():
    with sqlite3.connect(DB) as conn:
        query = """
            SELECT day_of_week, SUM(visit_duration_sec) / 3600.0 as total_duration
            FROM visits
            GROUP BY day_of_week
            ORDER BY day_of_week
        """
        rows = conn.execute(query).fetchall()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data = [0] * 7
    for row in rows:
        data[row[0]] = row[1]

    return jsonify({'labels': days, 'data': data})

@app.route('/api/mood_weekly_avg')
def mood_weekly_avg():
    with sqlite3.connect(DB) as conn:
        query = """
        SELECT 
            strftime('%w', visit_datetime) AS weekday,   -- 0=Sunday, 6=Saturday
            (SUM(visit_duration_sec) / 3600.0) / 
            ( (julianday(MAX(visit_datetime)) - julianday(MIN(visit_datetime))) / 7.0 ) 
            AS avg_hours
        FROM visits
        GROUP BY weekday
        ORDER BY weekday;

        """
        rows = conn.execute(query).fetchall()

    # Map weekdays to names
    weekday_map = {
        "0": "Sunday",
        "1": "Monday",
        "2": "Tuesday",
        "3": "Wednesday",
        "4": "Thursday",
        "5": "Friday",
        "6": "Saturday"
    }

    labels = [weekday_map[str(row[0])] for row in rows]
    data = [row[1] for row in rows]

    return {
        "labels": labels,
        "datasets": [{
            "label": "Average Hours Spent per Weekday",
            "data": data
        }]
    }


    datasets = {}
    dates = sorted(list(set([row[0] for row in rows])))
    
    moods = sorted(list(set([row[1] for row in rows])))

    for mood in moods:
        datasets[mood] = {'label': mood, 'data': [0] * len(dates), 'fill': True}

    for row in rows:
        date_index = dates.index(row[0])
        datasets[row[1]]['data'][date_index] = row[2]
    
    return jsonify({'datasets': list(datasets.values())})

@app.route('/api/mood_distribution_by_hour')
def mood_distribution_by_hour():
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT
                CAST(strftime('%H', visit_datetime) AS INTEGER) as hour_of_day,
                mood,
                SUM(visit_duration_sec) / 3600.0 as total_duration
            FROM visits
            WHERE mood IS NOT NULL
            GROUP BY hour_of_day, mood
            ORDER BY hour_of_day, mood
        """
        rows = conn.execute(query).fetchall()

    # Initialize data structure for Chart.js
    hours = [f"{h:02d}:00" for h in range(24)] # Labels for 00:00 to 23:00
    all_moods = sorted(list(set([row['mood'] for row in rows])))

    # Prepare datasets for Chart.js stacked bar chart
    datasets = []
    for mood in all_moods:
        data_for_mood = [0] * 24 # Initialize counts for each hour to 0
        datasets.append({
            'label': mood,
            'data': data_for_mood,
            'backgroundColor': get_mood_color(mood), # Function to get color for mood
            'borderColor': get_mood_color(mood),
            'borderWidth': 1
        })
    
    # Populate data
    for row in rows:
        hour_index = row['hour_of_day']
        mood = row['mood']
        total_duration = row['total_duration']
        
        # Find the correct dataset for the mood
        for dataset in datasets:
            if dataset['label'] == mood:
                dataset['data'][hour_index] = total_duration
                break
    
    return jsonify({
        'labels': hours,
        'datasets': datasets
    })

def get_mood_color(mood):
    # Simple color mapping for moods. Extend as needed.
    colors = {
        'happy': 'rgba(75, 192, 192, 0.6)',
        'neutral': 'rgba(255, 206, 86, 0.6)',
        'sad': 'rgba(255, 99, 132, 0.6)',
        'calm': 'rgba(153, 102, 255, 0.6)',
        'excited': 'rgba(255, 159, 64, 0.6)',
        'angry': 'rgba(200, 0, 0, 0.6)',
        'anxious': 'rgba(54, 162, 235, 0.6)',
        # Add more moods and colors as needed
    }
    return colors.get(mood.lower(), 'rgba(100, 100, 100, 0.6)') # Default grey



@app.route("/api/total_time_spent")
def tot():
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT sum(visit_duration_sec)/3600.00 as total_duration
                FROM visits
            """
        rows = conn.execute(query).fetchall()
    result = [dict(row) for row in rows]
    return result

# user profile chosing
import os, json, re
from PIL import Image, ImageDraw, ImageFont
import shutil

CHROME_USER_DATA = os.path.join(os.environ.get("USERPROFILE"), "AppData", "Local", "Google", "Chrome", "User Data")
PROFILE_PIC_DIR = os.path.join("static", "profile_pics")
os.makedirs(PROFILE_PIC_DIR, exist_ok=True)

def generate_placeholder(name, profile_id):
    """Generate a colored placeholder avatar with initials."""
    initials = "".join([part[0].upper() for part in name.split()[:2]]) or "?"
    
    img = Image.new("RGB", (128, 128), (100, 100, 200))  # background color
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except:
        font = ImageFont.load_default()
    
    # Use textbbox (Pillow 8.0+)
    bbox = draw.textbbox((0, 0), initials, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    draw.text(((128 - w) / 2, (128 - h) / 2), initials, fill="white", font=font)
    
    path = os.path.join(PROFILE_PIC_DIR, f"{profile_id}.png")
    img.save(path)
    return f"/static/profile_pics/{profile_id}.png"

def get_avatar_path(profile_id, name, avatar_icon):
    # 1. Check Accounts/Avatar Images (preferred, real profile photo)
    avatar_images_dir = os.path.join(CHROME_USER_DATA, profile_id, "Accounts", "Avatar Images")
    if os.path.isdir(avatar_images_dir):
        files = [f for f in os.listdir(avatar_images_dir) ]
        if files:
            newest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(avatar_images_dir, f)))
            src = os.path.join(avatar_images_dir, newest_file)  # take newest image
            dst = os.path.join(PROFILE_PIC_DIR, f"{profile_id}.png")
            if not os.path.exists(dst):
                try:
                    Image.open(src).save(dst)  # normalize to PNG
                except:
                    shutil.copy2(src, dst)
            return f"/static/profile_pics/{profile_id}.png"


   

    # 3. Fallback: generate placeholder
    return generate_placeholder(name, profile_id)



def get_chrome_profiles():
    local_state_path = os.path.join(CHROME_USER_DATA, "Local State")
    with open(local_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profile", {}).get("info_cache", {})

    profile_list = []
    for profile_id, info in profiles.items():
        name = info.get("name", profile_id)
        avatar = get_avatar_path(profile_id, name, info.get("avatar_icon"))
        profile_list.append({
            "id": profile_id,
            "name": name,
            "avatar": avatar
        })
    return profile_list

@app.route("/choose_profile")
def choose_profile():
    profiles = get_chrome_profiles()
    return render_template("choose_profile.html", profiles=profiles)

def load_profile_data_with_cleanup(profile, name,time):
    try:
        # yield actual work
        yield from load_profile_data(profile, time)
    finally:

        with open("last_profile.json", "w") as f:
            json.dump({"profile": profile, "name": name,"last_updated": datetime.now().isoformat(),"current": time}, f)

@app.route("/api/load_profile/<profile>/<name>/<time>")
def load_profile(profile,name,time):
    try:
        return Response(
            stream_with_context(load_profile_data_with_cleanup(profile,name, time)),
            mimetype="text/event-stream"
        )
    except Exception as e:
        return Response(f"Error loading profile: {e}", status=500)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)