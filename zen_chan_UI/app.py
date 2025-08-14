from flask import Flask, render_template, request, jsonify
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

app = Flask(__name__)
DB = "..\\data.db"

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/map')
def map_master():
    return render_template('map_master.html')

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

@app.route('/api/mood_mapping_data')
def get_mood_mapping_data():
    with sqlite3.connect(DB) as conn: # Use 'with' statement for proper connection handling
        conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        cursor = conn.cursor()
        rules = cursor.execute('SELECT pre_label, time_of_day, mood FROM mood_rules').fetchall()
    
    # Collect all unique labels for each category
    all_pre_labels = sorted(list(set(r['pre_label'] for r in rules)))
    all_times_of_day = sorted(list(set(r['time_of_day'] for r in rules)))
    all_moods = sorted(list(set(r['mood'] for r in rules)))

    # Prepare links data with counts, using string labels (JS will map to IDs)
    raw_links = defaultdict(int) # Counts for (source_label, target_label) tuples

    for rule in rules:
        # Link from pre_label to time_of_day
        raw_links[(rule['pre_label'], rule['time_of_day'])] += 1
        # Link from time_of_day to mood
        raw_links[(rule['time_of_day'], rule['mood'])] += 1

    # Convert raw_links to a list of dicts for JSON serialization
    processed_links = []
    for (source_label, target_label), value in raw_links.items():
        processed_links.append({
            "source": source_label,
            "target": target_label,
            "value": value
        })

    # Return only the raw data needed by JS
    return jsonify({
        "pre_labels": all_pre_labels,
        "time_of_day_labels": all_times_of_day,
        "mood_labels": all_moods,
        "links": processed_links
    })
    
    

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
            SELECT domain, SUM(visit_duration_sec) as total_duration
            FROM visits
            GROUP BY domain
            ORDER BY total_duration DESC
            LIMIT 5
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

@app.route('/api/mood_over_time')
def mood_over_time_data():
    with sqlite3.connect(DB) as conn:
        query = """
            SELECT date(visit_datetime) as visit_date, mood, COUNT(*) as mood_count
            FROM visits
            WHERE visit_datetime >= date('now', '-7 days')
            GROUP BY visit_date, mood
            ORDER BY visit_date, mood
        """
        rows = conn.execute(query).fetchall()

    datasets = {}
    dates = sorted(list(set([row[0] for row in rows])))
    
    moods = sorted(list(set([row[1] for row in rows])))

    for mood in moods:
        datasets[mood] = {'label': mood, 'data': [0] * len(dates), 'fill': True}

    for row in rows:
        date_index = dates.index(row[0])
        datasets[row[1]]['data'][date_index] = row[2]
    
    return jsonify({'datasets': list(datasets.values())})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
