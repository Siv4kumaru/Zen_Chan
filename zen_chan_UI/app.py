from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__)
DB = "..\data.db"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bubble')
def bubble():
    return render_template('bubble.html')

@app.route('/category')
def category():
    return render_template('category_master.html')

@app.route('/embedding')
def embedding():
    return render_template('embedding.html')

@app.route('/map')
def map_master():
    return render_template('map_master.html')

@app.route('/mood')
def mood_master():
    return render_template('mood_master.html')

@app.route('/radial')
def radial():
    return render_template('radial.html')

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



if __name__ == "__main__":
    app.run(debug=True)
