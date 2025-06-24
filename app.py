
import os
import shutil
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
import datetime
import re
from sentence_transformers import SentenceTransformer, util
import emoji
import umap

def copy_tat_data_boi():
    try:
        user_profile = os.environ.get("USERPROFILE") # C:\Users\YourUsername
        history_db_path = os.path.join(user_profile, "Appdata","Local","Google","Chrome","User Data","Profile 11","History")
        if not os.path.exists(history_db_path):
            print("History database does not exist.")
        else:
            shutil.copy2(history_db_path, "chrome_history.db")

    except Exception as e:
        print(f"step1 error: DB init copy failed: {e}")

def arrange_tat_data_boi():
    conn = sqlite3.connect("chrome_history.db")
    url_df= pd.read_sql_query("""
    SELECT
        urls.id AS url_id,
        urls.url,
        urls.title
    FROM
        urls;       
        """,conn)
    visits_df = pd.read_sql_query("""
    SELECT
        visits.id AS visit_id,
        visits.url AS url_id,
        visits.visit_time,
        visits.visit_duration
    FROM
        visits
    """, conn)
    conn.close()

    # Convert visit_time and visit_duration
    def webkit_to_datetime(webkit_ts):
        if webkit_ts is None:
            return None
        return datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=webkit_ts)
    def get_time_of_day(hour):
        if 4 <= hour <= 6:
            return "early morning"
        elif 7 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 16:
            return "afternoon"
        elif 17 <= hour <= 20:
            return "evening"
        elif 21 <= hour <= 23 or hour == 0:
            return "late night"
        else:
            return "deep night"
    visits_df["visit_datetime"] = visits_df["visit_time"].apply(webkit_to_datetime)
    visits_df["visit_duration_sec"] = visits_df["visit_duration"] / 1_000_000
    visits_df["day_of_week"] = visits_df["visit_datetime"].dt.weekday
    visits_df['time_of_day'] = visits_df['visit_datetime'].apply(lambda x: get_time_of_day(pd.to_datetime(x).hour))
    # visits_df["is_weekend"] = visits_df["day_of_week"] >= 5
    visits_df.drop(columns=["visit_time", "visit_duration"], inplace=True)
    # Extract only domain from URL
    def extract_domain(url):
        match = re.match(r'^(?:https?:\/\/)?(?:www\.)?([^\/]+)', url)
        return match.group(1) if match else None

    url_df["domain"] = url_df["url"].apply(extract_domain)

    # If you don't need the full URL anymore, you can drop it
    url_df.drop(columns=["url"], inplace=True)

    return visits_df,url_df

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'\s*-\s*YouTube$', '', text, flags=re.IGNORECASE)
    text = emoji.demojize(text, language='en')
    return text

def get_embeddings(model,df):
    embeddings = model.encode(
    df["title"].tolist(),
    batch_size=64,
    show_progress_bar=True)
    return embeddings

#pre_label auto namer
def auto_label_tat_boi(pre_labels,urls_embed_df,threshold=0.10):
    sim_label_embeddings = embeddor_model.encode(pre_labels, show_progress_bar=True)

    # Collect embedding column names
    embedding_columns = [col for col in urls_embed_df.columns if col.startswith("embedding_")]
    
    # Assign topics using cosine similarity
    sim_assigned_topics = []
    
    for i in range(len(urls_embed_df)):
        row_values = urls_embed_df.iloc[i][embedding_columns].values.astype(np.float32)
        similarity_scores = util.cos_sim(row_values, sim_label_embeddings)[0]
    
        best_index = similarity_scores.argmax().item()
 
        best_score = similarity_scores[best_index].item()
        best_label = pre_labels[best_index]

        if best_score < threshold:
            sim_assigned_topics.append("others")
        else:
            sim_assigned_topics.append(best_label)
    return sim_assigned_topics

def umap_embedding(embeddings_df):
    X = embeddings_df.values.astype("float32")  # optional but recommended for speed
    reducer_2d = umap.UMAP(n_components=2)
    reducer_3d = umap.UMAP(n_components=3)
    
    umap_2d = reducer_2d.fit_transform(X)
    umap_3d = reducer_3d.fit_transform(X)
    
    x_2d, y_2d = umap_2d[:, 0], umap_2d[:, 1]
    x_3d, y_3d, z_3d = umap_3d[:, 0], umap_3d[:, 1], umap_3d[:, 2]
    return pd.DataFrame({
    "x_2d": x_2d,
    "y_2d": y_2d,
    "x_3d": x_3d,
    "y_3d": y_3d,
    "z_3d": z_3d,
    })
    
def title_pre_labelling(embeddings_df):
    """change  it to db"""
    conn = sqlite3.connect("data.db")
    sim_topic_labels = pd.read_sql_query("SELECT pre_labels FROM pre_labels", conn)["pre_labels"].tolist()
    conn.close()
    # Get label embeddings
    sim_label_embeddings = embeddor_model.encode(sim_topic_labels)

    
    # Assign topics using cosine similarity
    sim_assigned_topics = []
    
    for i in range(len(embeddings_df)):
        row_values= embeddings_df.iloc[i].values.astype(np.float32)
        similarity_scores = util.cos_sim(row_values, sim_label_embeddings)[0]
        
        best_index = similarity_scores.argmax().item()
    
        best_score = similarity_scores[best_index].item()
        best_label = sim_topic_labels[best_index]

        if best_score < 0.10:
            sim_assigned_topics.append("others")
        else:
            sim_assigned_topics.append(best_label)
            
    return sim_assigned_topics
        

# Add the results to a new column


if __name__ == "__main__":
    # --- Data Load ---#
    """ handle multi chrome profiles and combo of it"""
    copy_tat_data_boi()
    visits_df,urls_df = arrange_tat_data_boi()

    urls_df.loc[:,"title"]=urls_df["title"].apply(lambda x: text_preprocessing(x))

    
    # ---- embeddor ---# 
    embeddor_model = SentenceTransformer("all-MiniLM-L6-v2_local", device='cpu')
    embeddings = get_embeddings(embeddor_model,urls_df)
    embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])
    # Combine the embeddings with the original DataFrame
    urls_df.reset_index(drop=True, inplace=True)
    urls_df=urls_df.join(umap_embedding(embeddings_df))
    urls_df["pre_labels"]=title_pre_labelling(embeddings_df)
    final_df = visits_df.merge(urls_df, left_on="url_id", right_on="url_id", how="inner")
    final_df.drop(columns=["url_id"], inplace=True)
    final_df["mood"]= None
    
    # mood mapper
    conn = sqlite3.connect("data.db")
    rules_df = pd.read_sql("SELECT * FROM mood_rules", conn)

    # Turn rules into a lookup dictionary
    rule_map = {
        (row['pre_label'], row['time_of_day']): row['mood']
        for _, row in rules_df.iterrows()
    }

    # Apply the rule
    def heuristic_mood_label(row):
        return rule_map.get((row['pre_labels'], row['time_of_day']), 'neutral')

    # Apply it once
    final_df['mood'] = final_df.apply(heuristic_mood_label, axis=1)
    final_df.to_sql("visits", conn, if_exists="replace", index=False)
    conn.close()
    
    
    
    
