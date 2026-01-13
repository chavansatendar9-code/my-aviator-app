import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import pytz
import sqlite3 # Using database
import google.generativeai as genai
import re

# --- Streamlit App Configuration (This must be the FIRST command) ---
st.set_page_config(page_title="Aviator Analysis Dashboard", layout="wide")

DB_FILE = "aviator_data.db" # Database file name
IST = pytz.timezone('Asia/Kolkata')
# AI Setup
genai.configure(api_key="AIzaSyDst6uQXykPNqtrRbisxBieFtTGjOhEp_s") 
model = genai.GenerativeModel('gemini-1.5-flash')

# --- आकर्षक आणि लहान दिसण्यासाठी CSS ---
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 1.5rem; padding-right: 1.5rem; }
    h1 { font-size: 1.75rem !important; text-align: center; margin-bottom: 10px;}
    h3 { font-size: 0.9rem !important; text-transform: uppercase; color: #888; margin-top:15px; margin-bottom:5px; border-bottom: 1px solid #444;}
    /* Indicator table styling */
    .indicator-table { width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }
    .indicator-table th, .indicator-table td { border: 1px solid #444; text-align: center; padding: 2px; height: 40px; vertical-align: middle;}
    .indicator-table th { background-color: #333; font-size: 1rem; font-weight: bold; }
    .indicator-table td { font-size: 0.8rem; line-height: 1.1; font-weight: bold; }
    .indicator-table .time-label { font-size: 0.7rem; color: #aaa; text-align: right; padding-right: 5px; width: 90px;}
    .live-block-row td { background-color: #1a1a1a; }
    .indicator-table td.highlight-cell {
        background-color: #28a745 !important;
        color: white !important;
    }
/* हा कोड CSS मध्ये ॲड कर */
    .indicator-table td.target-cell {
        background-color: #00d2ff !important; /* छान निळा रंग */
        color: black !important;
        border: 2px solid white !important;
        font-weight: 900 !important;
    }
    /* Tooltip styling */
    .indicator-table td[title]:hover::after {
        content: attr(title);
        position: absolute;
        transform: translate(-50%, -110%); /* Position above the cell */
        background: #f0f0f0; /* Lighter background */
        color: black;
        border: 1px solid #ccc;
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 0.8rem; /* Slightly larger tooltip font */
        font-weight: normal; /* Normal weight for tooltip */
        white-space: nowrap;
        z-index: 10;
        pointer-events: none; /* Prevent tooltip from blocking hover */
    }
</style>
""", unsafe_allow_html=True)

# --- Database Functions ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data (Timestamp TEXT PRIMARY KEY, Multiplier REAL)''')
    conn.commit()
    conn.close()

def load_data():
    init_db()
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT Timestamp, Multiplier FROM data ORDER BY Timestamp", conn)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='ISO8601')
        if not df.empty:
            if df['Timestamp'].dt.tz is None: df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
            df['Timestamp'] = df['Timestamp'].dt.tz_convert(IST)
    except Exception as e:
        df = pd.DataFrame(columns=["Timestamp", "Multiplier"])
    conn.close()
    return df

def insert_data(timestamp, multiplier):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        utc_timestamp = timestamp.astimezone(pytz.utc)
        cursor.execute("INSERT INTO data (Timestamp, Multiplier) VALUES (?, ?)", (utc_timestamp.isoformat(), multiplier))
        conn.commit()
    except sqlite3.IntegrityError: st.warning("Duplicate timestamp. Skipped.")
    except Exception as e: st.error(f"Data insertion failed: {e}")
    conn.close()

def delete_last_entry():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT Timestamp FROM data ORDER BY Timestamp DESC LIMIT 1")
        last_timestamp = cursor.fetchone()
        if last_timestamp:
            cursor.execute("DELETE FROM data WHERE Timestamp = ?", (last_timestamp[0],))
            conn.commit()
    except Exception as e: st.error(f"Delete failed: {e}")
    conn.close()

# --- Other Functions ---
def get_color(m):
    if m >= 3.0: return "🟢 3x+"
    if m >= 2.0: return "🟡 2x-3x"
    return "🔵 < 2x"
def get_jarvis_prediction(df):
    if df.empty: return None, "3x+", "डेटा एन्ट्री सुरू करा..."
  # --- गॅप डिटेक्शन (हा भाग तुझ्या मागील कोडमध्ये नव्हता) ---
    latest_time = df['Timestamp'].max()
    now_ist = datetime.now(IST)
    gap_minutes = (now_ist - latest_time).total_seconds() / 60
    
    if gap_minutes > 15:
        return None, "Wait", "⚠️ १५ मिनिटांचा गॅप! नवीन पॅटर्नसाठी ३-४ एन्ट्री टाका."
    # -----------------------------------------------------
    
    # शेवटचे १० मोठे मल्टिप्लायर (3x+) फिल्टर करा
    hits = df[df['Multiplier'] >= 3.0].tail(5).copy()
    
    # १२ मिनिटांच्या ब्लॉकनुसार मिनिटाचा क्रमांक (१ ते १२) काढा
    hits['M'] = (hits['Timestamp'].dt.minute % 12) + 1
    
    try:
        # AI ला प्रश्न विचारणे
        prompt = f"Data: {hits[['M', 'Multiplier']].to_string()}. Predict Next 3x+ Min (1-12) and Range. Output format: Min: [No], Range: [X-X], Msg: [Short Marathi Advice]"
        resp = model.generate_content(prompt).text
        
        # AI च्या उत्तरातील मिनिट आणि रेंज वेगळी करणे
        t_min = re.search(r'Min: (\d+)', resp)
        t_range = re.search(r'Range: ([\d.x-]+)', resp)
        
        return (int(t_min.group(1)) if t_min else None), (t_range.group(1) if t_range else "3x+"), resp
    except Exception as e:
        return None, "3x+", "Jarvis विचार करत आहे..."

# --- Main App Logic ---
st.title("✈️ Aviator Graph")

if 'data' not in st.session_state:
    st.session_state.data = load_data()

data = st.session_state.data
# प्रेडिक्शन मिळवण्यासाठी हे ऍड कर
t_min, t_range, jarvis_msg = get_jarvis_prediction(data)
# जार्विसचे प्रेडिक्शन स्क्रीनवर दाखवण्यासाठी
if t_min:
    st.info(f"🤖 *Jarvis AI प्रेडिक्शन:* पुढील जॅकपॉट मिनिट: *{t_min}* | अपेक्षित रेंज: *{t_range}*")
    st.write(f"💡 सल्ला: {jarvis_msg}")
else:
    st.warning("🤖 Jarvis: अजून डेटाची गरज आहे (किमान ३-४ एन्ट्री ३x च्या वरच्या टाका).")
col1, col2 = st.columns([1, 3], gap="medium")

# ================= Left Column (Controls & History Table) =================
with col1:
    with st.expander("A: X Value Entry", expanded=True):
        if 'next_round_start_time' not in st.session_state:
            st.session_state.next_round_start_time = None
        if st.session_state.next_round_start_time:
             st.info(f"पुढील वेळ: {st.session_state.next_round_start_time.strftime('%I:%M:%S %p')}")
        else:
             st.info("Multiplier टाकून सुरुवात करा.")
        with st.form(key='data_entry_form', clear_on_submit=True):
            new_multiplier_str = st.text_input("Multiplier:", label_visibility="collapsed", placeholder="उदा. 2.54")
            submitted = st.form_submit_button("Add X Value (Enter)")
            if submitted and new_multiplier_str:
                current_press_time = datetime.now(IST)
                if st.session_state.next_round_start_time:
                    round_timestamp = st.session_state.next_round_start_time
                else:
                    round_timestamp = current_press_time - timedelta(seconds=5)
                try:
                    new_multiplier = float(new_multiplier_str)
                    insert_data(round_timestamp, new_multiplier)
                    st.session_state.next_round_start_time = current_press_time
                    st.session_state.data = load_data()
                    st.rerun() # Refresh page automatically
                except ValueError: st.error("कृपया योग्य आकडा टाका.")

    st.subheader("B: Privius Round History")
    now = datetime.now(IST)
    # --- नवीन बदल: आता २४ मिनिटांचा डेटा दिसेल ---
    twenty_four_minutes_ago = now - timedelta(minutes=24) 
    
    previous_rounds = pd.DataFrame() # Initialize empty
    if not data.empty:
        if data['Timestamp'].dt.tz is None:
             data['Timestamp'] = data['Timestamp'].dt.tz_localize(IST, ambiguous='infer')
        # --- नवीन बदल: फिल्टर २४ मिनिटांसाठी केले ---
        previous_rounds = data[data['Timestamp'] >= twenty_four_minutes_ago] 

    if not previous_rounds.empty:
        st.dataframe(previous_rounds[['Timestamp', 'Multiplier']].sort_values(by="Timestamp", ascending=False),
                     use_container_width=True, hide_index=True, height=300,
                     column_config={ "Timestamp": st.column_config.DatetimeColumn("वेळ", format="hh:mm:ss A"), "Multiplier": st.column_config.NumberColumn("Multiplier", format="%.2f x") })
    else:
        st.info("शेवटच्या २४ मिनिटांत डेटा नाही.")

    if not data.empty:
        if st.button("D: ❌ Last X Value Dillit"):
            delete_last_entry()
            st.session_state.data = load_data()
            if 'next_round_start_time' in st.session_state: del st.session_state.next_round_start_time
            st.rerun() # Refresh page automatically

# ================= Right Column (Graph E & Indicator Table) =================
with col2:
    now_for_graphs = datetime.now(IST)

    # --- Graph E: Fixed 12-minute block ---
    st.subheader("Graph E (Live Block)")
    current_block_minute = (now_for_graphs.minute // 12) * 12
    start_time_e = now_for_graphs.replace(minute=current_block_minute, second=0, microsecond=0)
    end_time_e = start_time_e + timedelta(minutes=12)

    graph_e_data = pd.DataFrame() 
    if not data.empty:
        if data['Timestamp'].dt.tz is None: data['Timestamp'] = data['Timestamp'].dt.tz_localize(IST, ambiguous='infer')
        graph_e_data = data[(data['Timestamp'] >= start_time_e) & (data['Timestamp'] < end_time_e)]

    fig_e = px.bar(labels={"Timestamp": "", "Multiplier": "Multiplier (x)"})
    if not graph_e_data.empty:
        graph_e_data['रंग'] = graph_e_data['Multiplier'].apply(get_color)
        fig_e = px.bar(graph_e_data, x="Timestamp", y="Multiplier", color='रंग',
                     color_discrete_map={"🟢 3x+": "#28a745", "🟡 2x-3x": "#ffc107", "🔵 < 2x": "#007bff"})
    max_y_e = 10 if graph_e_data.empty else max(10, graph_e_data['Multiplier'].max() + 2)
    fig_e.update_yaxes(tick0=0, dtick=1, range=[0, max_y_e])
    fig_e.add_hline(y=3, line_dash="dash", line_color="red")
    fig_e.update_layout(xaxis_range=[start_time_e, end_time_e], xaxis_tickformat='%I:%M:%S %p',
                      showlegend=False, margin=dict(l=0, r=0, t=5, b=0))
    if not graph_e_data.empty:
        for index, row in graph_e_data.iterrows():
            if row['Multiplier'] >= 3.0:
                fig_e.add_annotation(x=row['Timestamp'], y=row['Multiplier']/2, text=row['Timestamp'].strftime('%I:%M:%S'),
                                   showarrow=False, font=dict(size=13, color="white", family="Arial Black, sans-serif"), textangle=-90)
    st.plotly_chart(fig_e, use_container_width=True)

    # --- History Indicator Table ---
    st.subheader("Previous Blocks 3x+ History (Values)")

    header_html = "<tr><th>Block Time</th>" + "".join([f"<th>{m}</th>" for m in range(1, 13)]) + "</tr>"
    table_html = f'<table class="indicator-table"><thead>{header_html}</thead><tbody>'

    if not data.empty and data['Timestamp'].dt.tz is None:
         data['Timestamp'] = data['Timestamp'].dt.tz_localize(IST, ambiguous='infer')

    for i in range(8): # 1 live + 7 previous blocks
        if i == 0: # Current Live Block
            start_time_hist = start_time_e
            end_time_hist = end_time_e
            row_class = "live-block-row" 
        else: # Previous Blocks
            start_time_hist = start_time_e - timedelta(minutes=12 * i)
            end_time_hist = start_time_e - timedelta(minutes=12 * (i - 1))
            row_class = ""

        block_data = pd.DataFrame() 
        if not data.empty:
             start_time_hist_aware = IST.localize(start_time_hist.replace(tzinfo=None)) if start_time_hist.tzinfo is None else start_time_hist.astimezone(IST)
             end_time_hist_aware = IST.localize(end_time_hist.replace(tzinfo=None)) if end_time_hist.tzinfo is None else end_time_hist.astimezone(IST)
             block_data = data[(data['Timestamp'] >= start_time_hist_aware) & (data['Timestamp'] < end_time_hist_aware) & (data['Multiplier'] >= 3.0)]

        row_html = f"<tr class='{row_class}'><td class='time-label'>{start_time_hist.strftime('%H:%M')}-{end_time_hist.strftime('%H:%M')}</td>" # 24-hour format

        for m in range(12): # Each minute within the block
            current_minute_start_time = start_time_hist + timedelta(minutes=m)
            current_minute_end_time = start_time_hist + timedelta(minutes=m+1)
            # Make sure these are timezone-aware for comparison
            current_minute_start_time_aware = IST.localize(current_minute_start_time.replace(tzinfo=None))
            current_minute_end_time_aware = IST.localize(current_minute_end_time.replace(tzinfo=None))

            minute_data_in_block = pd.DataFrame()
            if not block_data.empty:
                 minute_data_in_block = block_data[(block_data['Timestamp'] >= current_minute_start_time_aware) & (block_data['Timestamp'] < current_minute_end_time_aware)]

            hour_minute = current_minute_start_time.minute + 1 # मिनिटाचा क्रमांक (0-59)+1 = (1-60)
            tooltip_text = f"Min {hour_minute}"

            # --- जुना कोड काढून हा नवीन कोड टाका ---
            
            # 1. Target आहे का ते तपासा (फक्त चालू i=0 ब्लॉकसाठी)
            is_target = (i == 0 and t_min is not None and t_min == m + 1)

            # 2. Cell Content आणि Class ठरवा
            if not minute_data_in_block.empty:
                values = minute_data_in_block['Multiplier'].tolist()
                formatted_values = "<br>".join([f"{v:.2f}x" for v in values])
                cell_content = formatted_values
                cell_class = "highlight-cell" # हिरवा (डेटा आहे)
            elif is_target:
                cell_content = "🎯"
                cell_class = "target-cell"  # निळा (Jarvis चे टार्गेट)
            else:
                cell_content = ""
                cell_class = ""
            
            # ----------------------------------------

            row_html += f"<td class='{cell_class}' title='{tooltip_text}'>{cell_content}</td>" 
            
        row_html += "</tr>"
        table_html += row_html

    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)
