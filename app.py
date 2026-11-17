import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="HydroServer Battery Power Check",
    page_icon="🔋",
    layout="wide",
)

KNOWN_DATASTREAMS = {
    '01a0f3f2-1800-7000-8000-000000000018': {'code': '18', 'name': 'Site 18'},
    '01a0f3f2-1900-7000-8000-000000000019': {'code': '19', 'name': 'Site 19'},
    '01a0f3f2-2900-7000-8000-000000000029': {'code': '29', 'name': 'Site 29'},
    '01a0f3f2-5000-7000-8000-000000000050': {'code': '50', 'name': 'Site 50'},
    '01a0f3f2-5500-7000-8000-000000000055': {'code': '55', 'name': 'Site 55'},
    '01a0f3f1-d00a-7e10-a297-fd82ebe84912': {'code': '56', 'name': 'Crockett Canal'},
    '01a0f3f2-6000-7000-8000-000000000060': {'code': '60', 'name': 'Site 60'},
    '01a0f3f2-0423-7424-916a-cb264e3f83e0': {'code': '99', 'name': 'Fort Field Canal'},
}

VOLTAGE_COLOR_STOPS = [
    (11.2, (255, 204, 204)),  # Light Red
    (11.7, (255, 224, 178)),  # Light Orange
    (12.1, (255, 249, 196)),  # Light Yellow
    (12.4, (255, 255, 255)),  # White
    (12.8, (200, 230, 201)),  # Light Green
    (14.4, (200, 230, 201)),  # Light Green
    (14.8, (225, 190, 231)),  # Light Purple
]

FLOAT_COLOR_STOPS = [
    (0.0, (255, 204, 204)),   # Light Red
    (0.5, (255, 224, 178)),   # Light Orange
    (2.0, (255, 249, 196)),   # Light Yellow
    (4.5, (255, 255, 255)),   # White
    (6.0, (200, 230, 201)),   # Light Green
]

# ==========================================
# CORE ANALYSIS FUNCTIONS
# ==========================================
def voltage_to_soc(v):
    soc = (v - 11.70) / (12.85 - 11.70) * 100.0
    return float(np.clip(soc, 0.0, 100.0))


def extract_metadata(props, target_keys):
    if isinstance(props, dict):
        for k, v in props.items():
            k_lower = str(k).strip().lower()
            if k_lower in target_keys:
                return str(v).strip()
            if isinstance(v, (dict, list)):
                res = extract_metadata(v, target_keys)
                if res != "Unknown":
                    return res
    elif isinstance(props, list):
        for item in props:
            if isinstance(item, (dict, list)):
                res = extract_metadata(item, target_keys)
                if res != "Unknown":
                    return res
            elif isinstance(item, str):
                for tk in target_keys:
                    if item.lower().startswith(tk + ":"):
                        return item.split(":", 1)[1].strip()
    return "Unknown"


def get_most_recent_window(max_dt, target_month, target_day):
    target_this_year = pd.Timestamp(year=max_dt.year, month=target_month, day=target_day, tz=max_dt.tz)
    if max_dt >= target_this_year - pd.Timedelta(days=14):
        target = target_this_year
    else:
        target = pd.Timestamp(year=max_dt.year - 1, month=target_month, day=target_day, tz=max_dt.tz)
    return target - pd.Timedelta(days=14), target + pd.Timedelta(days=14)


def evaluate_diagnostic_status(v_min, v_max, f_hours):
    if pd.isna(v_min) or pd.isna(v_max) or pd.isna(f_hours):
        return "No Data"
    if (v_max - v_min) < 0.2:
        return "6 AC Power"
    if v_min < 12.3:
        if f_hours < 3.0:
            return "1 Check Solar"
        elif f_hours <= 6.0:
            return "1 Check Solar and Battery"
        else:
            return "1 Check Battery"
    elif 12.3 <= v_min <= 12.5:
        if f_hours <= 6.0:
            return "2 Good"
        else:
            return "3 Surplus Solar"
    else:
        if f_hours <= 6.0:
            return "4 Surplus Battery"
        else:
            return "5 Surplus Solar and Battery"


def process_period(period_df, prefix):
    status_col_name = f'Priority_{prefix}'
    if period_df.empty:
        return {
            status_col_name: 'No Data',
            f'{prefix}_Sunny_Day_1': 'No Data', f'{prefix}_Sunny_Day_2': 'No Data', f'{prefix}_Sunny_Day_3': 'No Data',
            f'{prefix}_Min_Voltage': 'No Data', f'{prefix}_Max_Voltage': 'No Data',
            f'{prefix}_Morning_Min_SoC_%': 'No Data', f'{prefix}_Est_4pm_SoC_%': 'No Data',
            f'{prefix}_Float_Hours_per_Day': 'No Data',
        }

    period_df = period_df.copy()
    period_df['SoC_Gap'] = period_df['Est_4pm_SoC_%'] - period_df['Morning_Min_SoC_%']
    sorted_df = period_df.sort_values(by=['Float_Hours', 'SoC_Gap'], ascending=[False, False])

    top = sorted_df.iloc[1:4] if len(sorted_df) > 1 else sorted_df.head(1)

    dates = top['Date'].tolist()
    d1 = dates[0] if len(dates) > 0 else 'No Data'
    d2 = dates[1] if len(dates) > 1 else 'No Data'
    d3 = dates[2] if len(dates) > 2 else 'No Data'

    avg_min_v = top['Min_Voltage'].mean()
    avg_max_v = top['Max_Voltage'].mean()
    avg_min_soc = top['Morning_Min_SoC_%'].mean()
    avg_4pm_soc = top['Est_4pm_SoC_%'].mean()
    avg_float = top['Float_Hours'].mean()

    status_diag = evaluate_diagnostic_status(avg_min_v, avg_max_v, avg_float)

    return {
        status_col_name: status_diag,
        f'{prefix}_Sunny_Day_1': d1,
        f'{prefix}_Sunny_Day_2': d2,
        f'{prefix}_Sunny_Day_3': d3,
        f'{prefix}_Min_Voltage': round(avg_min_v, 2) if not pd.isna(avg_min_v) else "No Data",
        f'{prefix}_Max_Voltage': round(avg_max_v, 2) if not pd.isna(avg_max_v) else "No Data",
        f'{prefix}_Morning_Min_SoC_%': round(avg_min_soc, 1) if not pd.isna(avg_min_soc) else "No Data",
        f'{prefix}_Est_4pm_SoC_%': round(avg_4pm_soc, 1) if not pd.isna(avg_4pm_soc) else "No Data",
        f'{prefix}_Float_Hours_per_Day': round(avg_float, 1) if not pd.isna(avg_float) else "No Data",
    }


def fetch_with_retry(url, timeout=60, retries=3):
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(retries):
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            if res.status_code == 200:
                return res
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def discover_voltage_sites():
    unique_sites = {}
    url = "https://hydroserver.waterrights.utah.gov/api/sensorthings/v1.1/Things?$expand=Datastreams($expand=ObservedProperty)&$top=200"

    while url:
        res = fetch_with_retry(url, timeout=60, retries=3)
        if not res:
            break

        res_json = res.json()
        for thing in res_json.get('value', []):
            site_id = thing.get('@iot.id')
            thing_name = thing.get('name', 'Unknown Site')
            props = thing.get('properties', {})

            possible_datastreams = []
            for ds in thing.get('Datastreams', []):
                obs_name = str(ds.get('ObservedProperty', {}).get('name', '')).lower()
                ds_name = str(ds.get('name', '')).lower()
                ds_desc = str(ds.get('description', '')).lower()
                unit_sym = str(ds.get('unitOfMeasurement', {}).get('symbol', '')).lower()
                unit_name = str(ds.get('unitOfMeasurement', {}).get('name', '')).lower()

                is_voltage = (
                    'voltage' in obs_name or 'voltage' in ds_name or 'voltage' in ds_desc or
                    'battery' in ds_name or unit_sym in ['v', 'volt', 'volts'] or unit_name in ['volt', 'volts', 'voltage']
                )
                if is_voltage:
                    phenom_time = str(ds.get('phenomenonTime', ''))
                    end_time = phenom_time.split('/')[-1] if '/' in phenom_time else phenom_time
                    possible_datastreams.append({
                        'id': ds.get('@iot.id'),
                        'name': ds.get('name', 'Battery Voltage'),
                        'end_time': end_time,
                    })

            if not possible_datastreams:
                continue

            possible_datastreams.sort(key=lambda x: x['end_time'], reverse=True)
            voltage_ds_id = possible_datastreams[0]['id']
            voltage_ds_name = possible_datastreams[0]['name']

            sys_name = extract_metadata(props, ['system_name', 'systemname', 'system'])

            if voltage_ds_id in KNOWN_DATASTREAMS:
                s_code = KNOWN_DATASTREAMS[voltage_ds_id]['code']
                s_name = KNOWN_DATASTREAMS[voltage_ds_id]['name']
            else:
                s_name = thing_name
                s_code = extract_metadata(props, ['station_id', 'stationid'])
                if " - " in s_name:
                    if s_code in ("Unknown", "N/A"):
                        s_code = s_name.split(" - ")[-1].strip()
                    s_name = s_name.split(" - ")[0].strip()
                if s_code in ("Unknown", "N/A"):
                    s_code = str(site_id)

            if site_id not in unique_sites:
                unique_sites[site_id] = {
                    'site_id': site_id,
                    'site_code': str(s_code),
                    'site_name': s_name,
                    'system_name': sys_name,
                    'datastream_id': voltage_ds_id,
                    'datastream_name': voltage_ds_name,
                }

        next_link = res_json.get('@iot.nextLink')
        if next_link:
            if next_link.startswith('http://'):
                next_link = next_link.replace('http://', 'https://')
            elif next_link.startswith('/'):
                next_link = "https://hydroserver.waterrights.utah.gov" + next_link
            url = next_link
        else:
            url = None

    return list(unique_sites.values())


def parse_datastream_csv(ds_id):
    csv_url = f'https://hydroserver.waterrights.utah.gov/api/data/datastreams/{ds_id}/csv'
    res = fetch_with_retry(csv_url, timeout=60, retries=3)
    if not res:
        return None

    valid_lines = [
        line for line in res.text.splitlines()
        if line.startswith('ResultTime') or (not line.startswith('#') and ',' in line and not line.startswith('Flow'))
    ]
    if len(valid_lines) < 2:
        return None

    df_site = pd.read_csv(io.StringIO("\n".join(valid_lines)))
    if 'ResultTime' not in df_site.columns or 'Result' not in df_site.columns:
        return None

    df_site = df_site[['ResultTime', 'Result']].rename(columns={'Result': 'Voltage'})
    df_site['Time'] = pd.to_datetime(df_site['ResultTime']).dt.tz_convert('America/Denver')
    df_site.set_index('Time', inplace=True)
    df_site.sort_index(inplace=True)
    return df_site


def analyze_single_site(site, collection_time_denver):
    site_id = site['site_id']
    site_code = site['site_code']
    site_name = site['site_name']
    sys_name = site['system_name']
    ds_id = site['datastream_id']
    ds_name = site['datastream_name']

    try:
        df_site = parse_datastream_csv(ds_id)
        if df_site is None or df_site.empty:
            return None

        max_ts = df_site.index.max()
        if pd.isna(max_ts):
            return None

        cutoff_date = collection_time_denver - pd.DateOffset(months=6)
        if max_ts < cutoff_date:
            return None

        recent_start = max_ts - pd.Timedelta(days=28)
        recent_raw = df_site[df_site.index >= recent_start]
        if recent_raw.empty:
            return None

        latest_meas_dt = max_ts.strftime('%Y-%m-%d %H:%M')
        latest_meas_iso = max_ts.isoformat()
        latest_voltage_val = round(float(df_site['Voltage'].iloc[-1]), 2)

        summer_start, summer_end = get_most_recent_window(max_ts, 6, 21)
        winter_start, winter_end = get_most_recent_window(max_ts, 12, 21)

        needed_mask = (
            (df_site.index >= recent_start) |
            ((df_site.index >= summer_start) & (df_site.index <= summer_end + pd.Timedelta(days=1))) |
            ((df_site.index >= winter_start) & (df_site.index <= winter_end + pd.Timedelta(days=1)))
        )
        df_eval = df_site.loc[needed_mask].copy()
        df_eval['SoC_%'] = df_eval['Voltage'].apply(voltage_to_soc)
        df_eval['Is_Float'] = df_eval['Voltage'] >= 13.6

        night_df = df_eval.between_time('23:00', '04:00').copy()
        night_df['Night_Date'] = (night_df.index - pd.Timedelta(hours=5)).strftime('%Y-%m-%d')

        nightly_results = []
        for night, group in night_df.groupby('Night_Date'):
            group = group.sort_index()
            if len(group) >= 2:
                socs = group['SoC_%'].values
                hourly_drops = [socs[i] - socs[i + 1] for i in range(len(socs) - 1)]
                nightly_results.append({
                    'Date': night,
                    'SoC_11PM_%': socs[0],
                    'Night_Hourly_Draw_%': round(float(np.mean(hourly_drops)), 2),
                })
        nightly_capacity_df = pd.DataFrame(nightly_results)

        daily_records = []
        for date, group in df_eval.groupby(df_eval.index.strftime('%Y-%m-%d')):
            if len(group) >= 8:
                group = group.sort_index()
                if not nightly_capacity_df.empty and date in nightly_capacity_df['Date'].values:
                    night_row = nightly_capacity_df.loc[nightly_capacity_df['Date'] == date]
                    soc_11pm = night_row['SoC_11PM_%'].values[0]
                    r_night_val = night_row['Night_Hourly_Draw_%'].values[0]
                else:
                    soc_11pm = np.nan
                    r_night_val = np.nan

                est_4pm_soc = min(100.0, soc_11pm + (7.0 * abs(r_night_val))) if not np.isnan(soc_11pm) else np.nan
                float_group = group[group['Is_Float']]
                float_hours = round((len(float_group) / len(group)) * 24.0, 1) if len(float_group) > 0 else 0.0

                daily_records.append({
                    'Date': date,
                    'Min_Voltage': group['Voltage'].min(),
                    'Max_Voltage': group['Voltage'].max(),
                    'Morning_Min_SoC_%': round(group['SoC_%'].min(), 1),
                    'Est_4pm_SoC_%': round(est_4pm_soc, 1) if not np.isnan(est_4pm_soc) else np.nan,
                    'Night_Hourly_Draw_%': r_night_val,
                    'Float_Hours': float_hours,
                })

        site_url = f"https://hydroserver.waterrights.utah.gov/sites/{site_id}"

        if len(daily_records) == 0:
            min_v = recent_raw['Voltage'].min()
            max_v = recent_raw['Voltage'].max()
            recent_stats = process_period(pd.DataFrame(), "Recent")
            winter_stats = process_period(pd.DataFrame(), "Winter")
            summer_stats = process_period(pd.DataFrame(), "Summer")
            recent_stats['Priority_Recent'] = '1 Check Battery'
            recent_stats['Recent_Sunny_Day_1'] = 'Sparse Data'
            recent_stats['Recent_Min_Voltage'] = round(min_v, 2)
            recent_stats['Recent_Max_Voltage'] = round(max_v, 2)
        else:
            daily_df = pd.DataFrame(daily_records)
            recent_df = daily_df[daily_df['Date'] >= recent_start.strftime('%Y-%m-%d')]
            summer_df = daily_df[
                (daily_df['Date'] >= summer_start.strftime('%Y-%m-%d')) &
                (daily_df['Date'] <= summer_end.strftime('%Y-%m-%d'))
            ]
            winter_df = daily_df[
                (daily_df['Date'] >= winter_start.strftime('%Y-%m-%d')) &
                (daily_df['Date'] <= winter_end.strftime('%Y-%m-%d'))
            ]
            recent_stats = process_period(recent_df, "Recent")
            winter_stats = process_period(winter_df, "Winter")
            summer_stats = process_period(summer_df, "Summer")

        # Priority_Recent is placed directly before Recent_Sunny_Day_1 via recent_stats
        site_record = {
            'Site_Code': site_code,
            'Site_Name': site_name,
            'System_Name': sys_name,
            'Latest_Measurement': latest_meas_dt,
            'Latest_Voltage': latest_voltage_val,
            'Source_Site_URL': site_url,
            'Latest_Measurement_ISO': latest_meas_iso,
            'Datastream_ID': ds_id,
            'Datastream_Name': ds_name,
        }
        site_record.update(recent_stats)
        site_record.update(winter_stats)
        site_record.update(summer_stats)
        return site_record

    except Exception:
        return None


# ==========================================
# CONTINUOUS COLOR GRADIENT HELPERS
# ==========================================
def interpolate_color(val, stops):
    if val <= stops[0][0]:
        r, g, b = stops[0][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    if val >= stops[-1][0]:
        r, g, b = stops[-1][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        if x0 <= val <= x1:
            t = (val - x0) / (x1 - x0) if x1 != x0 else 0.0
            r = round(c0[0] + t * (c1[0] - c0[0]))
            g = round(c0[1] + t * (c1[1] - c0[1]))
            b = round(c0[2] + t * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ffffff"


def style_voltage_series(series):
    styles = []
    for val in series:
        num = pd.to_numeric(val, errors='coerce')
        if pd.isna(num):
            styles.append("")
        else:
            hex_color = interpolate_color(float(num), VOLTAGE_COLOR_STOPS)
            styles.append(f"background-color: {hex_color}; color: #000000;")
    return styles


def style_float_hours_series(series):
    styles = []
    for val in series:
        num = pd.to_numeric(val, errors='coerce')
        if pd.isna(num):
            styles.append("")
        else:
            hex_color = interpolate_color(float(num), FLOAT_COLOR_STOPS)
            styles.append(f"background-color: {hex_color}; color: #000000;")
    return styles


# ==========================================
# CACHED DATA PIPELINE (24-HOUR TTL)
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def run_full_analysis():
    collected_at_utc = pd.Timestamp.now(tz='UTC')
    collected_at_denver = collected_at_utc.tz_convert('America/Denver')

    sites = discover_voltage_sites()
    rows = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(analyze_single_site, site, collected_at_denver)
            for site in sites
        ]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                rows.append(res)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.fillna("No Data", inplace=True)

        # Put Not Reporting (>48h at pull) stations at the top, preserving existing sort order within groups
        cutoff_48h_utc = collected_at_utc - pd.Timedelta(hours=48)
        latest_ts_utc = pd.to_datetime(df['Latest_Measurement_ISO'], utc=True, errors='coerce')
        df['Is_Not_Reporting_Sort_Key'] = latest_ts_utc.isna() | (latest_ts_utc < cutoff_48h_utc)
        df['Voltage_Sort_Key'] = pd.to_numeric(df['Recent_Min_Voltage'], errors='coerce')

        df = df.sort_values(
            by=['Is_Not_Reporting_Sort_Key', 'Priority_Recent', 'Voltage_Sort_Key'],
            ascending=[False, True, True],
            na_position='last',
        ).drop(columns=['Is_Not_Reporting_Sort_Key', 'Voltage_Sort_Key']).reset_index(drop=True)

    pulled_at_str = collected_at_denver.strftime('%Y-%m-%d %H:%M %Z')
    return df, pulled_at_str, collected_at_utc


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_site_timeseries(ds_id):
    return parse_datastream_csv(ds_id)


# ==========================================
# WEB UI
# ==========================================
st.title("🔋 HydroServer Remote Site Power Check")

top_left, top_right = st.columns([3, 1])
with top_right:
    if st.button("🔄 Refresh Data Now", type="primary", use_container_width=True):
        run_full_analysis.clear()
        fetch_site_timeseries.clear()
        st.rerun()

with st.spinner("Pulling and analyzing site voltage data from HydroServer..."):
    summary_df, last_updated, collected_at_utc = run_full_analysis()

with top_left:
    st.caption(f"Dataset collected: **{last_updated}** (auto-refreshes every 24 hours, or click Refresh Data Now)")

if summary_df.empty:
    st.warning("No active voltage sites found.")
    st.stop()

# Filters (System Name + Text Search + CSV Download)
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    systems = sorted(summary_df['System_Name'].astype(str).unique().tolist())
    selected_systems = st.multiselect("Filter by System Name", systems, default=systems)
with col2:
    search_text = st.text_input("Search Site Name, Code, or System", placeholder="e.g. Crockett, 11590...")
with col3:
    internal_cols = ['Latest_Measurement_ISO', 'Datastream_ID', 'Datastream_Name']
    export_df = summary_df.drop(columns=internal_cols, errors='ignore')
    st.download_button(
        label="⬇️ Download CSV",
        data=export_df.to_csv(index=False).encode('utf-8'),
        file_name=f"Power_Check_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

filtered_df = summary_df[summary_df['System_Name'].astype(str).isin(selected_systems)]
if search_text.strip():
    q = search_text.strip().lower()
    mask = (
        filtered_df['Site_Code'].astype(str).str.lower().str.contains(q) |
        filtered_df['Site_Name'].astype(str).str.lower().str.contains(q) |
        filtered_df['System_Name'].astype(str).str.lower().str.contains(q)
    )
    filtered_df = filtered_df[mask]

# 48-hour reporting status check relative to when the dataset was collected
cutoff_48h_utc = collected_at_utc - pd.Timedelta(hours=48)
latest_ts_utc = pd.to_datetime(filtered_df['Latest_Measurement_ISO'], utc=True, errors='coerce')
is_not_reporting = latest_ts_utc.isna() | (latest_ts_utc < cutoff_48h_utc)

# Summary metrics strip (3 metrics)
m1, m2, m3 = st.columns(3)
m1.metric("Sites Shown", len(filtered_df))
m2.metric("Active (≤48 hours)", int((~is_not_reporting).sum()))
m3.metric("Not Reporting (>48 hours)", int(is_not_reporting.sum()))

# Build styled table
display_df = filtered_df.drop(columns=internal_cols, errors='ignore')

voltage_cols = [c for c in display_df.columns if c.endswith('_Voltage')]
float_cols = [c for c in display_df.columns if c.endswith('Float_Hours_per_Day')]

def style_latest_measurement(series):
    return [
        "background-color: #ffcccc; color: #000000;" if is_not_reporting.loc[idx] else ""
        for idx in series.index
    ]

styled_df = display_df.style
if voltage_cols:
    styled_df = styled_df.apply(style_voltage_series, subset=voltage_cols)
if float_cols:
    styled_df = styled_df.apply(style_float_hours_series, subset=float_cols)
if 'Latest_Measurement' in display_df.columns:
    styled_df = styled_df.apply(style_latest_measurement, subset=['Latest_Measurement'])

st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Source_Site_URL": st.column_config.LinkColumn(
            "Source Data",
            display_text="Open HydroServer ↗",
            help="Opens this site's page on Utah HydroServer",
        )
    },
)

# ==========================================
# EMBEDDED SUBDAILY VOLTAGE GRAPH
# ==========================================
st.divider()
st.subheader("📈 Subdaily Battery Voltage Graph")

if not filtered_df.empty:
    site_options = {
        f"{row['Site_Code']} — {row['Site_Name']} ({row['System_Name']})": row
        for _, row in filtered_df.iterrows()
    }
    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        chosen_label = st.selectbox("Select a site to inspect its subdaily voltage:", list(site_options.keys()))
    chosen_row = site_options[chosen_label]

    with sel_col2:
        st.write("")
        st.link_button(
            "Open Source Site on HydroServer ↗",
            url=chosen_row['Source_Site_URL'],
            use_container_width=True,
        )

    with st.spinner(f"Loading subdaily readings for {chosen_row['Site_Name']}..."):
        ts_df = fetch_site_timeseries(chosen_row['Datastream_ID'])

    if ts_df is not None and not ts_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ts_df.index,
                y=ts_df['Voltage'],
                mode='lines+markers',
                marker=dict(size=4, color='#3f51b5'),
                line=dict(width=1.5, color='#3f51b5'),
                name='Voltage (V)',
            )
        )
        fig.add_hline(y=13.6, line_dash="dot", line_color="green", annotation_text="13.6V Float")
        fig.add_hline(y=11.8, line_dash="dash", line_color="red", annotation_text="11.8V Critical")

        max_date = ts_df.index.max()
        default_start = max_date - pd.Timedelta(days=7)

        fig.update_layout(
            title=f"{chosen_row['Datastream_Name']} — {chosen_row['Site_Name']} ({chosen_row['Site_Code']})",
            yaxis_title="Voltage (Volt)",
            xaxis_title="Datetime",
            hovermode="x unified",
            height=480,
            xaxis=dict(
                range=[default_start, max_date],
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="all"),
                    ]
                ),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Could not load time-series data for plotting.")