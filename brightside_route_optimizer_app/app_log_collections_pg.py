# app_log_collections_pg.py

import streamlit as st
import pandas as pd
import boto3
import json
import logging
import os
from io import StringIO
from datetime import datetime
from collections import OrderedDict

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename=f'logs/app_errors_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Configuration
aws_access_key_id = st.secrets.get("aws_access_key_id")
aws_secret_access_key = st.secrets.get("aws_secret_access_key")
aws_region = st.secrets.get("aws_region")

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

BUCKET = "brightside-route-optimizer"
ROUTES_FILE = "pwyc_delivery_routes_dummy_file.csv"
TEAM_MEMBERS_FILE = "brightside_team_members_dummy_file.csv"
COLLECTIONS_PREFIX = "collections/"  # S3 prefix for saved collection logs, e.g. collections/2025-01-19_Carter.json

# Common quick-tap dollar amounts shown as buttons for fast entry
QUICK_AMOUNTS = [0, 3, 5, 6, 7, 8, 12]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def load_routes_df():
    """Load the full routes CSV from S3. Returns a DataFrame."""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=ROUTES_FILE)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        df = df[df['Address'].notna()]
        df['Address'] = df['Address'].str.strip()
        df = df[df['Address'] != ''].reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load routes CSV: {e}")
        return pd.DataFrame()


def load_team_members():
    """Load team members from S3. Returns {name: email} dict."""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=TEAM_MEMBERS_FILE)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return dict(zip(df['name'], df['email']))
    except Exception as e:
        logger.error(f"Failed to load team members: {e}")
        return {}


def save_collection_log(driver_name, date_str, log_data):
    """Save a collection log as JSON to S3.

    File path: collections/YYYY-MM-DD_DriverName.json
    log_data is a list of stop dicts with collected amounts.
    """
    try:
        key = f"{COLLECTIONS_PREFIX}{date_str}_{driver_name.replace(' ', '_')}.json"
        payload = {
            'driver': driver_name,
            'date': date_str,
            'submitted_at': datetime.now().isoformat(),
            'stops': log_data
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(payload, indent=2)
        )
        logger.info(f"Saved collection log to s3://{BUCKET}/{key}")
        return True, key
    except Exception as e:
        logger.error(f"Failed to save collection log: {e}")
        return False, None


def load_collection_logs():
    """List and load all saved collection logs from S3. Returns list of log dicts."""
    try:
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=COLLECTIONS_PREFIX)
        logs = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.json'):
                continue
            try:
                file_response = s3.get_object(Bucket=BUCKET, Key=key)
                data = json.loads(file_response['Body'].read().decode('utf-8'))
                data['_s3_key'] = key
                logs.append(data)
            except Exception as e:
                logger.error(f"Failed to load collection log {key}: {e}")
        # Sort newest first
        logs.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        return logs
    except Exception as e:
        logger.error(f"Failed to list collection logs: {e}")
        return []


# ---------------------------------------------------------------------------
# Stop grouping (mirrors logic in app_home_pg.py)
# Groups rows at the same address that share a phone number into one entry.
# ---------------------------------------------------------------------------

def group_stops_for_address(address, routes_df):
    """Return grouped stop entries for a given address.

    Rows sharing the same phone number are merged: names joined,
    amounts summed, per-person breakdown stored. Different phones
    or blank phones stay separate.

    Returns a list of dicts with keys:
      names, phone, expected_amount, per_person_note,
      delivery_instructions, language, notes
    """
    if routes_df.empty or 'Address' not in routes_df.columns:
        return []
    norm = address.strip().lower()
    matches = routes_df[routes_df['Address'].str.strip().str.lower() == norm].copy()
    if matches.empty:
        return []

    raw = []
    for _, row in matches.iterrows():
        raw.append({
            'name': str(row.get('Name', '')).strip() if pd.notna(row.get('Name')) else '',
            'delivery_instructions': str(row.get('Delivery Instructions', '')).strip() if pd.notna(row.get('Delivery Instructions')) else '',
            'language': str(row.get('Language', '')).strip() if pd.notna(row.get('Language')) else '',
            'amount': str(row.get('Amount', '')).strip() if pd.notna(row.get('Amount')) else '',
            'notes': str(row.get('Notes', '')).strip() if pd.notna(row.get('Notes')) else '',
            'phone': str(row.get('Phone', '')).strip() if pd.notna(row.get('Phone')) else '',
        })

    phone_groups = OrderedDict()
    no_phone = []
    for r in raw:
        if r['phone']:
            phone_groups.setdefault(r['phone'], []).append(r)
        else:
            no_phone.append(r)

    grouped = []

    for phone, rows in phone_groups.items():
        names = [r['name'] for r in rows if r['name']]
        numeric_amounts = []
        per_person_parts = []
        for r in rows:
            try:
                val = float(r['amount']) if r['amount'] else 0.0
                numeric_amounts.append(val)
                per_person_parts.append(f"{r['name'] or 'Unknown'}: ${r['amount']}")
            except ValueError:
                numeric_amounts.append(0.0)
                per_person_parts.append(f"{r['name'] or 'Unknown'}: {r['amount']}")

        total = sum(numeric_amounts)
        total_str = str(int(total)) if total == int(total) else str(round(total, 2))
        per_person_note = " | ".join(per_person_parts) if len(rows) > 1 else None

        grouped.append({
            'names': ", ".join(names) if names else "Unknown",
            'phone': phone,
            'expected_amount': total_str,
            'per_person_note': per_person_note,
            'delivery_instructions': next((r['delivery_instructions'] for r in rows if r['delivery_instructions']), ''),
            'language': next((r['language'] for r in rows if r['language']), ''),
            'notes': next((r['notes'] for r in rows if r['notes']), ''),
        })

    for r in no_phone:
        grouped.append({
            'names': r['name'] if r['name'] else "Unknown",
            'phone': '',
            'expected_amount': r['amount'],
            'per_person_note': None,
            'delivery_instructions': r['delivery_instructions'],
            'language': r['language'],
            'notes': r['notes'],
        })

    return grouped


# ---------------------------------------------------------------------------
# Build the stop list for a driver from the routes CSV.
# Deduplicates addresses while preserving order so each address appears once.
# ---------------------------------------------------------------------------

def build_stop_list(routes_df):
    """Return an ordered list of unique addresses from the routes CSV."""
    if routes_df.empty:
        return []
    seen = set()
    stops = []
    for addr in routes_df['Address']:
        addr = addr.strip()
        if addr and addr not in seen:
            seen.add(addr)
            stops.append(addr)
    return stops


# ---------------------------------------------------------------------------
# Summary text generator
# ---------------------------------------------------------------------------

def build_summary_text(driver_name, date_str, stop_entries):
    """Build a copy-paste text summary of the collection log.

    stop_entries: list of dicts with keys address, names, expected, collected, status, notes
    """
    lines = [
        f"📦 Brightside Collection Summary",
        f"Driver: {driver_name}",
        f"Date: {date_str}",
        f"{'─' * 35}",
    ]

    total_expected = 0
    total_collected = 0

    for i, s in enumerate(stop_entries, 1):
        status_icon = "✅" if s['status'] == 'Collected' else ("⚠️" if s['status'] == 'Partial' else "❌")
        lines.append(f"{status_icon} Stop {i}: {s['address']}")
        lines.append(f"   Recipient(s): {s['names']}")
        lines.append(f"   Expected: ${s['expected']}  |  Collected: ${s['collected']}")
        if s['status'] == 'Not home':
            lines.append(f"   Note: Not home / no payment left")
        if s.get('driver_notes'):
            lines.append(f"   Note: {s['driver_notes']}")
        try:
            total_expected += float(s['expected']) if s['expected'] else 0
            total_collected += float(s['collected']) if s['collected'] else 0
        except ValueError:
            pass

    lines.append(f"{'─' * 35}")
    lines.append(f"💰 Total Expected:  ${int(total_expected) if total_expected == int(total_expected) else round(total_expected, 2)}")
    lines.append(f"💰 Total Collected: ${int(total_collected) if total_collected == int(total_collected) else round(total_collected, 2)}")

    shortfall = total_expected - total_collected
    if shortfall > 0:
        lines.append(f"⚠️  Shortfall: ${int(shortfall) if shortfall == int(shortfall) else round(shortfall, 2)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if 'collections_driver' not in st.session_state:
    st.session_state.collections_driver = None
if 'collections_date' not in st.session_state:
    st.session_state.collections_date = datetime.now().strftime('%Y-%m-%d')
if 'collections_submitted' not in st.session_state:
    st.session_state.collections_submitted = False
if 'collections_summary_text' not in st.session_state:
    st.session_state.collections_summary_text = ''


# ---------------------------------------------------------------------------
# PAGE CONTENT
# ---------------------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>Log Collections</h1>", unsafe_allow_html=True)
st.write("---")

TEAM_MEMBERS = load_team_members()
routes_df = load_routes_df()
stop_addresses = build_stop_list(routes_df)

# ---------------------------------------------------------------------------
# TAB LAYOUT: Log Collections | View Past Logs
# ---------------------------------------------------------------------------

tab_log, tab_history = st.tabs(["📋 Log Today's Collections", "📂 View Past Logs"])


# ============================================================
# TAB 1: LOG TODAY'S COLLECTIONS
# ============================================================
with tab_log:

    if st.session_state.collections_submitted:
        # --- POST-SUBMIT: show summary and copy text ---
        st.success("✅ Collection log saved successfully!")
        st.markdown("**Copy and send this summary to your route leader:**")
        st.code(st.session_state.collections_summary_text, language="text")

        if st.button("Log Another Route", use_container_width=True):
            st.session_state.collections_driver = None
            st.session_state.collections_submitted = False
            st.session_state.collections_summary_text = ''
            st.rerun()

    else:
        # --- STEP A: Driver selects their name and date ---
        st.write("**Who are you?**")
        col1, col2 = st.columns([2, 1])
        with col1:
            driver_options = ["— Select your name —"] + list(TEAM_MEMBERS.keys())
            selected_driver = st.selectbox(
                "Your name",
                options=driver_options,
                index=0 if st.session_state.collections_driver is None else
                driver_options.index(st.session_state.collections_driver)
                if st.session_state.collections_driver in driver_options else 0,
                label_visibility="collapsed"
            )
        with col2:
            selected_date = st.date_input(
                "Date",
                value=datetime.now(),
                label_visibility="collapsed"
            )

        if selected_driver == "— Select your name —":
            st.info("Select your name above to start logging collections.")
            st.stop()

        st.session_state.collections_driver = selected_driver
        date_str = selected_date.strftime('%Y-%m-%d')
        st.session_state.collections_date = date_str

        st.write("---")

        if not stop_addresses:
            st.warning("No stops found in the routes CSV. Please add stops on the Update Routes page first.")
            st.stop()

        # --- STEP B: Per-stop collection entry ---
        st.write(f"**Logging collections for {selected_driver} — {selected_date.strftime('%B %d, %Y')}**")
        st.caption("Check the box next to each stop you delivered, then enter the amount collected. Uncheck any stops that weren't yours.")

        stop_entries = []  # Only populated for checked (included) stops

        for addr in stop_addresses:
            grouped = group_stops_for_address(addr, routes_df)

            # Build display label: recipient names if available
            if grouped:
                recipient_label = " & ".join(g['names'] for g in grouped if g['names'] and g['names'] != 'Unknown')
                if not recipient_label:
                    recipient_label = "Unknown"
                expected_total = sum(
                    float(g['expected_amount']) if g['expected_amount'] else 0
                    for g in grouped
                )
                expected_str = str(int(expected_total)) if expected_total == int(expected_total) else str(round(expected_total, 2))
                lang = next((g['language'] for g in grouped if g['language']), '')
            else:
                recipient_label = "Unknown"
                expected_str = "?"
                lang = ''

            # Checkbox key for including this stop
            include_key = f"include_{addr}"
            if include_key not in st.session_state:
                st.session_state[include_key] = False

            # Outer row: checkbox on left, stop card on right
            left_col, card_col = st.columns([0.06, 0.94])
            with left_col:
                # Vertical spacer to align checkbox with top of card content
                st.write("")
                st.write("")
                include = st.checkbox("", key=include_key, label_visibility="collapsed")
            with card_col:
                with st.container(border=True):
                    # Dim the card slightly when not included
                    if not include:
                        st.caption("⬅ Check to include this stop")

                    header_cols = st.columns(3)
                    with header_cols[0]:
                        st.markdown(f"**{addr}**")
                    with header_cols[1]:
                        st.caption(f"👤 {recipient_label}")
                    with header_cols[2]:
                        st.caption(f"💰 Expected: ${expected_str}" + (f"  🗣 {lang}" if lang else ""))

                    # Per-person breakdown if grouped
                    for g in grouped:
                        if g['per_person_note']:
                            st.caption(f"Per person: {g['per_person_note']}")

                    if include:
                        # Not home toggle
                        not_home_key = f"nothome_{addr}"
                        if not_home_key not in st.session_state:
                            st.session_state[not_home_key] = False

                        not_home = st.checkbox("Not home / no payment left", key=not_home_key)

                        if not not_home:
                            # Quick-tap amount buttons
                            st.caption("Quick amounts:")
                            btn_cols = st.columns(len(QUICK_AMOUNTS))
                            quick_key = f"quick_{addr}"
                            custom_key = f"custom_{addr}"

                            if quick_key not in st.session_state:
                                try:
                                    pre = int(float(expected_str)) if expected_str not in ('', '?') else None
                                except ValueError:
                                    pre = None
                                st.session_state[quick_key] = pre

                            for col, amt in zip(btn_cols, QUICK_AMOUNTS):
                                with col:
                                    is_selected = st.session_state[quick_key] == amt
                                    label = f"**${amt}**" if is_selected else f"${amt}"
                                    if st.button(label, key=f"btn_{addr}_{amt}", use_container_width=True):
                                        st.session_state[quick_key] = amt
                                        st.rerun()

                            custom_val = st.text_input(
                                "Or enter custom amount ($)",
                                value=str(st.session_state[quick_key]) if st.session_state[quick_key] is not None else '',
                                key=custom_key,
                                placeholder="e.g. 4.50"
                            )

                            try:
                                collected = float(custom_val) if custom_val.strip() else (
                                    float(st.session_state[quick_key]) if st.session_state[quick_key] is not None else 0.0
                                )
                            except ValueError:
                                collected = 0.0

                            collected_str = str(int(collected)) if collected == int(collected) else str(round(collected, 2))
                            status = 'Collected'
                            try:
                                exp_float = float(expected_str) if expected_str not in ('', '?') else None
                                if exp_float is not None and collected < exp_float and collected > 0:
                                    status = 'Partial'
                                elif collected >= (exp_float or 0):
                                    status = 'Collected'
                                elif collected == 0:
                                    status = 'Partial'
                            except ValueError:
                                pass
                        else:
                            collected_str = '0'
                            status = 'Not home'

                        # Optional driver note
                        driver_note = st.text_input(
                            "Note (optional)",
                            key=f"note_{addr}",
                            placeholder="e.g. left on porch, will pay next time…"
                        )

                        stop_entries.append({
                            'address': addr,
                            'names': recipient_label,
                            'expected': expected_str,
                            'collected': collected_str,
                            'status': status,
                            'driver_notes': driver_note,
                            'language': lang,
                        })

        st.write("---")

        # --- STEP C: Submit ---
        checked_count = len(stop_entries)
        if checked_count == 0:
            st.markdown(f"0 stop(s) selected — $0")
        else:
            total_collected_disp = sum(float(s['collected']) for s in stop_entries)
            total_expected_disp = sum(float(s['expected']) if s['expected'] not in ('', '?') else 0 for s in stop_entries)
            st.markdown(
                f"**{checked_count} stop(s) selected — "
                f"Collected: ${total_collected_disp:.2f} of ${total_expected_disp:.2f} expected**"
            )

        submit_disabled = checked_count == 0
        if st.button("✅ Submit Collection Log", use_container_width=True, type="primary", disabled=submit_disabled):
            summary_text = build_summary_text(selected_driver, date_str, stop_entries)
            ok, saved_key = save_collection_log(selected_driver, date_str, stop_entries)
            if ok:
                st.session_state.collections_submitted = True
                st.session_state.collections_summary_text = summary_text
                st.rerun()
            else:
                st.error("Failed to save to S3. Please screenshot this page and send manually.")
                st.code(summary_text, language="text")


# ============================================================
# TAB 2: VIEW PAST LOGS
# ============================================================
with tab_history:
    st.write("Past collection logs saved to S3:")

    logs = load_collection_logs()

    if not logs:
        st.info("No collection logs found yet. Submit a log from the Log tab first.")
    else:
        # Summary table across all logs
        summary_rows = []
        for log in logs:
            total_expected = sum(
                float(s.get('expected', 0)) if s.get('expected') not in ('', '?', None) else 0
                for s in log.get('stops', [])
            )
            total_collected = sum(
                float(s.get('collected', 0)) if s.get('collected') not in ('', None) else 0
                for s in log.get('stops', [])
            )
            not_home_count = sum(1 for s in log.get('stops', []) if s.get('status') == 'Not home')
            summary_rows.append({
                'Date': log.get('date', ''),
                'Driver': log.get('driver', ''),
                'Stops': len(log.get('stops', [])),
                'Expected ($)': round(total_expected, 2),
                'Collected ($)': round(total_collected, 2),
                'Not Home': not_home_count,
                'Shortfall ($)': round(total_expected - total_collected, 2),
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.write("---")

        # Expandable detail per log
        for log in logs:
            label = f"{log.get('date', '?')} — {log.get('driver', '?')}"
            with st.expander(label):
                stops = log.get('stops', [])

                for i, s in enumerate(stops, 1):
                    status = s.get('status', '')
                    icon = "✅" if status == 'Collected' else ("⚠️" if status == 'Partial' else "❌")
                    st.markdown(f"{icon} **Stop {i}: {s.get('address', '')}**")
                    detail_cols = st.columns(4)
                    with detail_cols[0]:
                        st.caption(f"Recipient(s): {s.get('names', '—')}")
                    with detail_cols[1]:
                        st.caption(f"Expected: ${s.get('expected', '—')}")
                    with detail_cols[2]:
                        st.caption(f"Collected: ${s.get('collected', '—')}")
                    with detail_cols[3]:
                        st.caption(f"Status: {status}")
                    if s.get('driver_notes'):
                        st.caption(f"   📝 {s['driver_notes']}")

                st.write("")
                # Regenerate summary text for copy-paste
                summary_text = build_summary_text(log.get('driver', ''), log.get('date', ''), stops)
                st.markdown("**Copy summary:**")
                st.code(summary_text, language="text")
