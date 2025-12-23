import streamlit as st
import pandas as pd
import numpy as np  
from typing import List
from datetime import datetime, timedelta, date
from streamlit import progress
import io
import time

# Set of special employee IDs allowed till 9:25
LATE_ALLOWED_IDS = {'GCU010013', 'GCU010017', 'GCU010025', 'GCU030010', 'GCU010005', 'GCU020004'}


def stepwise_file_upload(
    upload_labels, 
    key_prefix=None,
    dfs_key=None,
    idx_key=None
):
    """
    Step-by-step uploader: only one uploader visible at a time.
    Supports:
        - stepwise_file_upload(labels, key_prefix="attendance")
        - stepwise_file_upload(labels, dfs_key="x", idx_key="y")
        - stepwise_file_upload(labels)
    """

    # --- Resolve storage keys ---
    if key_prefix is not None:
        dfs_key = f"{key_prefix}_dfs"
        idx_key = f"{key_prefix}_index"
        bytes_key = f"{key_prefix}_bytes"
    else:
        dfs_key = dfs_key or "uploaded_dfs"
        idx_key = idx_key or "upload_index"
        bytes_key = f"{dfs_key}_bytes"

    labels = upload_labels

    # --- Reset button ---
    if st.button("🔄 Reset Uploads"):
        for k in [dfs_key, idx_key, bytes_key]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # --- Initialize session state ---
    if dfs_key not in st.session_state:
        st.session_state[dfs_key] = {}
        st.session_state[idx_key] = 0
    if bytes_key not in st.session_state:
        st.session_state[bytes_key] = {}

    current_index = st.session_state[idx_key]

    # --- All uploads done ---
    if current_index >= len(labels):
        return st.session_state[dfs_key]

    label = labels[current_index]

    st.markdown("---")
    st.subheader(f"Upload {label} File")

    uploaded_file = st.file_uploader(
        f"Choose {label}",
        type=["csv", "xlsx", "xls"],
        key=f"{dfs_key}_uploader_{current_index}",
    )

    # --- When a file is uploaded ---
    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            st.session_state[bytes_key][label] = file_bytes

            bio = io.BytesIO(file_bytes)

            # Special case for LEAVE CSV
            if label == "LEAVE":
                df = pd.read_csv(bio, skiprows=6, encoding="windows-1252")

            elif uploaded_file.name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(bio, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(bio, encoding="latin-1")

            else:
                df = pd.read_excel(bio)

            bio.close()

            st.session_state[dfs_key][label] = df
            st.session_state[idx_key] += 1

            #st.rerun()

        except Exception as e:
            # If it is NOT a rerun-related exception, show it
            if not _is_rerun_exc(e):
                st.error(f"❌ Failed to read {uploaded_file.name}: {e}")
            # Always rethrow rerun exceptions so Streamlit handles them
            else:
                raise

        # IMPORTANT: rerun must be OUTSIDE the try/except
        st.rerun()

    return st.session_state[dfs_key]



def read_session_bytes_with_retry(bytes_key: str, attempts: int = 3, delay: float = 0.1):
    """Read bytes stored in `st.session_state[bytes_key]` with retries.

    This helper handles transient Streamlit rerun-control exceptions by
    re-raising them (so Streamlit can manage reruns) and retrying on other
    transient failures. Returns bytes if available, or raises the last
    exception if attempts are exhausted.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            if bytes_key not in st.session_state:
                raise KeyError(f"{bytes_key} not in session_state")
            data = st.session_state[bytes_key]
            if data is None:
                raise ValueError(f"{bytes_key} is None in session_state")
            return data
        except Exception as ex:
            # If this is a Streamlit rerun-control exception, re-raise it.
            try:
                if getattr(ex, "is_fragment_scoped_rerun", False):
                    raise
            except Exception:
                pass

            tname = type(ex).__name__
            if "Rerun" in tname or "rerun" in tname.lower() or "RerunData" in repr(ex):
                raise

            last_exc = ex
            if attempt < attempts:
                time.sleep(delay)
                continue
            raise last_exc


def merge_files(df_in, df_out, no_working_days):
    total_days = len(calculate_working_days(df_in))
    cols_in = [col for col in df_in.columns if col.startswith('clock_in_') and 'nan' not in col]
    cols_out = [col for col in df_out.columns if col.startswith('clock_out_') and 'nan' not in col]

    late_df = calculate_late(df_in, cols_in)
    early_df = calculate_early(df_out, cols_out)
    holiday_cols = detect_holidays(df_in)  # detects holidays from clock-in data

    merged_data = []

    for i in range(len(df_in)):
        emp_id = df_in.loc[i, 'Emp Id']
        name = df_in.loc[i, 'Names']
        #present = df_in.loc[i, 'Present']

        late_flags = [
            col.replace('clock_in_', '')
            for col in cols_in
            if late_df.loc[i, col] == 'Late' and col not in holiday_cols
        ]

        in_date_keys = [col.replace('clock_in_', '') for col in cols_in]
        out_col_map = {col.replace('clock_out_', ''): col for col in cols_out}

        designation = df_in.loc[i, 'Designation'] if 'Designation' in df_in.columns else ''

        early_flags = []
        half_day_flags = []
        morning_abs = []
        afternoon_abs = []

        for date in in_date_keys:
            col_in = f'clock_in_{date}'
            col_out = f'clock_out_{date}'

            if col_in in holiday_cols or col_out in holiday_cols:
                continue  # Skip holidays

            in_val = str(df_in.loc[i, col_in]) if col_in in df_in.columns else '0'
            out_val = str(df_out.loc[i, col_out]) if col_out in df_out.columns else '0'

            try:
                clock_in = datetime.strptime(in_val, '%H:%M:%S') if in_val != '0' else None
            except:
                clock_in = None
            try:
                clock_out = datetime.strptime(out_val, '%H:%M:%S') if out_val != '0' else None
            except:
                clock_out = None

            # --- NEW LOGIC FOR DRIVER DESIGNATION ---
            if designation == 'Driver':
                # If a Driver has at least one punch, consider it a full present day.
                if not clock_in and not clock_out:
                    # No punches at all, considered fully absent for the day
                    morning_abs.append(date)
                    afternoon_abs.append(date)
                # else: If there's at least one punch, do not add to morning_abs, afternoon_abs, half_day_flags
                continue # Skip the rest of the standard half-day logic for drivers
            # --- END NEW LOGIC FOR DRIVER DESIGNATION ---

            # Default full absence
            if not clock_in and not clock_out:
                morning_abs.append(date)
                afternoon_abs.append(date)
                continue

            # Half-day logic
            if not clock_in and clock_out:
                morning_abs.append(date)
                half_day_flags.append(date)
            elif clock_in and not clock_out:
                afternoon_abs.append(date)
                half_day_flags.append(date)
            else:
                if clock_in and clock_in > datetime.strptime('10:30:00', '%H:%M:%S'):
                    morning_abs.append(date)
                    half_day_flags.append(date)

                if clock_out:
                    if datetime.strptime('12:15:00', '%H:%M:%S') <= clock_out < datetime.strptime('15:30:00', '%H:%M:%S'):
                        afternoon_abs.append(date)
                        half_day_flags.append(date)

                # Early leave if before threshold
                if clock_out and clock_out < datetime.strptime('15:45:00', '%H:%M:%S'):
                    early_flags.append(date)

        # Final adjustments
        morning_set = set(morning_abs)
        afternoon_set = set(afternoon_abs)
        day_abs = list(morning_set & afternoon_set)

        morning_abs = list(morning_set - set(day_abs))
        afternoon_abs = list(afternoon_set - set(day_abs))
        
        # Remove full-day absences from half_day_flags
        half_day_flags = [date for date in half_day_flags if date not in day_abs]

        merged_data.append({
            'Emp Id': emp_id,
            'Names': name,
            #'Present': present,
            'late_flags': late_flags,
            'early_flags': early_flags,
            'half_day_flags': half_day_flags,
            'AM_abs': morning_abs,
            'PM_abs': afternoon_abs,
            'days_abs': day_abs,
            'No_of_AM_abs': len(morning_abs),
            'No_of_PM_abs': len(afternoon_abs),
            'No_of_late': len(late_flags),
            'No_of_day_abs': len(day_abs),
        })

    df = pd.DataFrame(merged_data)
    df = drop_columns_by_prefix(df, 'Unnamed')
    df.fillna(0, inplace=True)
    columns_to_sum = ['No_of_AM_abs','No_of_PM_abs','No_of_day_abs']
    df = weighted_sum_and_replace_columns(df, columns_to_sum, 'Absent', [0.5,0.5,1.0])
    df['Working Days'] = no_working_days
    
    # Calculate Present and Working Days
    df['Present'] = no_working_days - df['Absent']
    df['Working Days'] = no_working_days 
    return df
"""
def merge_files_staffs(df_in, df_out, official_working_days):
    # Calculate the total number of official working days from the list
    #no_official_working_days = calculate_total_official_working_days(official_working_days)
    no_official_working_days = official_working_days

    # Convert official_working_days list to a set of 'MM_DD' strings for efficient lookup
    official_working_days_set = set()
    for wd_str in official_working_days:
        try:
            month, day = map(int, wd_str.split('_'))
            # Format to 'MM_DD' (e.g., '06_12' for June 12th)
            official_working_days_set.add(f"{month:02d}_{day:02d}")
        except ValueError:
            print(f"Warning: Could not parse official working day '{wd_str}'. Skipping for set conversion.")
            continue


    # Initial cleaning of column names for both DataFrames
    df_in.columns = df_in.columns.str.strip().str.replace('\xa0', '', regex=True)
    df_out.columns = df_out.columns.str.strip().str.replace('\xa0', '', regex=True)

    # Standardize 'Emp Id' and 'Names' columns if needed
    # Assuming 'Employee ID' maps to 'Emp Id' and 'Name' maps to 'Names' in your data
    if 'Employee ID' in df_in.columns and 'Emp Id' not in df_in.columns:
        df_in.rename(columns={'Employee ID': 'Emp Id'}, inplace=True)
    if 'Name' in df_in.columns and 'Names' not in df_in.columns:
        df_in.rename(columns={'Name': 'Names'}, inplace=True)
    
    # Do the same for df_out to ensure consistency for merging later if needed (though not directly merged here)
    if 'Employee ID' in df_out.columns and 'Emp Id' not in df_out.columns:
        df_out.rename(columns={'Employee ID': 'Emp Id'}, inplace=True)
    if 'Name' in df_out.columns and 'Names' not in df_out.columns:
        df_out.rename(columns={'Name': 'Names'}, inplace=True)


    # Identify clock-in and clock-out columns
    cols_in = [col for col in df_in.columns if col.startswith('clock_in_') and 'nan' not in col]
    cols_out = [col for col in df_out.columns if col.startswith('clock_out_') and 'nan' not in col]

    # Calculate late and early flags
    late_df = calculate_late(df_in, cols_in)
    early_df = calculate_early(df_out, cols_out) # Note: early_df now returns flags per column
    
    # Detect widespread holidays based on clock-in data
    holiday_cols_list = detect_holidays(df_in)
    holiday_cols_set = set(holiday_cols_list) # Convert to set for efficient lookup


    merged_data = []

    for i in range(len(df_in)):
        emp_id = df_in.loc[i, 'Emp Id']
        name = df_in.loc[i, 'Names']

        # Get late flags, excluding detected holidays
        # The late_df contains 'clock_in_date_key' as columns
        current_emp_late_flags = [
            date_key
            for col_name, status in late_df.loc[i].items()
            if status == 'Late' and col_name not in holiday_cols_set # Ensure it's not a detected holiday
        ]
        # Re-map from 'clock_in_DD_MM_YYYY' to 'DD_MM_YYYY'
        late_flags = [col.replace('clock_in_', '') for col in current_emp_late_flags]


        # Get early flags, excluding detected holidays
        current_emp_early_flags = [
            date_key
            for col_name, status in early_df.loc[i].items()
            if status == 'Early Leave' and col_name not in holiday_cols_set # Ensure it's not a detected holiday
        ]
        # Re-map from 'clock_out_DD_MM_YYYY' to 'DD_MM_YYYY'
        early_flags = [col.replace('clock_out_', '') for col in current_emp_early_flags]


        half_day_flags = []
        morning_abs = []
        afternoon_abs = []
        
        # Initialize surplus for each employee
        surplus_days_count = 0 

        # Iterate through all possible date keys (from clock-in columns)
        for date_key in [col.replace('clock_in_', '') for col in cols_in]:
            col_in = f'clock_in_{date_key}'
            col_out = f'clock_out_{date_key}'

            # --- Robustly parse date_key to 'MM_DD' for comparison with official_working_days_set ---
            current_day_month_key = None
            try:
                parsed_date_obj = None
                current_year_for_date_key = datetime.now().year # Use current year for date_key that doesn't include year
                
                # Attempt to parse date_key into a datetime.date object
                # Try DD_MM_YYYY first (e.g., '26_06_2025')
                try:
                    parsed_date_obj = datetime.strptime(date_key, '%d_%m_%Y').date()
                except ValueError:
                    # Try MM_DD_YYYY (e.g., '06_26_2025')
                    try:
                        parsed_date_obj = datetime.strptime(date_key, '%m_%d_%Y').date()
                    except ValueError:
                        # Try DD_MM (assuming current year, e.g., '26_06')
                        try:
                            parsed_date_obj = datetime.strptime(f"{date_key}_{current_year_for_date_key}", '%d_%m_%Y').date()
                        except ValueError:
                            # Try MM_DD (assuming current year, e.g., '06_26')
                            try:
                                parsed_date_obj = datetime.strptime(f"{date_key}_{current_year_for_date_key}", '%m_%d_%Y').date()
                            except ValueError:
                                pass # Parsing failed, parsed_date_obj remains None
                
                if parsed_date_obj:
                    current_day_month_key = f"{parsed_date_obj.month:02d}_{parsed_date_obj.day:02d}"
                else:
                    # This warning indicates a problem with the date_key format itself
                    print(f"Warning: Could not reliably parse date_key '{date_key}' for surplus calculation. Skipping this day for Emp Id {emp_id}.")
                    continue # Skip to next date_key if parsing fails

            except Exception as e:
                # Catch any unexpected errors during date_key processing
                print(f"Warning: Unexpected error processing date_key '{date_key}' for Emp Id {emp_id}: {e}. Skipping surplus check.")
                continue


            in_val = str(df_in.loc[i, col_in]) if col_in in df_in.columns else '0'
            out_val = str(df_out.loc[i, col_out]) if col_out in df_out.columns else '0'

            try:
                clock_in = datetime.strptime(in_val, '%H:%M:%S') if in_val != '0' else None
            except ValueError:
                clock_in = None
            try:
                clock_out = datetime.strptime(out_val, '%H:%M:%S') if out_val != '0' else None
            except ValueError:
                clock_out = None

            # Check for actual presence on this date (clocked in or out)
            is_present_on_this_day = (clock_in is not None) or (clock_out is not None)

            # --- SURPLUS DAY LOGIC ---
            # A day is surplus if:
            # 1. Employee was present (clocked in OR out)
            # 2. It is NOT an official working day (based on official_working_days_set)
            # 3. It is NOT a holiday detected by the 'detect_holidays' function
            if current_day_month_key and \
               current_day_month_key not in official_working_days_set and \
               col_in not in holiday_cols_set and \
               is_present_on_this_day:
                surplus_days_count += 1 

            # --- Original half-day/absence/late/early logic ---
            # Skip any day identified as a widespread holiday by detect_holidays
            if col_in in holiday_cols_set or col_out in holiday_cols_set:
                continue


            # Default full absence if no clock_in and no clock_out
            if not clock_in and not clock_out:
                morning_abs.append(date_key)
                afternoon_abs.append(date_key)
                continue

            # Half-day logic: one clock but not the other
            if not clock_in and clock_out:
                morning_abs.append(date_key)
                half_day_flags.append(date_key)
            elif clock_in and not clock_out:
                afternoon_abs.append(date_key)
                half_day_flags.append(date_key)
            else: # Both clock_in and clock_out are present
                # Morning absence (late clock-in)
                if clock_in > datetime.strptime('10:30:00', '%H:%M:%S'):
                    morning_abs.append(date_key)
                    half_day_flags.append(date_key)

                # Afternoon absence (early clock-out before 15:30, after 12:15)
                # Note: The early_flags already capture those who left before 15:45.
                # This check is for defining a PM_abs half-day specifically.
                if clock_out:
                    if datetime.strptime('12:15:00', '%H:%M:%S') <= clock_out < datetime.strptime('15:30:00', '%H:%M:%S'):
                        afternoon_abs.append(date_key)
                        half_day_flags.append(date_key)

                # Early leave for flagging (independent of half-day logic here, handled by early_df)
                # No explicit 'early_flags.append(date_key)' needed here because it's derived from early_df already.

        # Final adjustments for morning/afternoon/full-day absences
        morning_set = set(morning_abs)
        afternoon_set = set(afternoon_abs)
        day_abs = list(morning_set & afternoon_set) # Days where both AM and PM are absent (full day)

        morning_abs = list(morning_set - set(day_abs)) # AM absent but not full day
        afternoon_abs = list(afternoon_set - set(day_abs)) # PM absent but not full day

        merged_data.append({
            'Emp Id': emp_id,
            'Names': name,
            'late_flags': late_flags,
            'early_flags': early_flags, # These are dates (DD_MM_YYYY) where employee was early
            'half_day_flags': list(set(half_day_flags)), # Ensure unique half-day flags
            'AM_abs': morning_abs,
            'PM_abs': afternoon_abs,
            'days_abs': day_abs,
            'No_of_AM_abs': len(morning_abs),
            'No_of_PM_abs': len(afternoon_abs),
            'No_of_late': len(late_flags),
            'No_of_day_abs': len(day_abs),
            'Surplus': surplus_days_count # The new column
        })

    df = pd.DataFrame(merged_data)
    df = drop_columns_by_prefix(df, 'Unnamed')
    df.fillna(0, inplace=True) # Fills NaNs potentially introduced if an employee has no instances of a certain flag type

    # Calculate 'Absent' column
    columns_to_sum = ['No_of_AM_abs','No_of_PM_abs','No_of_day_abs']
    df = weighted_sum_and_replace_columns(df, columns_to_sum, 'Absent', [0.5,0.5,1.0])
    
    # Calculate 'Present' and 'Working Days' based on official working days
    df['Present'] = no_official_working_days - df['Absent']
    df['Working Days'] = no_official_working_days
    
    return df
"""

def split_file(df):
    dates = calculate_date_month(df)
    gap = 13

    idx_name = list(range(4, len(df), gap))
    idx_in = list(range(7, len(df), gap))
    idx_out = list(range(8, len(df), gap))

    df_bio = df.iloc[idx_name].copy()
    df_in = df.iloc[idx_in].copy()
    df_out = df.iloc[idx_out].copy()

    # Assign dynamic column names
    if len(dates) == df_in.shape[1] == df_out.shape[1]:
        df_in.columns = [f'clock_in_{d}' for d in dates]
        df_out.columns = [f'clock_out_{d}' for d in dates]
    else:
        raise ValueError("Mismatch between extracted dates and clock columns")

    # Drop any spurious nan columns
    df_in = df_in.loc[:, ~df_in.columns.str.contains("nan", case=False)]
    df_out = df_out.loc[:, ~df_out.columns.str.contains("nan", case=False)]

    # Biometric info cleanup
    df_bio = df_bio.rename(columns={
        'Monthly Attendance Summary': 'Emp Id',  # Changed from 'Emp Id' to 'Emp Id' for consistency
        'Unnamed: 2': 'Names',
        'Unnamed: 7': 'Present',
    })

    # Reset indexes
    for d in (df_bio, df_in, df_out):
        d.reset_index(drop=True, inplace=True)

    # Combine
    df_all = pd.concat([df_bio, df_in, df_out], axis=1).fillna(0)
    df_all = drop_columns_by_prefix(df_all, "Unnamed")
    df_all.drop(columns="Present", axis=1, inplace=True, errors="ignore")

    return df_all, pd.concat([df_bio, df_in], axis=1).fillna(0), pd.concat([df_bio, df_out], axis=1).fillna(0)


# This function calcualtes late entries
# -------------------- Function: Calculate Late Entries --------------------
def calculate_late(df, cols_in):
    def classify(clock_in_str, emp_id):
        if str(clock_in_str) == '0':
            return 'Absent'
        try:
            clock_in = datetime.strptime(str(clock_in_str), '%H:%M:%S')
            threshold = datetime.strptime('09:25:00', '%H:%M:%S') if emp_id in LATE_ALLOWED_IDS else datetime.strptime('08:45:00', '%H:%M:%S')
            return 'Late' if clock_in > threshold else 'On Time'
        except:
            return 'Invalid'

    return pd.DataFrame({
        col: df.apply(lambda row: classify(row[col], row['Emp Id']), axis=1)
        for col in cols_in
    })


# NEW
def calculate_early(df, cols_out):
    """
    Flags early leave, absent, on-time, and holidays for each employee.

    Parameters
    ----------
    df : pd.DataFrame
        Attendance dataframe with clock_out columns.
    cols_out : list[str]
        Columns like ['clock_out_08_01', 'clock_out_08_02', ...].

    Returns
    -------
    pd.DataFrame
        Flags per employee per day with values:
        'Holiday', 'Absent', 'Early Leave', 'On Time', or 'Invalid'.
    """
    threshold = datetime.strptime("15:45:00", "%H:%M:%S")
    early_leave_map = {}

    # --- Step 1: Detect Holiday columns (>= 90% leave early) ---
    for col in cols_out:
        times = pd.to_datetime(
            df[col].astype(str).replace("0", pd.NA),
            errors="coerce",
            format="%H:%M:%S"
        )
        total_count = times.notna().sum()
        early_count = (times < threshold).sum()

        if total_count > 0 and (early_count / total_count) >= 0.9:
            early_leave_map[col] = "Holiday"
        else:
            early_leave_map[col] = "Normal"

    # --- Step 2: Flag per row ---
    result = pd.DataFrame(index=df.index)

    for col in cols_out:
        flags = []
        holiday_status = early_leave_map[col]

        if holiday_status == "Holiday":
            result[col] = "Holiday"
            continue

        for val in df[col]:
            if str(val) == "0" or pd.isna(val):
                flags.append("Absent")
            else:
                try:
                    clock_out = datetime.strptime(str(val), "%H:%M:%S")
                    if clock_out < threshold:
                        flags.append("Early Leave")
                    else:
                        flags.append("On Time")
                except Exception:
                    flags.append("Invalid")
        result[col] = flags

    return result

def calculate_working_days(df, threshold=0.95):
    """
    Identify valid working days from biometric data by excluding dates where >= 90% employees are absent.
    
    Parameters:
    - df: DataFrame with biometric clock-in data.
    - threshold: Percentage of absentees to mark a day as non-working (default 90%).

    Returns:
    - List of working days in 'MM_DD' format.
    """

    # Step 1: Extract relevant clock_in columns
    clock_in_cols = [col for col in df.columns if col.startswith('clock_in_')]
    
    working_days = []

    for col in clock_in_cols:
        total_employees = len(df)
        
        # Count how many entries are '0' or NaN (considered absent)
        absent_count = df[col].apply(lambda x: pd.isna(x) or str(x).strip() == '0').sum()
        
        absent_ratio = absent_count / total_employees

        if absent_ratio < threshold:
            # This is considered a working day
            working_days.append(col.replace('clock_in_', ''))

    return working_days



# this detect holidays
def detect_holidays(df_clock_in, threshold=0.9):
    """
    Detect holidays in biometric data by marking dates (columns) 
    where the absentee ratio is >= threshold.

    Parameters:
    - df_clock_in (pd.DataFrame): Clock-in data.
    - threshold (float): Ratio of absentees required to mark as holiday (default 0.9).

    Returns:
    - List[str]: Columns considered holidays.
    """
    clock_in_cols = [col for col in df_clock_in.columns if col.startswith("clock_in_")]
    if not clock_in_cols:
        return []

    total_employees = len(df_clock_in)

    # Absent if NaN or '0'
    absent_mask = df_clock_in[clock_in_cols].isna() | df_clock_in[clock_in_cols].astype(str).eq("0")

    # Ratio of absentees per column
    absent_ratio = absent_mask.sum(axis=0) / total_employees

    # Columns where absentee ratio ≥ threshold → holidays
    holiday_cols = [col for col in clock_in_cols if absent_ratio[col] >= threshold]

    return holiday_cols

def drop_columns_by_prefix(df, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return df.loc[:, ~df.columns.str.startswith(tuple(prefixes))]

def weighted_sum_and_replace_columns(df, cols, new_col, weights):
    if len(cols) != len(weights):
        raise ValueError("Length mismatch between cols and weights")
    df[new_col] = np.dot(df[cols], weights)
    return df.drop(columns=cols)

def calculate_date_month(df, date_row_idx=6):
    """
    Extract full date labels (e.g., '3_01', '3_02', ..., '4_01') from a biometric DataFrame.

    Assumes:
    - Date range string is in the first cell of the first row: 'March-01-2024 To April-01-2024'
    - Raw day numbers are in `date_row_idx` (default 6).
    """

    # --- Step 1: Extract and parse date range ---
    date_range = str(df.iloc[0, 0])
    try:
        start_date, end_date = date_range.split(" To ")
        start_month = datetime.strptime(start_date.split("-")[0].strip(), "%B").month
        end_month = datetime.strptime(end_date.split("-")[0].strip(), "%B").month
        month_numbers = [start_month, end_month]
    except Exception as e:
        raise ValueError(f"Invalid date range format in first cell: {date_range}") from e

    # --- Step 2: Build date labels ---
    raw_dates = df.iloc[date_row_idx]
    dates_full = []
    month_index = 0
    prev_day = 0

    for item in raw_dates:
        try:
            day = int(str(item).strip())
        except ValueError:
            # Non-numeric (e.g., NaN, 'Clock In')
            dates_full.append("nan")
            continue

        # Detect month change
        if day < prev_day and month_index < len(month_numbers) - 1:
            month_index += 1

        prev_day = day
        full_date = f"{month_numbers[month_index]}_{day:02d}"
        dates_full.append(full_date)

    return dates_full

def get_attendance_data(label: str, kind: str = "all"):
    """
    Retrieve attendance data from session_state.

    Parameters
    ----------
    label : str
        One of ["GIMT", "GIPS", "ADMIN", "LEAVE", "EXEMPTED"].
    kind : str
        - For GIMT, GIPS, ADMIN → one of ["all", "in", "out"]
        - For LEAVE, EXEMPTED → use "raw"

    Returns
    -------
    pd.DataFrame or None
    """
    splits = st.session_state.get("attendance_splits", {})
    if label not in splits:
        return None

    return splits[label].get(kind)

def move_columns(df, col_index_map):
    cols = list(df.columns)
    for col, new_index in sorted(col_index_map.items(), key=lambda x: x[1]):
        if col in cols:
            cols.insert(new_index, cols.pop(cols.index(col)))
    return df[cols]  # <-- Ensure it returns a DataFrame


def weighted_sum_and_replace_columns(
    df: pd.DataFrame,
    columns_to_sum: list,
    new_column_name: str,
    weights: list,
    fillna=0,
    drop: bool = True,
):
    """
    Compute a weighted sum of multiple columns, create a new column, 
    and optionally drop originals.

    Args:
        df : pd.DataFrame
            Input DataFrame
        columns_to_sum : list[str]
            Columns to combine
        new_column_name : str
            Name of the new weighted column
        weights : list[float]
            Weights matching the columns
        fillna : scalar, default 0
            Value to fill NaNs before computation
        drop : bool, default True
            Whether to drop the original columns after creating the weighted sum

    Returns:
        pd.DataFrame
    """
    # --- Validation ---
    if len(columns_to_sum) != len(weights):
        raise ValueError("Length of columns_to_sum and weights must match.")

    missing = [col for col in columns_to_sum if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    # --- Weighted sum ---
    arr = df[columns_to_sum].fillna(fillna).to_numpy(dtype=float)
    weighted = np.dot(arr, np.array(weights, dtype=float))
    df[new_column_name] = weighted

    # --- Drop originals if requested ---
    if drop:
        df.drop(columns=columns_to_sum, inplace=True)

    return df

def calculate_leave_summary_with_wd_leaves_old(df, working_days_list):
    """
    Processes leave data, aggregates leave types, and calculates 'Total WD leaves'
    (working day leaves excluding 'Extraordinary Leave' and 'Casual Leave').
    Correctly handles half-day leaves (0.5).

    Assumes 'From Date' and 'To Date' are already converted to datetime dtype
    before calling this function.

    Args:
        df (pd.DataFrame): The raw DataFrame containing leave data.
                            Expected columns include 'Emp Id', 'Name',
                            'Leave Type', 'From Date', 'To Date', 'Status', 'Total Days'.
        working_days_list (list): A list of strings in 'm_d' format (e.g., '3_21' for March 21st)
                                  representing working days. The year is assumed to be the current year.

    Returns:
        pd.DataFrame: A DataFrame summarizing leave data per employee, including
                      a 'Total WD leaves' column.
    """
    # --- Part 1: Initial data cleaning and aggregation ---

    df_processed = df.copy()
    df_processed.columns = df_processed.columns.str.strip().str.replace("\xa0", "", regex=True)

    # Rename columns if needed
    rename_map = {
        "Employee ID": "Emp Id",
        "Name": "Name",
        "Leave Type": "Leave Type",
        "From Date": "From Date",
        "To Date": "To Date",
        "Status": "Status",
        "Total Days": "Total Days",
    }
    df_processed.rename(
        columns={col: rename_map[col] for col in rename_map if col in df_processed.columns},
        inplace=True,
    )

    # Keep only approved leaves
    df_approved = df_processed[df_processed["Status"] == "Approved"].copy()

    # Aggregate leave types
    results = {}
    for _, row in df_approved.iterrows():
        emp_id = row["Emp Id"]
        name = row["Name"]
        leave_type = row["Leave Type"]
        total_days = float(row["Total Days"]) if pd.notnull(row["Total Days"]) else 0

        if emp_id not in results:
            results[emp_id] = {"Emp Id": emp_id, "Name": name}

        results[emp_id][leave_type] = results[emp_id].get(leave_type, 0) + total_days

    leave_summary = pd.DataFrame(results.values())

    # Fix naming
    if "Duty Leave assigned by the University" in leave_summary.columns:
        leave_summary.rename(
            columns={"Duty Leave assigned by the University": "Duty Leave (GCU)"}, inplace=True
        )

    leave_summary.fillna(0, inplace=True)

    # --- Part 2: Calculate 'Total WD leaves' ---

    # Working days → set of datetime.date objects
    current_year = datetime.now().year
    working_days_set = set()
    for wd_str in working_days_list:
        try:
            month, day = map(int, wd_str.split("_"))
            working_days_set.add(datetime(current_year, month, day).date())
        except ValueError:
            continue

    total_wd_leaves_map = {}

    for _, row in df_approved.iterrows():
        emp_id = row["Emp Id"]
        leave_type = row["Leave Type"]
        from_date = row["From Date"]
        to_date = row["To Date"]
        total_days_for_entry = float(row["Total Days"]) if pd.notnull(row["Total Days"]) else 0

        # Skip Extraordinary and Casual leave for *Total WD leaves*,
        # but they are still in leave_summary
        if leave_type in ["Extraordinary Leave", "Casual Leave"]:
            continue

        current_wd_leaves_for_entry = 0

        if total_days_for_entry == 0.5:
            if pd.notnull(from_date) and from_date.date() in working_days_set:
                current_wd_leaves_for_entry = 0.5

        elif total_days_for_entry > 0:
            if pd.notnull(from_date) and pd.notnull(to_date):
                delta = timedelta(days=1)
                current_date = from_date
                while current_date <= to_date:
                    if current_date.date() in working_days_set:
                        current_wd_leaves_for_entry += 1.0
                    current_date += delta

        total_wd_leaves_map[emp_id] = total_wd_leaves_map.get(emp_id, 0) + current_wd_leaves_for_entry

    # Merge back into summary
    total_wd_leaves_df = (
        pd.DataFrame.from_dict(total_wd_leaves_map, orient="index", columns=["Total WD leaves"])
        .rename_axis("Emp Id")
        .reset_index()
    )

    final_leave_summary = pd.merge(leave_summary, total_wd_leaves_df, on="Emp Id", how="left")
    final_leave_summary["Total WD leaves"].fillna(0, inplace=True)

    return final_leave_summary


def calculate_leave_summary_with_wd_leaves(df, working_days_list, debug_emp_id=None):
    """
    Safer/revised version:
    - normalizes columns
    - coerces From/To to datetimes
    - treats Total Days as authoritative and caps the counted working-days contribution
    - handles half-days (0.5)
    - debug_emp_id -> prints per-row details for that employee
    """
    # copy + normalize column names (strip only; keep case for display)
    dfp = df.copy()
    dfp.columns = [str(c).strip() for c in dfp.columns]

    # canonical column names (do not force if they don't exist)
    rename_map = {
        "Employee ID": "Emp Id",
        "EmployeeID": "Emp Id",
        "EmpID": "Emp Id",
        "FromDate": "From Date",
        "ToDate": "To Date",
        "Total Days": "Total Days",
        "TotalDays": "Total Days",
        "LeaveType": "Leave Type",
    }
    present_renames = {k:v for k,v in rename_map.items() if k in dfp.columns}
    if present_renames:
        dfp.rename(columns=present_renames, inplace=True)

    # ensure necessary columns exist (best-effort)
    for col in ["Emp Id", "Name", "Leave Type", "From Date", "To Date", "Status", "Total Days"]:
        if col not in dfp.columns:
            dfp[col] = pd.NA

    # coerce dates
    dfp["From Date"] = pd.to_datetime(dfp["From Date"], errors="coerce")
    dfp["To Date"]   = pd.to_datetime(dfp["To Date"], errors="coerce")

    # coerce total days numeric
    dfp["Total Days"] = pd.to_numeric(dfp["Total Days"], errors="coerce").fillna(0.0)

    # normalize status and leave type strings
    dfp["Status_norm"] = dfp["Status"].astype(str).fillna("").str.strip().str.lower()
    dfp["LeaveType_norm"] = dfp["Leave Type"].astype(str).fillna("").str.strip()

    # filter approved rows
    df_approved = dfp[dfp["Status_norm"] == "approved"].copy()

    # --- aggregate leave-type totals (per employee) ---
    results = {}
    for _, row in df_approved.iterrows():
        emp = row["Emp Id"]
        name = row["Name"]
        lt = row["LeaveType_norm"]
        tot = float(row["Total Days"]) if pd.notnull(row["Total Days"]) else 0.0
        if emp not in results:
            results[emp] = {"Emp Id": emp, "Name": name}
        results[emp][lt] = results[emp].get(lt, 0.0) + tot

    leave_summary = pd.DataFrame(list(results.values())).fillna(0.0)

    # --- build working-days set (assumes working_days_list like ['3_21','3_22', ...]) ---
    # We use current year for those m_d pairs (you can modify if needed)
    current_year = datetime.now().year
    working_days_set = set()
    for wd in working_days_list:
        try:
            m, d = wd.split("_")
            working_days_set.add(datetime(current_year, int(m), int(d)).date())
        except Exception:
            # skip malformed entries
            continue

    # --- calculate Total WD leaves per approved entry, but CAP by Total Days value ---
    total_wd_map = {}
    debug_rows = []

    for idx, row in df_approved.iterrows():
        emp = row["Emp Id"]
        lt_raw = row["LeaveType_norm"]
        lt_key = lt_raw.strip()
        total_days_entry = float(row["Total Days"]) if pd.notnull(row["Total Days"]) else 0.0
        fd = row["From Date"]
        td = row["To Date"]

        # exclude casual and extraordinary from WD count
        if lt_key.lower() in {"casual leave", "extraordinary leave"}:
            if debug_emp_id and emp == debug_emp_id:
                debug_rows.append({
                    "emp": emp, "lt": lt_key, "reason": "excluded leave type", "total_days": total_days_entry
                })
            continue

        # compute working days present in the date-range
        counted_wd = 0.0
        if total_days_entry == 0.5:
            # half-day: count only if the from_date exists and is a working day
            if pd.notnull(fd) and fd.date() in working_days_set:
                counted_wd = 0.5
            else:
                counted_wd = 0.0
        elif total_days_entry > 0:
            # if both dates present, compute actual working-day count in the range
            if pd.notnull(fd) and pd.notnull(td):
                # create date range and count working days intersection
                try:
                    # use pd.date_range with freq='D'
                    dr = pd.date_range(start=fd.normalize(), end=td.normalize(), freq='D')
                    counted_days = sum(1 for d in dr if d.date() in working_days_set)
                except Exception:
                    # fallback to manual loop
                    counted_days = 0
                    cur = fd
                    while pd.notnull(cur) and cur <= td:
                        if cur.date() in working_days_set:
                            counted_days += 1
                        cur += timedelta(days=1)
                # cap by the reported total_days
                counted_wd = min(float(counted_days), total_days_entry)
            else:
                # missing to_date: treat as single-day or as total_days_entry but ensure working-day check
                if pd.notnull(fd) and fd.date() in working_days_set:
                    counted_wd = min(1.0, total_days_entry)
                else:
                    counted_wd = 0.0

        # accumulate
        total_wd_map[emp] = total_wd_map.get(emp, 0.0) + counted_wd

        if debug_emp_id and emp == debug_emp_id:
            debug_rows.append({
                "emp": emp,
                "leave_type": lt_key,
                "from": fd,
                "to": td,
                "total_days_entry": total_days_entry,
                "counted_wd_in_range": counted_wd
            })

    # debug printing if requested
    if debug_emp_id:
        print(f"DEBUG rows for Emp Id = {debug_emp_id}")
        # show raw approved rows for this emp
        emp_rows = df_approved[df_approved["Emp Id"] == debug_emp_id]
        if emp_rows.empty:
            print("No approved rows found for this emp.")
        else:
            print("Approved rows (raw):")
            print(emp_rows[["Emp Id", "Leave Type", "Total Days", "From Date", "To Date"]].to_string(index=False))
        print("Computed per-entry contributions:")
        for r in debug_rows:
            print(r)

    # convert map to df and merge
    total_wd_df = (
        pd.DataFrame.from_dict(total_wd_map, orient="index", columns=["Total WD leaves"])
        .rename_axis("Emp Id")
        .reset_index()
    )

    final = pd.merge(leave_summary, total_wd_df, on="Emp Id", how="left")
    final["Total WD leaves"] = final["Total WD leaves"].fillna(0.0)

    # ensure numeric columns are floats
    numeric_cols = [c for c in final.columns if c not in ["Emp Id", "Name"]]
    final[numeric_cols] = final[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return final



def calculate_working_days(df, threshold=0.95):
    """
    Identify valid working days from biometric data by excluding dates where >= 90% employees are absent.
    
    Parameters:
    - df: DataFrame with biometric clock-in data.
    - threshold: Percentage of absentees to mark a day as non-working (default 90%).

    Returns:
    - List of working days in 'MM_DD' format.
    """

    # Step 1: Extract relevant clock_in columns
    clock_in_cols = [col for col in df.columns if col.startswith('clock_in_')]
    
    working_days = []

    for col in clock_in_cols:
        total_employees = len(df)
        
        # Count how many entries are '0' or NaN (considered absent)
        absent_count = df[col].apply(lambda x: pd.isna(x) or str(x).strip() == '0').sum()
        
        absent_ratio = absent_count / total_employees

        if absent_ratio < threshold:
            # This is considered a working day
            working_days.append(col.replace('clock_in_', ''))

    return working_days

# this function calculate the exempted leaves
def process_exempted_leaves(file) : #, working_days):
    #xls = pd.ExcelFile(file_path)
    sheet_names = ['late', 'half_day', 'full_day']
    merged_df = None

    for sheet in sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)
        # Handle both 'Emp Id' and 'Emp ID' column names
        if 'Emp ID' in df.columns:
            df.rename(columns={'Emp ID': 'Emp Id'}, inplace=True)
        df = df.dropna(subset=['Emp Id'])  # skip empty rows
        
        leave_pairs = df.columns[2:]
        valid_dates = [leave_pairs[i] for i in range(0, len(leave_pairs), 2)]  # assume every 2nd col is date
        
        count = df[valid_dates].notna().sum(axis=1)  # no //2 — each date is one leave

        temp_df = df[['Emp Id', 'Name']].copy()
        temp_df[f'{sheet}_count'] = count

        merged_df = temp_df if merged_df is None else pd.merge(merged_df, temp_df, on=['Emp Id', 'Name'], how='outer')

    merged_df.fillna(0, inplace=True)

    # Convert all leave counts to float
    for col in merged_df.columns[2:]:
        merged_df[col] = merged_df[col].astype(float)

    #merged_df = drop_columns_if_exist(merged_df, ['Name', 'Names'])

    return merged_df

def merge_with_emp_data(df_fac_all, emp_df, col_index_map=None):
    """
    Merge biometric faculty data with ERP employee data.

    Parameters:
    -----------
    df_fac_all : pd.DataFrame
        Faculty attendance data (from biometric).
    emp_df : pd.DataFrame
        Employee master data (from ERP).
    col_index_map : dict, optional
        Mapping of columns to their desired positions.

    Returns:
    --------
    df_fac_with_ID : pd.DataFrame
        Merged DataFrame with Name, Designation, Department.
    unmatched_ids : list
        List of Emp Ids not found in ERP data.
    """

    # Ensure Emp Id is clean in both DataFrames
    df_fac_all["Emp Id"] = df_fac_all["Emp Id"].astype(str).str.strip()
    emp_df["Emp Id"] = emp_df["Emp Id"].astype(str).str.strip()

    # Drop duplicate 'Names' from biometric
    df_fac_all = df_fac_all.drop(columns=["Names"], errors="ignore")

    # Merge with ERP data
    df_fac_with_ID = pd.merge(df_fac_all, emp_df, how="left", on="Emp Id")

    # Find unmatched IDs
    unmatched_ids = df_fac_with_ID[df_fac_with_ID["Name"].isna()]["Emp Id"].unique().tolist()

    # Handle missing ERP values
    df_fac_with_ID["Name"].fillna("Unknown", inplace=True)
    df_fac_with_ID["Designation"].fillna("Unknown", inplace=True)
    df_fac_with_ID["Department"].fillna("Unknown", inplace=True)

    # Move columns if mapping provided
    if col_index_map:
        df_fac_with_ID = move_columns(df_fac_with_ID, col_index_map)

    return df_fac_with_ID, unmatched_ids

def preprocess_date(value):
    if pd.isna(value) or value is None or (isinstance(value, str) and not value.strip()):
        return None
    
    if isinstance(value, pd.Timestamp):
        return value.date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        cleaned_value = value.strip().replace("/", "-")
        date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"]

        for fmt in date_formats:
            try:
                return datetime.strptime(cleaned_value, fmt).date()
            except (ValueError, TypeError):
                continue
    
    return None

import pandas as pd

def pad_month_in_columns(df, prefix):
    """
    Pads the month part of column names with a leading zero (e.g., '9_18' to '09_18')
    for columns starting with a specified prefix.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        prefix (str): The starting string of the columns to be renamed 
                      (e.g., 'clock_in_', 'clock_out_').

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    # --- Step 1: remove columns with Unnamed.........
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # -- Step 2: do padding for mm_dd
    rename_map = {}
    
    # Ensure the prefix ends with an underscore for easier splitting, e.g., 'clock_in' -> 'clock_in_'
    if not prefix.endswith('_'):
        search_prefix = prefix + '_'
    else:
        search_prefix = prefix

    for old_col in df.columns:
        if old_col.startswith(search_prefix):
            # The part after the prefix will be the date: '9_18'
            date_part = old_col[len(search_prefix):] 
            
            # Split the date part: '9_18' -> ['9', '18']
            parts = date_part.split('_')
            
            # We expect exactly two parts: month and day
            if len(parts) == 2:
                month = parts[0] # '9'
                day = parts[1]   # '18'
                
                # Pad the month with a leading zero (e.g., '9' becomes '09')
                padded_month = month.zfill(2)
                padded_day = day.zfill(2)
                
                # Reconstruct the new column name
                new_col = f"{search_prefix}{padded_month}_{padded_day}"

                # Add to the rename map if the name has actually changed
                if old_col != new_col:
                    rename_map[old_col] = new_col

    # Rename the columns in the DataFrame in place
    df.rename(columns=rename_map, inplace=True)
    # print(f"DEBUG: pad_month_in_columns renamed {len(rename_map)} columns with prefix '{search_prefix}'. Examples: {list(rename_map.items())[:3]}")
 
    return df
def detect_holidays_staffs(df_clock_in, year=None, misc_holidays=None, misc_working_days="", verbose=True):
    """
    Detect holidays for staff attendance data.

    Rules:
    - Sundays and 1st & 3rd Saturdays are holidays.
    - misc_holidays (list[str]) are *always* treated as holidays if they fall within attendance period.
    - misc_working_days override everything and are *always* working days.

    Parameters
    ----------
    df_clock_in : pd.DataFrame
        Clock-in dataframe with columns like 'clock_in_09_10', 'clock_in_09_11', etc.
    year : int, optional
        Year of attendance period (default = current year).
    misc_holidays : list[str], optional
        List of manual holidays (e.g., ['29-sep-2024', '30-sep-2024']).
    misc_working_days : str
        Comma-separated list of dates to force as working days.
    verbose : bool
        Whether to print debug info.

    Returns
    -------
    list[str]
        List of column names (e.g., ['clock_in_09_06', 'clock_in_09_07', ...]) that are holidays.
    """

    clock_in_cols = [c for c in df_clock_in.columns if c.startswith("clock_in_")]
    if not clock_in_cols:
        if verbose:
            print("⚠️ No 'clock_in_' columns found.")
        return []

    current_year = year or datetime.now().year
    misc_holidays = misc_holidays or []

    # ---------- Helper functions ----------
    def try_parse_date(s):
        s = s.strip()
        for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Invalid date format: {s}")

    def col_variants(d):
        """Generate possible clock_in column name formats for a given date"""
        m, day = d.month, d.day
        return {
            f"clock_in_{m}_{day}",
            f"clock_in_{m:02d}_{day}",
            f"clock_in_{m}_{day:02d}",
            f"clock_in_{m:02d}_{day:02d}",
        }

    def parse_misc_list(txt):
        if not txt.strip():
            return []
        return [try_parse_date(x) for x in txt.split(",") if x.strip()]

    def find_matching_cols(dates):
        matched = set()
        for d in dates:
            for variant in col_variants(d):
                if variant in clock_in_cols:
                    matched.add(variant)
                    break
        return matched

    # ---------- Determine attendance date range ----------
    date_objs = []
    for col in clock_in_cols:
        try:
            parts = col.split("_")
            m, d = int(parts[-2]), int(parts[-1])
            date_objs.append(datetime(current_year, m, d))
        except Exception:
            continue

    if not date_objs:
        if verbose:
            print("⚠️ No valid clock-in date columns found.")
        return []

    start_date, end_date = min(date_objs), max(date_objs)

    # ---------- Filter misc_holidays within date range ----------
    misc_holiday_dates = []
    for h in misc_holidays:
        try:
            dt = try_parse_date(h)
            if start_date <= dt <= end_date:
                misc_holiday_dates.append(dt)
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipping invalid misc holiday '{h}': {e}")

    misc_holiday_cols = find_matching_cols(misc_holiday_dates)
    misc_working_cols = find_matching_cols(parse_misc_list(misc_working_days))

    # ---------- Calendar-based holidays ----------
    calendar_based = set()
    for col in clock_in_cols:
        try:
            parts = col.split("_")
            month, day = int(parts[-2]), int(parts[-1])
            date_obj = datetime(current_year, month, day)
            weekday = date_obj.weekday()  # Mon=0 ... Sun=6
            week_num = (date_obj.day - 1) // 7 + 1
            if weekday == 6 or (weekday == 5 and week_num in [1, 3]):
                calendar_based.add(col)
        except Exception:
            continue

    # ---------- Combine all ----------
    holidays = set(calendar_based)
    holidays.update(misc_holiday_cols)
    holidays.difference_update(misc_working_cols)

    # ---------- Normalize column names ----------
    normalized_holidays = []
    for col in holidays:
        parts = col.split("_")
        if len(parts) >= 4:
            try:
                m, d = int(parts[-2]), int(parts[-1])
                normalized_holidays.append(f"clock_in_{m:02d}_{d:02d}")
            except:
                normalized_holidays.append(col)
        else:
            normalized_holidays.append(col)

    all_holidays = sorted(set(normalized_holidays))

    if verbose:
        print(f"📅 Attendance period: {start_date.strftime('%d-%b-%Y')} → {end_date.strftime('%d-%b-%Y')}")
        print(f"📆 Calendar holidays: {sorted(calendar_based)}")
        print(f"➕ Manual holidays (within range): {sorted(misc_holiday_cols)}")
        print(f"➖ Manual working days: {sorted(misc_working_cols)}")
        print(f"🎯 Final holidays: {all_holidays}")

    return all_holidays

# this is final
def merge_files_staffs(df_admin_in, df_admin_out, emp_df, no_working_days,
                       misc_holidays="", misc_working_days=""):
    """
    Merge IN/OUT attendance data with employee details and compute daily & summary flags.

    Automatically detects the year from date columns (e.g. 'clock_in_9_17').

    Parameters
    ----------
    df_admin_in : pd.DataFrame
        DataFrame containing clock-in times (columns like 'clock_in_9_10', ...).
    df_admin_out : pd.DataFrame
        DataFrame containing clock-out times (columns like 'clock_out_9_10', ...).
    emp_df : pd.DataFrame
        Employee details with columns ['Emp ID', 'Name', 'Designation', 'Department'].
    no_working_days : int
        Total number of working days in the period.
    misc_holidays : str
        Comma-separated list of extra holidays (dd-mmm-yyyy).
    misc_working_days : str
        Comma-separated list of days that should be treated as working even if normally holidays.

    Returns
    -------
    pd.DataFrame
        Summary attendance report per employee.
    """
    # --- Step 2: Identify date columns ---
    clock_in_cols = [c for c in df_admin_in.columns if c.startswith('clock_in_')]
    clock_out_cols = [c for c in df_admin_out.columns if c.startswith('clock_out_')]

    if not clock_in_cols or not clock_out_cols:
        raise ValueError("No clock_in_ or clock_out_ columns found in input dataframes.")

    # --- Step 3: Infer year automatically ---
    # Pick the middle date column, e.g. 'clock_in_9_17'
    sample_col = clock_in_cols[len(clock_in_cols)//2]
    parts = sample_col.split("_")
    month, day = int(parts[-2]), int(parts[-1])
    today = datetime.now()
    inferred_year = today.year
    # optional heuristic: if month ahead of current month (e.g., data for past year)
    if month > today.month + 1:
        inferred_year -= 1

    # --- Step 4: Merge IN/OUT data ---
    df_admin_in.rename(columns={'Names': 'Name'}, inplace=True)
    emp_df.rename(columns={'Name': 'Name'}, inplace=True)

    df = pd.merge(df_admin_in[['Emp Id', 'Name'] + clock_in_cols],
                  df_admin_out[['Emp Id'] + clock_out_cols],
                  on='Emp Id',
                  how='outer')

    # --- Step 5: Merge employee details ---
    df = pd.merge(df, emp_df[['Emp Id', 'Designation', 'Department']],
                  on='Emp Id', how='left')

    # --- Step 6: Detect holidays (including misc overrides) ---
    """ --- this has been removed
    holiday_cols = detect_holidays_staffs(
        df[clock_in_cols],
        year=inferred_year,
        misc_holidays=misc_holidays,
        misc_working_days=misc_working_days,
        verbose=False
    )
    """
    # --- Step 7: Compute late/early flags ---
    late_flags_df = calculate_late(df, clock_in_cols)
    early_flags_df = calculate_early(df, clock_out_cols)

    results = []

    # --- Step 8: Process each employee ---
    for idx, row in df.iterrows():
        emp_id = row['Emp Id']
        name = row['Name']
        desig = str(row.get('Designation', '')).strip().lower()

        late_flags, early_flags = [], []
        am_abs, pm_abs = [], []
        full_abs_dates, half_day_dates = [], []

        for col_in, col_out in zip(clock_in_cols, clock_out_cols):
            dd_mm = col_in.replace('clock_in_', '')
            #if col_in in holiday_cols:
            #    continue  # skip processing for holidays

            val_in = str(row[col_in])
            val_out = str(row[col_out])

            # Full day absent
            if val_in == '0' and val_out == '0':
                full_abs_dates.append(dd_mm)
                continue

            # Half-day absent
            # Check if one is '0' (absent) and the other has a time value (present)
            if val_in == '0' and val_out != '0':
                # Present in PM only (clock_out has value)
                half_day_dates.append(dd_mm)
                am_abs.append(dd_mm)
            elif val_in != '0' and val_out == '0':
                # Present in AM only (clock_in has value)
                half_day_dates.append(dd_mm)
                pm_abs.append(dd_mm)
            elif val_in != '0' and val_out != '0':
                # Both have values - check for late arrival or early departure
                # Check if clock_in is after 10:30 AM (late arrival = AM absence)
                try:
                    clock_in_time = datetime.strptime(val_in, '%H:%M:%S').time()
                    if clock_in_time > datetime.strptime('10:30:00', '%H:%M:%S').time():
                        half_day_dates.append(dd_mm)
                        am_abs.append(dd_mm)
                except (ValueError, TypeError):
                    pass
                
                # Check if clock_out is between 12:15 PM and 3:30 PM (early departure = PM absence)
                try:
                    clock_out_time = datetime.strptime(val_out, '%H:%M:%S').time()
                    out_low = datetime.strptime('12:15:00', '%H:%M:%S').time()
                    out_high = datetime.strptime('15:30:00', '%H:%M:%S').time()
                    if out_low <= clock_out_time < out_high:
                        half_day_dates.append(dd_mm)
                        pm_abs.append(dd_mm)
                except (ValueError, TypeError):
                    pass

            # Late and early flags
            if late_flags_df.loc[idx, col_in] == 'Late':
                late_flags.append(dd_mm)
            if early_flags_df.loc[idx, col_out] == 'Early Leave':
                early_flags.append(dd_mm)

            # Driver exception rule
            if 'driver' in desig:
                if (val_in != '0') or (val_out != '0'):
                    continue
                else:
                    full_abs_dates.append(dd_mm)

        # --- Step 9: Calculate totals ---
        # Separate full-day absences from half-day absences
        # Count unique half-days by combining am_abs and pm_abs into a set
        half_day_set = set(am_abs) | set(pm_abs)
        total_abs_days = len(full_abs_dates) + 0.5 * len(half_day_set)

        results.append({
            'Emp Id': emp_id,
            'Name': name,
            'Designation': row.get('Designation', ''),
            'Department': row.get('Department', ''),
            'late_flags': ', '.join(late_flags),
            'early_flags': ', '.join(early_flags),
            'half_day_flags': ', '.join(half_day_dates),
            'AM_abs': ', '.join(am_abs),
            'PM_abs': ', '.join(pm_abs),
            'days_abs': ', '.join(full_abs_dates),
            'No_of_AM_abs': len(am_abs),
            'No_of_PM_abs': len(pm_abs),
            'actual_No_of_late': len(late_flags),
            'actual_half_day': len(half_day_dates),
            'actual_full_day': len(full_abs_dates),
            'Working Days': no_working_days,
            'Present': no_working_days - total_abs_days,
            'Absent': total_abs_days
          })

    if not results:
        raise ValueError("No employee records processed.")

    return pd.DataFrame(results)

def _is_rerun_exc(ex):
    try:
        if getattr(ex, "is_fragment_scoped_rerun", False):
            return True
    except Exception:
        pass
    tname = type(ex).__name__
    if "Rerun" in tname or "rerun" in tname.lower():
        return True
    try:
        if "RerunData" in repr(ex) or "rerun" in repr(ex).lower():
            return True
    except Exception:
        pass
    try:
        mod = getattr(type(ex), "__module__", "")
        if mod and "streamlit" in mod:
            return True
    except Exception:
        pass
    return False