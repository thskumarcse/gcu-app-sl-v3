"""HR Attendance app (restored pipeline, guarded against reruns)

This module restores the HR attendance processing pipeline but only
executes the heavy processing when all required uploads are present and
when session-state bytes are stable. It uses `read_session_bytes_with_retry`
to avoid transient Streamlit rerun-control errors and marks processing
as done in `st.session_state['hr_processed']` to avoid duplicate runs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Import helpers from utility modules
from utility_attendance import (
    stepwise_file_upload, read_session_bytes_with_retry, process_exempted_leaves,_is_rerun_exc,
    split_file, pad_month_in_columns, detect_holidays_staffs, calculate_working_days,
    merge_files_staffs, calculate_leave_summary_with_wd_leaves, weighted_sum_and_replace_columns
)
from utility import preprocess_date

# Default holiday list (can be extended via UI)
HOLIDAY_LIST = ['29-sep-2025','30-sep-2025','01-oct-2025','02-oct-2025','03-oct-2025',
                '06-oct-2025','18-oct-2025','20-oct-2025','21-oct-2025','23-sep-2025',
                '05-nov-2025','25-dec-2025','13-jan-2026','14-jan-2026','15-jan-2026',
                '16-jan-2026','26-jan-2026','03-mar-2026','21-mar-2026','03-apr-2026',
                '13-apr-2026','14-apr-2026', '15-apr-2026','16-apr-2026','17-apr-2026',
                '01-may-2026','27-may-2026','15-aug-2026','04-sep-2026','12-sep-2026',
                '17-sep-2026','02-oct-2026','19-oct-2026','20-oct-2026','21-oct-2026',
                '22-oct-2026','23-oct-2026','24-oct-2026','25-oct-2026','08-nov-2026',
                '24-nov-2026','25-dec-2026']


def app():
    st.title("HR Attendance")

    # Stepwise upload for main attendance files
    labels = ["GIMT", "GIPS", "ADMIN", "LEAVE"]
    dfs = stepwise_file_upload(labels, key_prefix="attendance")

    st.markdown("---")
    st.write("Upload the Exempted Leaves file below (xlsx/xls).")

    exempted_file = st.file_uploader(
        "Exempted Leaves (xlsx/xls)",
        type=["xlsx", "xls"],
        key="exempted_uploader",
        label_visibility="visible"
    )

    # Persist uploaded bytes
    if exempted_file is not None:
        try:
            name_key = "exempted_bytes_name"
            bytes_key = "exempted_bytes"
            if st.session_state.get(name_key) != exempted_file.name:
                st.session_state[bytes_key] = exempted_file.read()
                st.session_state[name_key] = exempted_file.name
                #st.success(f"Stored {exempted_file.name} in session state")
        except Exception as e:
            if _is_rerun_exc(e):
                raise
            st.error(f"Failed to read uploaded file: {e}")

    # If processing already completed, show download buttons and a clear control
    if st.session_state.get("hr_processed"):
        st.success("Processing already completed for this session. Clear session to re-run.")
        # Show persisted downloads if available
        fac_bytes = st.session_state.get('hr_faculty_xlsx')
        adm_bytes = st.session_state.get('hr_admin_xlsx')
        if fac_bytes or adm_bytes:
            col1, col2 = st.columns(2)
            with col1:
                if fac_bytes:
                    st.download_button(
                        label="📥 Download Faculty Report",
                        data=fac_bytes,
                        file_name=f"faculty_attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="hr_faculty_download_stored",
                    )
            with col2:
                if adm_bytes:
                    st.download_button(
                        label="📥 Download Admin Report",
                        data=adm_bytes,
                        file_name=f"admin_attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="hr_admin_download_stored",
                    )
        """
        if st.button("Clear HR session data"):
            for k in [
                'hr_processed', 'hr_faculty_xlsx', 'hr_admin_xlsx',
                'exempted_bytes', 'exempted_bytes_name'
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()
        return
        """
    # Trigger automatic processing when we have all files and the exempted bytes
    if len(dfs) == len(labels) and "exempted_bytes" in st.session_state:
        #st.info("All files uploaded — starting processing...")
        try:
            # Read exempted bytes robustly
            bio_bytes = read_session_bytes_with_retry("exempted_bytes", attempts=5, delay=0.12)
            bio = io.BytesIO(bio_bytes)
            df_exempted = process_exempted_leaves(bio)

            # BEGIN: core processing (adapted from original pipeline)
            df_gimt = dfs.get("GIMT")
            df_admin = dfs.get("ADMIN")
            df_gips = dfs.get("GIPS")
            df_leave_erp = dfs.get("LEAVE")

            # Split and pad columns
            df_gimt_all, df_gimt_in, df_gimt_out = split_file(df_gimt)
            df_gips_all, df_gips_in, df_gips_out = split_file(df_gips)
            df_admin_all, df_admin_in, df_admin_out = split_file(df_admin)

            df_gimt_in = pad_month_in_columns(df_gimt_in, 'clock_in')
            df_gips_in = pad_month_in_columns(df_gips_in, 'clock_in')
            df_admin_in = pad_month_in_columns(df_admin_in, 'clock_in')
            df_gimt_out = pad_month_in_columns(df_gimt_out, 'clock_out')
            df_gips_out = pad_month_in_columns(df_gips_out, 'clock_out')
            df_admin_out = pad_month_in_columns(df_admin_out, 'clock_out')

            # Holidays detection & removal
            misc_holidays = st.text_input("Enter list of Misc. holidays : dd-mm-yyyy, comma-separated", value="")
            misc_working_days = st.text_input("Enter list of Misc. Working days : dd-mm-yyyy, comma-separated", value="")

            # Button placed right after the text input
            if st.button("Proceed"):
                misc_holidays_list = [h.strip() for h in misc_holidays.split(',') if h.strip()] if misc_holidays else []
                all_holidays = HOLIDAY_LIST.copy()
                if misc_holidays_list:
                    all_holidays.extend(misc_holidays_list)

                holidays = detect_holidays_staffs(df_gimt_in, year=datetime.now().year, misc_holidays=all_holidays, misc_working_days=misc_working_days, verbose=False)

                cols_to_delete_in = holidays
                cols_to_delete_out = [c.replace('clock_in', 'clock_out') for c in holidays]

                df_gimt_in = df_gimt_in.drop(columns=cols_to_delete_in, axis=1, errors='ignore')
                df_gimt_out = df_gimt_out.drop(columns=cols_to_delete_out, axis=1, errors='ignore')
                df_gips_in = df_gips_in.drop(columns=cols_to_delete_in, axis=1, errors='ignore')
                df_gips_out = df_gips_out.drop(columns=cols_to_delete_out, axis=1, errors='ignore')
                df_admin_in = df_admin_in.drop(columns=cols_to_delete_in, axis=1, errors='ignore')
                df_admin_out = df_admin_out.drop(columns=cols_to_delete_out, axis=1, errors='ignore')

                final_working_days = df_gimt_in.columns
                working_days_list = calculate_working_days(df_gimt_in)
                no_working_days = len(df_gimt_in.columns) - 3 if len(df_gimt_in.columns) > 3 else len(df_gimt_in.columns)

                # Merge with ERP employee master
                emp_df = pd.read_csv('./data/2015_10_27_employee_list.csv', skiprows=6, encoding='windows-1252')
                emp_df = emp_df[['Employee ID','Name','Designation','Department']]
                emp_df = emp_df.rename(columns={'Employee ID':'Emp Id'})
                emp_df.reset_index(drop=True, inplace=True)
                
                # extract employee names
                emp_names_df = emp_df[['Emp Id', 'Name']].copy()

                # Merge files for staffs - SAVE 1
                df_gimt_merged = merge_files_staffs(df_gimt_in, df_gimt_out, emp_df.copy(), no_working_days, all_holidays, misc_working_days)
                df_gips_merged = merge_files_staffs(df_gips_in, df_gips_out, emp_df.copy(), no_working_days, all_holidays, misc_working_days)
                df_admin_merged = merge_files_staffs(df_admin_in, df_admin_out, emp_df.copy(), no_working_days, all_holidays, misc_working_days)

                df_fac_conso = pd.concat([df_gimt_merged, df_gips_merged], ignore_index=True)
                df_admin_conso = df_admin_merged.copy()

                # Process LEAVE ERP data
                if 'From Date' in df_leave_erp.columns:
                    df_leave_erp['From Date'] = df_leave_erp['From Date'].apply(preprocess_date)
                    df_leave_erp['From Date'] = pd.to_datetime(df_leave_erp['From Date'], errors='coerce')
                if 'To Date' in df_leave_erp.columns:
                    df_leave_erp['To Date'] = df_leave_erp['To Date'].apply(preprocess_date)
                    df_leave_erp['To Date'] = pd.to_datetime(df_leave_erp['To Date'], errors='coerce')

                df_leave_erp_summary = calculate_leave_summary_with_wd_leaves(df_leave_erp, working_days_list)
                df_leave_erp_summary.fillna(0, inplace=True)
                df_leave_erp_summary['Approved leaves (ERP)'] = df_leave_erp_summary.get('Total WD leaves', 0) + df_leave_erp_summary.get('Casual Leave', 0)

                # Merge EXEMPTED and calculate adjusted values
                df_exempted.rename(columns={'late_count':'exempt_late','half_day_count':'exempt_HD','full_day_count':'exempt_FD'}, inplace=True)
                if 'Name' in df_exempted.columns:
                    df_exempted.drop('Name',axis=1,inplace=True)

                df_fac_actual_exempted = pd.merge(df_fac_conso, df_exempted, how='left', on=['Emp Id'])
                df_admin_actual_exempted = pd.merge(df_admin_conso, df_exempted, how='left', on=['Emp Id'])
                df_fac_actual_exempted.fillna(0, inplace=True)
                df_admin_actual_exempted.fillna(0, inplace=True)

                for df in [df_fac_actual_exempted, df_admin_actual_exempted]:
                    if not df.empty:
                        actual_am = pd.to_numeric(df.get('actual_AM_abs', 0), errors='coerce').fillna(0) if 'actual_AM_abs' in df.columns else 0
                        actual_pm = pd.to_numeric(df.get('actual_PM_abs', 0), errors='coerce').fillna(0) if 'actual_PM_abs' in df.columns else 0
                        actual_days_abs = pd.to_numeric(df.get('actual_days_abs', 0), errors='coerce').fillna(0) if 'actual_days_abs' in df.columns else 0
                        actual_late = pd.to_numeric(df.get('actual_No_of_late', 0), errors='coerce').fillna(0) if 'actual_No_of_late' in df.columns else 0

                        exempt_hd = pd.to_numeric(df.get('exempt_HD', 0), errors='coerce').fillna(0) if 'exempt_HD' in df.columns else 0
                        exempt_fd = pd.to_numeric(df.get('exempt_FD', 0), errors='coerce').fillna(0) if 'exempt_FD' in df.columns else 0
                        exempt_late = pd.to_numeric(df.get('exempt_late', 0), errors='coerce').fillna(0) if 'exempt_late' in df.columns else 0

                        df['Half Days'] = (actual_am + actual_pm - exempt_hd).clip(lower=0)
                        df['Full Days'] = (actual_days_abs - exempt_fd).clip(lower=0)
                        df['Late'] = (actual_late - exempt_late).clip(lower=0)

                if 'Name' in df_fac_actual_exempted.columns:
                    df_fac_actual_exempted.drop('Name',axis=1,inplace=True)
                if 'Name' in df_admin_actual_exempted.columns:
                    df_admin_actual_exempted.drop('Name',axis=1,inplace=True)
                
                df_fac_report = pd.merge(df_fac_actual_exempted, df_leave_erp_summary, how='left', on='Emp Id', suffixes=('','_leave'))
                df_admin_report = pd.merge(df_admin_actual_exempted, df_leave_erp_summary, how='left', on='Emp Id', suffixes=('','_leave'))
                df_fac_report.fillna(0, inplace=True)
                df_admin_report.fillna(0, inplace=True)

                df_fac_report = weighted_sum_and_replace_columns(df_fac_report, ['Half Days','Full Days'], 'Observed Leaves', [0.5,1.0])
                df_admin_report = weighted_sum_and_replace_columns(df_admin_report, ['Half Days','Full Days'], 'Observed Leaves', [0.5,1.0])

                # NEED to bring names from erp and merge properly
                # Calculate Unauthorized leaves = Absent - Total WD leaves
                #df_fac_report["Unauthorized leaves"] = (df_fac_report["Absent"] - df_fac_report["Approved leaves (ERP)"]).clip(lower=0) 
                #df_admin_report["Unauthorized leaves"] = (df_admin_report["Absent"] - df_admin_report["Approved leaves (ERP)"]).clip(lower=0)
                
                df_fac_report["Unauthorized leaves"] = (df_fac_report["Absent"] - df_fac_report["Approved leaves (ERP)"]).clip(lower=0)
                df_admin_report["Unauthorized leaves"] = (df_admin_report["Absent"] - df_admin_report["Approved leaves (ERP)"]).clip(lower=0)
                
                        
                
                # - merge names from ERP
                if 'Name' in df_fac_report.columns:
                    df_fac_report.drop('Name', axis=1, inplace=True)
                if 'Name' in df_admin_report.columns:
                    df_admin_report.drop('Name', axis=1, inplace=True)
                df_fac_report = pd.merge(df_fac_report, emp_names_df, how='left', on='Emp Id', suffixes=('','_erp'))
                df_admin_report = pd.merge(df_admin_report, emp_names_df, how='left', on='Emp Id', suffixes=('','_erp'))
                df_fac_report.fillna(0, inplace=True)
                df_admin_report.fillna(0, inplace=True)
                
                cols_selected = ['Emp Id', 'Name', 'Designation', 'Department', 'Working Days', 'Present','Absent', 'Late',
                                'Approved leaves (ERP)', 'Unauthorized leaves']
                df_fac_report = df_fac_report[[col for col in cols_selected if col in df_fac_report.columns]]
                df_admin_report = df_admin_report[[col for col in cols_selected if col in df_admin_report.columns]]
                
                # Final formatting and display
                for _df in (df_fac_report, df_admin_report):
                    for _col in ['Emp Id', 'Name', 'Designation', 'Department']:
                        if _col in _df.columns:
                            _df[_col] = _df[_col].astype(str).fillna('')

                st.subheader("👨‍🏫 Faculty Report")
                st.dataframe(df_fac_report)
                st.subheader("👨‍💼 Admin Report")
                st.dataframe(df_admin_report)

                # Prepare downloadable Excel reports (Faculty and Admin)
                try:
                    # Build buffers and persist them in session_state so downloads
                    # survive Streamlit reruns triggered by clicks on download buttons.
                    faculty_buffer = io.BytesIO()
                    with pd.ExcelWriter(faculty_buffer, engine='openpyxl') as writer:
                        try:
                            df_fac_conso.to_excel(writer, sheet_name='Bio Consolidated', index=False)
                        except Exception:
                            pass
                        try:
                            df_exempted.to_excel(writer, sheet_name='Exempted', index=False)
                        except Exception:
                            pass
                        try:
                            df_leave_erp_summary.to_excel(writer, sheet_name='ERP Leave', index=False)
                        except Exception:
                            pass
                        try:
                            df_fac_report.to_excel(writer, sheet_name='Report', index=False)
                        except Exception:
                            pass
                    faculty_buffer.seek(0)
                    st.session_state['hr_faculty_xlsx'] = faculty_buffer.getvalue()

                    admin_buffer = io.BytesIO()
                    with pd.ExcelWriter(admin_buffer, engine='openpyxl') as writer:
                        try:
                            df_admin_conso.to_excel(writer, sheet_name='Bio Consolidated', index=False)
                        except Exception:
                            pass
                        try:
                            df_exempted.to_excel(writer, sheet_name='Exempted', index=False)
                        except Exception:
                            pass
                        try:
                            df_leave_erp_summary.to_excel(writer, sheet_name='ERP Leave', index=False)
                        except Exception:
                            pass
                        try:
                            df_admin_report.to_excel(writer, sheet_name='Report', index=False)
                        except Exception:
                            pass
                    admin_buffer.seek(0)
                    st.session_state['hr_admin_xlsx'] = admin_buffer.getvalue()

                    # Show download buttons reading from session_state so they remain
                    # available across reruns.
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Faculty Report",
                            data=st.session_state.get('hr_faculty_xlsx'),
                            file_name=f"faculty_attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="hr_faculty_download_generated",
                        )
                    with col2:
                        st.download_button(
                            label="📥 Download Admin Report",
                            data=st.session_state.get('hr_admin_xlsx'),
                            file_name=f"admin_attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="hr_admin_download_generated",
                        )
                except Exception as e:
                    st.warning(f"Could not create download files: {e}")

                # END core processing
                st.session_state['hr_processed'] = True
                st.success("HR Attendance processing completed.")

        except Exception as e:
            if _is_rerun_exc(e):
                raise
            st.error(f"❌ Error in HR processing: {e}")


if __name__ == "__main__":
    app()