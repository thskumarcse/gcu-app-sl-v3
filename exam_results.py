import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io
from utility import connect_gsheet

# ReportLab imports
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Table, TableStyle,
    Spacer, Paragraph, SimpleDocTemplate, NextPageTemplate, 
    KeepInFrame, KeepTogether, PageBreak, Flowable
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from xml.sax.saxutils import escape

# Helper functions from transcript
def safe_round(value, ndigits=2, default=0):
    try:
        return round(float(value), ndigits)
    except (ValueError, TypeError):
        return default

def safe_paragraph(text, style):
    """Wrap text safely in a Paragraph, escaping XML chars"""
    if text is None:
        text = ""
    return Paragraph(escape(str(text)), style)

def clean_text_for_reportlab(text):
    if text is None:
        return ""
    text = str(text).replace('\xa0', ' ').replace('\u2013', '-').strip()
    return text

def safe_text(val, default=""):
    """Ensure text is safe for ReportLab Paragraphs/drawString."""
    if pd.isna(val) or val is None:
        return default
    return str(val)

def safe_number_text(val, default=""):
    """Ensure numbers don't come in scientific notation."""
    try:
        return str(int(float(val)))
    except Exception:
        return default

def safe_number(x):
    """Ensure numeric values are safe for ReportLab (no NaN)."""
    try:
        if pd.isna(x):
            return 0
        return int(x) if float(x).is_integer() else round(float(x), 2)
    except Exception:
        return 0

class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbering."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add total page count to each page."""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        page = f"Page {self._pageNumber} of {page_count}"
        self.setFont("Helvetica", 9)
        width, height = A4
        self.drawCentredString(width / 2.0, 30, page)

def fix_streamlit_layout():
    """Fix Streamlit layout issues"""
    # Page config is handled by main.py
    pass

def set_compact_theme():
    """Set compact theme for better UI"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

def draw_header_with_logo(canvas, doc, info, logo_path):
    """Draw header with logo (no student photo for results)."""
    try:
        width, height = A4
        usable_width = width - doc.leftMargin - doc.rightMargin

        # Logo
        try:
            if logo_path and os.path.exists(logo_path):
                canvas.drawImage(logo_path, 50, height - 140, width=80, height=80, mask='auto')
            else:
                print(f"⚠️ Logo not found at {logo_path}")
        except Exception as e:
            print(f"❌ Error drawing logo: {e}")
            pass

        # Titles
        canvas.setFont("Times-Roman", 18)
        canvas.drawCentredString(width / 2, height - 60, "Girijananda Chowdhury University")
        canvas.setFont("Times-Roman", 14)
        canvas.drawCentredString(width / 2, height - 85, "EXAMINATION RESULTS")
        canvas.setFont("Times-Roman", 12)
        canvas.drawCentredString(width / 2, height - 100, f"End Semester/Annual Examination, {info['session']}, {info['year']}")
        canvas.drawCentredString(width / 2, height - 115, f"Program: {info['program']}")
        
        # --------- the following is new added code ---------
        
        # Student data table
        info_data = [
            [f"Total Appeared: {info['total students appeared']}", f"All Cleared: {info['students passed']}"],
            [f"{info['annual_semester']}: {info['semester']} ({info['type']})", f"Pass Percentage: {info['pass percent']}%"],
            [f"Withheld: {info['students witheld']}", f"With Backlogs: {info['students failed']}"],
       
           
        ]
        info_table = Table(info_data, colWidths=[250, 250])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("ALIGN", (0,0), (0,-1), "LEFT"),
            ("ALIGN", (1,0), (1,-1), "LEFT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        table_width, table_height = info_table.wrap(0, 0)
        info_table.drawOn(canvas, 50, height - 150 - table_height)
        
    # --------- end of new added code ---------
    
    except Exception as e:
        print(f"❌ Error in draw_header_with_logo: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def draw_footer(canvas, doc, date_value):
    """Draw footer with date and signature."""
    width, height = A4
    canvas.setFont("Helvetica", 10)
    canvas.drawString(50, 60, f"Date : {date_value}")
    canvas.drawRightString(width - 55, 70, "Controller of Examination")
    canvas.drawRightString(width - 40, 60, "Girijananda Chowdhury University")
   
    
def app():
    fix_streamlit_layout()
    set_compact_theme()
    
    st.header('📊 Program-wise Compilation of Results')
    st.info("📋 Upload CSV from ERP and get consolidated results: Pass, Promoted with backlogs, and Withheld students")
    
    # Getting data from the user ====================================================================================
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8 = st.columns(2)

    with col1:
        session = st.selectbox("Select session:", options = ["Monsoon","Winter"])
    with col2:
        year = st.selectbox("Select year:",
            options = ["2023","2024", "2025","2026","2027","2028", "2029","2030","2031","2032", "2033","2034"])
    with col3:
        annual_semester = st.selectbox("Academic Type (Year/Semester):", options = ["Year","Semester"])
    with col4:
        type = st.selectbox("Select exam type:", options = ["Regular","Repeater"])
    with col5:
        semester = st.selectbox("Annual/Semester No.:", options = ["1","2","3","4","5","6","7","8"])
    with col6:
        date = st.date_input("Result declaration date:") 
    with col7:
        program = st.text_input("Enter Program:") 
    # End of getting data from user ============================================================================   
    
    # Convert date to string format (dd-mm-yyyy)
    date_str = date.strftime("%d-%m-%Y") if date else ""
    
    info = {"session": session,
            "year":year,
            "program":program,
            "semester":semester,
            "type":type,
            "date":date_str,
            "annual_semester":annual_semester}
    
    #st.write(info["annual_semester"])
    #st.write(info["program"])
    # This section processes the leave data from ERP
    
    # Get input data in .CSV format
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📁 Upload Result CSV File", 
        type=['csv'],
        help="Upload the result CSV file from ERP system"
    )
    
    if uploaded_file is not None:
        # Try multiple reading strategies with more robust options
        reading_strategies = [
            {"encoding": "utf-8", "skiprows": 0, "sep": ",", "name": "UTF-8, comma separator"},
            {"encoding": "utf-8", "skiprows": 0, "sep": "\t", "name": "UTF-8, tab separator"},
            {"encoding": "ISO-8859-1", "skiprows": 0, "sep": ",", "name": "ISO-8859-1, comma separator"},
            {"encoding": "ISO-8859-1", "skiprows": 0, "sep": "\t", "name": "ISO-8859-1, tab separator"},
            {"encoding": "latin-1", "skiprows": 0, "sep": ",", "name": "Latin-1, comma separator"},
            {"encoding": "latin-1", "skiprows": 0, "sep": "\t", "name": "Latin-1, tab separator"},
            {"encoding": "cp1252", "skiprows": 0, "sep": ",", "name": "CP1252, comma separator"},
            {"encoding": "cp1252", "skiprows": 0, "sep": "\t", "name": "CP1252, tab separator"},
            {"encoding": "utf-8", "skiprows": 6, "sep": ",", "name": "UTF-8, 6 rows skipped, comma"},
            {"encoding": "ISO-8859-1", "skiprows": 6, "sep": ",", "name": "ISO-8859-1, 6 rows skipped, comma"},
        ]
        
        result = None
        successful_strategy = None
        
        for strategy in reading_strategies:
            try:
                st.write(f"🔄 Trying: {strategy['name']}")
                result = pd.read_csv(
                    uploaded_file, 
                    encoding=strategy['encoding'], 
                    skiprows=strategy['skiprows'],
                    sep=strategy['sep'],
                    on_bad_lines='skip'  # Skip problematic lines
                )
                if not result.empty and len(result.columns) > 5:  # Ensure we have meaningful data
                    st.success(f"✅ Success with {strategy['name']}! Shape: {result.shape}")
                    successful_strategy = strategy
                    break
                else:
                    st.write(f"⚠️ Empty or insufficient data with {strategy['name']}")
            except Exception as e:
                st.write(f"❌ Failed: {str(e)}")
                continue
        
        if result is None:
            st.error("❌ All reading strategies failed. Trying fallback method...")
            
            # Fallback: Try to read the file content directly and parse manually
            try:
                uploaded_file.seek(0)  # Reset file pointer
                content = uploaded_file.read()
                
                # Try different encodings to decode the content
                for encoding in ['utf-8', 'ISO-8859-1', 'latin-1', 'cp1252']:
                    try:
                        text_content = content.decode(encoding)
                        st.write(f"✅ Successfully decoded with {encoding}")
                        
                        # Split into lines and try to find the header row
                        lines = text_content.split('\n')
                        for i, line in enumerate(lines[:10]):  # Check first 10 lines
                            if 'Student ID' in line and 'Student Name' in line:
                                st.write(f"Found header at line {i+1}: {line[:100]}...")
                                
                                # Method 1: Try with file pointer reset
                                uploaded_file.seek(0)
                                try:
                                    result = pd.read_csv(uploaded_file, encoding=encoding, skiprows=i)
                                    if not result.empty and len(result.columns) > 5:
                                        st.success(f"✅ Fallback success! Shape: {result.shape}")
                                        break
                                    else:
                                        st.write(f"⚠️ Empty or insufficient data with skiprows={i}")
                                except Exception as e:
                                    st.write(f"❌ Failed to read with skiprows={i}: {str(e)}")
                                
                                # Method 2: Try with StringIO (more reliable)
                                try:
                                    from io import StringIO
                                    # Create a new string with the content starting from the header row
                                    content_from_header = '\n'.join(lines[i:])
                                    string_io = StringIO(content_from_header)
                                    result = pd.read_csv(string_io)
                                    if not result.empty and len(result.columns) > 5:
                                        st.success(f"✅ Fallback success with StringIO! Shape: {result.shape}")
                                        break
                                    else:
                                        st.write(f"⚠️ Empty or insufficient data with StringIO")
                                except Exception as e:
                                    st.write(f"❌ Failed to read with StringIO: {str(e)}")
                                
                                continue
                        if not result.empty:
                            break
                    except:
                        continue
                        
            except Exception as e:
                st.error(f"❌ Fallback method also failed: {str(e)}")
                return None
            
            if result is None or result.empty:
                st.error("❌ All methods failed. Please check your CSV file format.")
                return None
        
        # Show raw data before any processing
        st.write("**🔍 Raw CSV Data (first 10 rows):**")
        st.dataframe(result.head(10))
        
        # Show all column names
        st.write("**📋 All Column Names Found:**")
        for i, col in enumerate(result.columns, 1):
            st.write(f"{i}. '{col}'")
            
    else:
        st.warning("⚠️ Please upload a CSV file to proceed")
        return None
    
    # Debug: Show column names and first few rows
    debug_mode = True  # Set to False to disable debugging
    if debug_mode:
        st.write("**📋 CSV Columns found:**")
        st.write(list(result.columns))
        st.write("**📊 First few rows:**")
        st.dataframe(result.head(3))
    
    # Check if columns are numbered (no proper headers) or contain data values
    st.write("**🔍 Checking column format...**")
    st.write(f"Column names: {list(result.columns)}")
    st.write(f"Are columns numbered? {all(str(i) == str(col) for i, col in enumerate(result.columns))}")
    
    # Check if columns contain proper headers
    has_proper_headers = any(col in ['Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status', 'Course Variant'] for col in result.columns)
    st.write(f"Has proper headers? {has_proper_headers}")
    
    if all(str(i) == str(col) for i, col in enumerate(result.columns)) or not has_proper_headers:
        st.warning("⚠️ CSV appears to have data values as column names instead of proper headers. Using manual mapping...")
        
        # Manual column mapping based on the data structure you showed
        # Based on your data: 5=Student ID, 6=Student Name, 7=Maximum Marks, 9=Obtained Marks, 16=Status
        manual_mapping = {
            '5': 'Student ID',
            '6': 'Student Name', 
            '7': 'Maximum Marks',
            '9': 'Obtained Marks',
            '16': 'Status',
            '2': 'Course Variant'  # This appears to be the course variant based on the data
        }
        
        st.write(f"**🔧 Applying manual mapping:** {manual_mapping}")
        
        # Apply manual mapping
        result = result.rename(columns=manual_mapping)
        st.write(f"✅ Manual column mapping applied!")
        
        # Show current columns after mapping
        st.write(f"**📋 Columns after mapping:** {list(result.columns)}")
        
        # Show the mapped data
        st.write("**📊 Data after manual mapping:**")
        try:
            st.dataframe(result[['Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status', 'Course Variant']].head())
        except KeyError as e:
            st.error(f"❌ Error displaying mapped data: {e}")
            st.write("Available columns:", list(result.columns))
        
        # Verify all required columns are present
        required_columns = ['Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status', 'Course Variant']
        missing_after_mapping = [col for col in required_columns if col not in result.columns]
        
        if missing_after_mapping:
            st.error(f"❌ Still missing columns after manual mapping: {missing_after_mapping}")
            st.write("Available columns after mapping:", list(result.columns))
            return None
        else:
            st.success("✅ All required columns found after manual mapping!")
        
    else:
        # Original flexible matching for proper headers
        required_columns = ['Course Variant', 'Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status']
        
        # Try to find columns with flexible matching
        column_mapping = {}
        missing_columns = []
        
        for req_col in required_columns:
            found = False
            for actual_col in result.columns:
                # Check for exact match
                if actual_col.strip() == req_col:
                    column_mapping[req_col] = actual_col
                    found = True
                    break
                # Check for case-insensitive match
                elif actual_col.strip().lower() == req_col.lower():
                    column_mapping[req_col] = actual_col
                    found = True
                    break
                # Check for partial match
                elif req_col.lower() in actual_col.lower() or actual_col.lower() in req_col.lower():
                    column_mapping[req_col] = actual_col
                    found = True
                    break
            
            if not found:
                missing_columns.append(req_col)
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.write("Available columns:", list(result.columns))
            st.write("Column mapping found:", column_mapping)
            return None
        
        # Rename columns to standard names
        result = result.rename(columns=column_mapping)
        st.write(f"✅ Column mapping successful: {column_mapping}")
    
    # Extract course code from course variant
    try:
        result['course code'] = result['Course Variant'].apply(lambda x: str(x).split('-')[0] if pd.notna(x) else '')
    except Exception as e:
        st.error(f"❌ Error processing Course Variant column: {str(e)}")
        st.write("Course Variant column values:", result['Course Variant'].head())
        return None
    
    # delete columns (only if they exist)
    cols_to_delete = ['Serial No.', 'Course Name', 'Course Variant','Assessment Scheme','Admission ID','Maximum Grades',  \
                  'Effective Marks', 'Grade Obtained','German Grade Scale', 'Grade Point', 'Section Wise Course Rank', 'Course Rank']
    
    # Only delete columns that actually exist
    existing_cols_to_delete = [col for col in cols_to_delete if col in result.columns]
    if existing_cols_to_delete:
        result = result.drop(existing_cols_to_delete, axis=1)
        if debug_mode:
            st.write(f"✅ Removed columns: {existing_cols_to_delete}")
    
    # Check if essential columns still exist after deletion
    essential_columns = ['Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status', 'course code']
    missing_essential = [col for col in essential_columns if col not in result.columns]
    
    if missing_essential:
        st.error(f"❌ Missing essential columns after processing: {missing_essential}")
        if debug_mode:
            st.write("Remaining columns:", list(result.columns))
        return None
    
    # groups the dataset w.r.t to Students name and picks the first entries
    try:
        result['Maximum Marks'] = pd.to_numeric(result['Maximum Marks'], errors='coerce')
        result['Obtained Marks'] = pd.to_numeric(result['Obtained Marks'], errors='coerce')
        
        if debug_mode:
            st.write("**📊 Data after numeric conversion:**")
            st.write(f"Shape: {result.shape}")
            st.write("Sample data:")
            st.dataframe(result[['Student ID', 'Student Name', 'Maximum Marks', 'Obtained Marks', 'Status']].head())
        
        result_obtained_marks = result.groupby('Student ID').agg({'Obtained Marks': [lambda x: x.iloc[0], 'sum']})
        
    except Exception as e:
        st.error(f"❌ Error in data processing: {str(e)}")
        st.write("Data types:", result.dtypes)
        return None
    
    # this section finds all the student with your total obtained marks
    try:
        final_result_obtained_marks = pd.DataFrame(result_obtained_marks['Obtained Marks']['sum'])
        final_result_obtained_marks['Student ID']=result_obtained_marks.index
        final_result_obtained_marks.reset_index(drop=True, inplace=True)
    except Exception as e:
        st.error(f"❌ Error processing student marks: {str(e)}")
        return None
    
    # Total marks of the examination
    total_marks_series = result.groupby('Student ID')['Maximum Marks'].sum()
    if len(total_marks_series) == 0:
        st.error("❌ No student data found for total marks calculation")
        return None
    
    total_marks = total_marks_series.iloc[0]
    if total_marks == 0:
        st.error("❌ Total marks is zero - cannot calculate CGPA")
        return None
    
    final_result_obtained_marks['CGPA'] = final_result_obtained_marks['sum']/total_marks*10
    final_result_obtained_marks['CGPA'] = np.round(final_result_obtained_marks['CGPA'],2)
    
    # This gives students with CGPA - single student multiple entries
    try:
        final_result_with_cgpa = pd.merge(result,final_result_obtained_marks, how='right', on='Student ID')
        
        # CGPA is being dropped for the time being
        final_result_with_cgpa.drop('CGPA', axis=1, inplace=True)
    except Exception as e:
        st.error(f"❌ Error merging student data: {str(e)}")
        return None
    
    # FAILED Studnets
    failed_student_multiple = final_result_with_cgpa[final_result_with_cgpa['Status']=='Fail']
    #failed_students = failed_student_multiple[['Student ID','Student Name','CGPA']]
    failed_students = failed_student_multiple[['Student ID','Student Name']]
    failed_students = failed_students.drop_duplicates()
    no_students_failed = len(failed_students)
    
    fail_students_with_subject = failed_student_multiple.groupby(['Student ID'], as_index = False).agg({'course code': ','.join})
    fail_students_with_subject = pd.merge(failed_students,fail_students_with_subject, how='right', on='Student ID') 
    fail_students_with_subject ################## pass parameter to report
    
    # PASSED Students
    passed_student_multiple = final_result_with_cgpa[final_result_with_cgpa['Status']=='Pass']
    
    # This consists of some student who failed in at least one subject
    #passed_failed_students = passed_student_multiple[['Student ID','Student Name','CGPA']]
    passed_failed_students = passed_student_multiple[['Student ID','Student Name']]
    passed_failed_students = passed_failed_students.drop_duplicates()
    final_passed_student = passed_failed_students[~passed_failed_students.apply(tuple,1).isin(failed_students.apply(tuple,1))]
    no_students_passed = len(final_passed_student)
    
    # WITHELD students
    witheld_student = final_result_with_cgpa[final_result_with_cgpa['Status']=='Withheld']
    witheld_student.drop(['Maximum Marks','Obtained Marks','Status','course code','sum'], axis=1, inplace=True)
    witheld_student = witheld_student.drop_duplicates()
    no_students_witheld = len(witheld_student)
    
    total_students_appeared = no_students_passed + no_students_failed + no_students_witheld
 
    info["total students appeared"]= total_students_appeared
    info["students passed"] = no_students_passed
    info["students failed"] = no_students_failed
    info["students witheld"] = no_students_witheld
    info["pass percent"] = np.round(no_students_passed/total_students_appeared*100, 2)
    
    # Display summary
    st.markdown("---")
    st.subheader("📊 Result Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Appeared", total_students_appeared)
    with col2:
        st.metric("All Cleared", no_students_passed, f"{info['pass percent']}%")
    with col3:
        st.metric("With Backlogs", no_students_failed)
    with col4:
        st.metric("Withheld", no_students_witheld)
    
    # This section generates the report in PDF
    def create_pdf(info, df_pass, df_fail, df_witheld):
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = BaseDocTemplate(buffer, pagesize=A4)
        doc.showFooter = True

        # Logo path
        logo_dir = os.path.join(os.getcwd(), "logo_dir")
        logo_path = os.path.join(logo_dir, "logo.png")
        if not os.path.exists(logo_path):
            logo_path = None

        # Styles
        styles = getSampleStyleSheet()
        
        # Table style matching transcript
        table_style = TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,-1), "Times-Roman"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ])

        # Frames
        frame_first = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 160, id='first_frame')
        frame_later = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height + 50, id='later_frame')

        # Page templates
        first_page_template = PageTemplate(
            id='FirstPage',
            frames=[frame_first],
            onPage=lambda c, d: draw_header_with_logo(c, d, info, logo_path)
        )
        
        middle_page_template = PageTemplate(
            id='MiddlePages',
            frames=[frame_later]
        )
        
        last_page_template = PageTemplate(
            id='LastPage',
            frames=[frame_later],
            onPageEnd=lambda c, d: draw_footer(c, d, info['date'])
        )
        
        doc.addPageTemplates([first_page_template, middle_page_template, last_page_template])

        # Elements
        elements = []
        
        # Start with first page template
        elements.append(NextPageTemplate('FirstPage'))
        
        # Add tables conditionally
        table_count = 0
        if len(df_pass) > 0:
            table_count += 1
        if len(df_fail) > 0:
            table_count += 1
        if len(df_witheld) > 0:
            table_count += 1
        
        current_table = 0
        first_table_added = False
        
        if len(df_pass) > 0:
            current_table += 1
            if not first_table_added:
                first_table_added = True
            else:
                elements.append(NextPageTemplate('MiddlePages'))
            
            if current_table == table_count:
                elements.append(NextPageTemplate('LastPage'))
            
            elements.extend(create_students_table(df_pass, "The following candidate(s) has/have cleared all the course(s)", table_style))
            elements.append(Spacer(1, 10))
        
        if len(df_fail) > 0:
            current_table += 1
            if not first_table_added:
                first_table_added = True
            else:
                elements.append(NextPageTemplate('MiddlePages'))
            
            if current_table == table_count:
                elements.append(NextPageTemplate('LastPage'))
            
            elements.extend(create_failed_students_table(df_fail, table_style))
            elements.append(Spacer(1, 20))
        
        if len(df_witheld) > 0:
            current_table += 1
            if not first_table_added:
                first_table_added = True
            else:
                elements.append(NextPageTemplate('MiddlePages'))
            
            if current_table == table_count:
                elements.append(NextPageTemplate('LastPage'))
            
            elements.extend(create_students_table(df_witheld, "The following candidate(s)'s result has been withheld:", table_style))
            elements.append(Spacer(1, 20))

        # Build PDF
        try:
            doc.build(elements, canvasmaker=NumberedCanvas)
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"❌ PDF build failed: {e}")
            return None
        
    def create_students_table(df, label, table_style):
        """Create a table for passed/witheld students"""
        # Prepare data
        data = [["SL No.", "Student ID", "Student Name"]]
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            data.append([
                str(i),
                str(row.get('Student ID', '')),
                str(row.get('Student Name', ''))
            ])
        
        # Create table
        table = Table(data, colWidths=[60, 120, 200])
        table.setStyle(table_style)
        
        # Label style matching transcript
        label_style = ParagraphStyle(
            'Label',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        # Keep label and table together
        return [KeepTogether([Paragraph(label, label_style), Spacer(1, 6), table])]
            
    def create_failed_students_table(df, table_style):
        """Create a table for failed students with subjects"""
        # Prepare data
        data = [["SL No.", "Student ID", "Student Name", "Failed Subjects"]]
        
        # Style for failed subjects text wrapping
        failed_subjects_style = ParagraphStyle(
            'FailedSubjects',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=9,
            fontName='Times-Roman',
            alignment=TA_LEFT,
            leading=11
        )
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            failed_subjects_text = str(row.get('course code', ''))
            # Create a Paragraph for text wrapping
            failed_subjects_para = Paragraph(failed_subjects_text, failed_subjects_style)
            
            data.append([
                str(i),
                str(row.get('Student ID', '')),
                str(row.get('Student Name', '')),
                failed_subjects_para
            ])
        
        # Create table
        table = Table(data, colWidths=[50, 100, 150, 250])
        table.setStyle(table_style)
        
        # Label style matching transcript
        label_style = ParagraphStyle(
            'Label',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        
        label = "The following candidate(s) has/have not cleared in the particular course(s) shown:"
        # Keep label and table together
        return [KeepTogether([Paragraph(label, label_style), Spacer(1, 6), table])]
    
    
    # Main report generator        
    try:
        pdf_buffer = create_pdf(info, final_passed_student, fail_students_with_subject, witheld_student)
        
        if pdf_buffer is not None:
            # Download button
            st.success("✅ Result report generated successfully!")
            st.download_button(
                label="📥 Download Result Report",
                data=pdf_buffer.getvalue(),
                file_name=f'result_report_{info["year"]}_{info["session"]}.pdf',
                mime="application/pdf"
            )
        else:
            st.error("❌ Failed to generate PDF. Please check the error messages above.")
    except Exception as e:
        st.error(f"❌ Error generating PDF: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
    

        
