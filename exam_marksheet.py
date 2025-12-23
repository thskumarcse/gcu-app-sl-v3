import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import shutil
import io
from datetime import datetime
import warnings
import re
from xml.sax.saxutils import escape
from utility import clean_course_name

"""
this generates marksheets for programs like M.Pharm, M.Tech, etc. 
with two years/4 semesters of study. The result is of two reports per student. 
Each report is of 1-page per year.
It calculates the percentage and result based on the marks obtained.
"""

# ReportLab imports
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Table, TableStyle,
    Spacer, Paragraph, NextPageTemplate, SimpleDocTemplate, 
    KeepInFrame, KeepTogether, PageBreak, Flowable
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")

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

# Helper functions
def safe_round(value, ndigits=2, default=0):
    try:
        return round(float(value), ndigits)
    except (ValueError, TypeError):
        return default

def remove_decimal(value):
    try:
        val = float(value)
        if val.is_integer():
            return int(val)
        return round(val, 2)
    except Exception:
        return value

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

def find_photo(photo_dir, enrollment_no):
    """Find the student's photo file based on enrollment number."""
    if photo_dir is None or not os.path.exists(photo_dir):
        return None
    extensions = ["jpg", "jpeg", "png"]
    for ext in extensions:
        candidate = os.path.join(photo_dir, f"{enrollment_no}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def process_course(df: pd.DataFrame, df_courses: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    rename_map = {}

    # Get the list of course codes
    course_codes = df_courses["course code"].tolist()

    # Base patterns for the first course (no suffix)
    base_patterns = [
        ("Formative (20)", "Formative"),
        ("Summative (80)", "Summative"),
        ("Total Marks", "Total Marks")
    ]

    # Handle the first set (no .1 suffix)
    if len(course_codes) > 0:
        code = str(course_codes[0])
        for old, new_base in base_patterns:
            new_name = f"{new_base} {code}"
            if old in df.columns:
                rename_map[old] = new_name

    # Handle subsequent sets (with .1, .2, etc.)
    for i, code in enumerate(course_codes[1:], start=1):
        suffix = f".{i}"
        for old, new_base in base_patterns:
            old_col = f"{old}{suffix}"
            new_col = f"{new_base} {code}"
            if old_col in df.columns:
                rename_map[old_col] = new_col

    # Rename columns
    df = df.rename(columns=rename_map)
    return df

def process_marks_long_format(df, df_courses=None):
    """
    Convert wide-format student marks DataFrame into long-format.
    Merges with df_courses to include course details like 'year', 'course name', etc.
    """

    # Identify base student info columns
    id_vars = ['Student Code', 'Student Name', 'APAAR ID', 'Serial Number', 'Program Name', 'Batch Name']
    existing_id_vars = [col for col in id_vars if col in df.columns]
    
    # --- Clean numeric-like ID fields ---
    for col in existing_id_vars:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].astype('Int64').astype(str)
        else:
            df[col] = df[col].astype(str)

        # Remove trailing .0 if present (e.g., 2313022001.0 → 2313022001)
        df[col] = df[col].str.replace(r'\.0$', '', regex=True).str.strip()
    

    # Melt to long format
    df_long = df.melt(id_vars=existing_id_vars, var_name='Assessment', value_name='Marks')

    # --- Extract Course Code dynamically ---
    def extract_course_code(row):
        prog = str(row.get('Program Name', '')).strip()
        assess = str(row['Assessment'])
        if prog == 'B.Pharm (Practice)':
            match = re.search(r'(\d+\.\d+)', assess)
        else:
            match = re.search(r'(ER\d{2}-\d{2}[A-Z]?)', assess)
        return match.group(1) if match else None

    df_long['Course Code'] = df_long.apply(extract_course_code, axis=1)

    # --- Extract Assessment Type ---
    df_long['Assessment Type'] = df_long['Assessment'].apply(
        lambda x: 'Formative' if 'Formative' in str(x)
        else ('Summative' if 'Summative' in str(x)
        else ('Total Marks' if 'Total Marks' in str(x) else None))
    )

    # Keep only valid rows
    df_long = df_long.dropna(subset=['Course Code', 'Assessment Type'])

    # Merge with df_courses if provided
    if df_courses is not None and not df_courses.empty:
        df_long['Course Code'] = df_long['Course Code'].astype(str).str.strip()
        df_courses['course code'] = df_courses['course code'].astype(str).str.strip()

        # Perform merge on matching course code
        df_long = df_long.merge(
            df_courses,
            left_on='Course Code',
            right_on='course code',
            how='left'
        )

    # Reorder columns
    base_cols = existing_id_vars + ['Course Code', 'Assessment Type', 'Marks']
    if df_courses is not None:
        merged_cols = [col for col in ['year', 'course name', 'formative', 'summative'] if col in df_long.columns]
        final_columns = base_cols + merged_cols
    else:
        final_columns = base_cols

    return df_long[final_columns]

def draw_header_with_photo(canvas, doc, student_data, logo_path, photo_dir):
    """Draw header with logo and student photo."""
    width, height = A4
    usable_width = width - doc.leftMargin - doc.rightMargin
    
    enrollment_no = safe_text(student_data.iloc[0].get("Student Code", ""))
    student_name = safe_text(student_data.iloc[0].get("Student Name", ""))
    apaar_id = safe_text(student_data.iloc[0].get("APAAR ID", "")) if "APAAR ID" in student_data.columns else "0"
    serial_no = safe_text(student_data.iloc[0].get("Serial Number", "")) if "Serial Number" in student_data.columns else "0"
    program_name = safe_text(student_data.iloc[0].get("Program Name", ""))
    level = safe_text(student_data.iloc[0].get("NCrF Level", "")) if "NCrF Level" in student_data.columns else ""
    
    try:
        if logo_path and os.path.exists(logo_path):
            canvas.drawImage(logo_path, 50, height - 140, width=80, height=80, mask='auto')
        else:
            print(f"⚠️ Logo not found at {logo_path}")
    except Exception as e:
        print(f"❌ Error drawing logo: {e}")
        pass

    # Photo
    photo_file = find_photo(photo_dir, enrollment_no)
    try:
        if photo_file:
            canvas.drawImage(photo_file, 450, height - 140, width=70, height=80, mask='auto')
        else:
            print(f"⚠️ No photo found for {student_name} ({enrollment_no}) in {photo_dir}")
    except Exception as e:
        print(f"❌ Error drawing photo for {enrollment_no}: {e}")
        pass

    # Titles
    canvas.setFont("Times-Roman", 18)
    canvas.drawCentredString(width / 2, height - 60, "Girijananda Chowdhury University")
    canvas.setFont("Times-Roman", 14)
    canvas.drawCentredString(width / 2, height - 85, "MARKSHEET")
    canvas.setFont("Times-Roman", 12)
    canvas.drawCentredString(width / 2, height - 100, program_name)
    canvas.drawCentredString(width / 2, height - 115, "2023-2025")
    canvas.translate(0, -30)
    # Student data table
    std_data = [
        ["APAAR ID", ":", apaar_id, f"Serial Number : {serial_no}"],
        ["Enrollment No.", ":", enrollment_no, ""],
        ["Name", ":", student_name, ""],
        #["NCrF/NHEQF Level", ":", level, ""],
    ]
    std_table = Table(std_data, colWidths=[120, 10, 200, 150])
    std_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
    ]))
    table_width, table_height = std_table.wrap(0, 0)
    std_table.drawOn(canvas, 50, height - 150 - table_height)
    canvas.translate(0, -30)

def draw_footer(canvas, doc, date_value):
    """Draw footer with date and signature."""
    width, height = A4
    canvas.setFont("Helvetica", 10)
    canvas.drawString(50, 100, f"Date : {date_value}")
    canvas.drawRightString(width - 55, 110, "Controller of Examination")
    canvas.drawRightString(width - 40, 100, "Girijananda Chowdhury University")

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

def format_subject_name(text, max_len=45):
    """Returns a Paragraph with adaptive font size."""
    styles = getSampleStyleSheet()
    subject_style = styles["Normal"]
    subject_style.fontName = "Times-Roman"
    subject_style.leading = 9  # Reduced from 11
    subject_style.alignment = TA_LEFT

    if len(str(text)) > max_len:  
        subject_style.fontSize = 7  # Reduced from 8
        subject_style.leading = 8  # Reduced from 10
    elif len(str(text)) > max_len * 1.5:  
        subject_style.fontSize = 6  # Reduced from 7
        subject_style.leading = 7  # Reduced from 9
    else:
        subject_style.fontSize = 8  # Reduced from 9
    return Paragraph(str(text), subject_style)

def generate_pdf_onepage(student_id, student_data, report_date, output_dir, logo_path, photo_dir, suffix=""):
    """Generate single-page PDF transcript with marks (ABC format)."""
    width, height = A4
    filename = os.path.join(output_dir, f"Transcript_Marks_{student_data.iloc[0]['Student Name']}_onepage{suffix}.pdf")
    doc = BaseDocTemplate(filename, pagesize=A4)

    # Styles - reduced for single page
    styles = getSampleStyleSheet()
    subject_style = styles["Normal"]
    subject_style.fontName = "Times-Roman"
    subject_style.fontSize = 8  # Reduced from 9
    subject_style.leading = 9  # Reduced from 11
    subject_style.alignment = TA_LEFT

    bold_subject_style = ParagraphStyle(
        name="SubjectBold",
        parent=subject_style,
        fontName="Times-Bold",
        fontSize=subject_style.fontSize,
        leading=subject_style.leading,
        alignment=subject_style.alignment
    )

    # Center-aligned style for Total Marks values
    center_style = ParagraphStyle(
        name="CenterStyle",
        parent=subject_style,
        alignment=TA_CENTER,
        fontSize=subject_style.fontSize,
        leading=subject_style.leading
    )
    
    bold_center_style = ParagraphStyle(
        name="BoldCenterStyle",
        parent=bold_subject_style,
        alignment=TA_CENTER
    )

    left_year_style = ParagraphStyle(
        name="LeftYearHeading",
        parent=styles['Heading4'],
        alignment=TA_LEFT,
        leftIndent=-30,
        fontName="Helvetica"
    )

    # Frame
    frame_main = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height - 140,
        id="normal_frame"
    )

    # Page template - use exact same pattern as working notebook
    single_page_template = PageTemplate(
        id="OnePage",
        frames=[frame_main],
        onPage=lambda c, d: draw_header_with_photo(c, d, student_data, logo_path, photo_dir),
        onPageEnd=lambda c, d: draw_footer(c, d, report_date)
    )
    doc.addPageTemplates([single_page_template])
    

    # Elements
    elements = []

    # Group by year
    if "year" not in student_data.columns:
        student_data = student_data.copy()
        student_data["year"] = 1

    for year, year_data in student_data.groupby("year"):
        courses = year_data["Course Code"].unique()
        
        # Two-row header: First row with merged cells, second row with sub-columns
        # Row 0: Main header - must have 7 columns, with empty cells where merging happens
        # Row 1: Sub-columns showing Formative and Summative under Max Marks and Max Obtained
        # Create a style for header text with wrapping
        header_style = ParagraphStyle(
            name="HeaderStyle",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=8,
            leading=9,
            alignment=TA_CENTER
        )
        header_row_0 = [
            "Sub Code", 
            "Subject/Papers", 
            "Max Marks", 
            "", 
            "Mark Obtained", 
            "", 
            Paragraph("Total Marks", header_style)  # Wrap Total Marks header
        ]
        header_row_1 = [
            "Sub Code", 
            "Subject/Papers", 
            "Formative", 
            "Summative", 
            "Formative", 
            "Summative", 
            Paragraph("Total Marks", header_style)  # Wrap Total Marks header
        ]
        
        data = [header_row_0, header_row_1]

        for course_code in courses:
            course_rows = year_data[year_data["Course Code"] == course_code]
            
            course_name = ""
            if "course name" in course_rows.columns:
                course_name = safe_text(course_rows.iloc[0].get("course name", ""))
            
            # Get marks obtained by student
            formative_marks_obtained = ""
            summative_marks_obtained = ""
            total_marks = ""
            
            # Get max marks from course details (if available)
            formative_max = ""
            summative_max = ""
            if "formative" in course_rows.columns:
                formative_max = safe_number(course_rows.iloc[0].get("formative", ""))
            if "summative" in course_rows.columns:
                summative_max = safe_number(course_rows.iloc[0].get("summative", ""))
            
            for _, row in course_rows.iterrows():
                assessment_type = row.get("Assessment Type", "")
                marks = safe_number(row.get("Marks", 0))
                
                if assessment_type == "Formative":
                    formative_marks_obtained = marks
                elif assessment_type == "Summative":
                    summative_marks_obtained = marks
                elif assessment_type == "Total Marks":
                    total_marks = marks

            if not course_name and not course_code:
                continue

            # Calculate total marks obtained (formative + summative)
            total_marks_obtained = safe_number(formative_marks_obtained) + safe_number(summative_marks_obtained)
            
            data.append([
                safe_text(course_code),
                safe_paragraph(course_name, subject_style),
                safe_number(formative_max),
                safe_number(summative_max),
                safe_number(formative_marks_obtained),
                safe_number(summative_marks_obtained),
                Paragraph(str(total_marks_obtained), center_style)  # Wrap Total Marks in Paragraph with center alignment
            ])

        if len(data) == 2:  # Only headers, no data
            continue

        total_formative_obtained = safe_number(year_data[year_data["Assessment Type"] == "Formative"]["Marks"].sum())
        total_summative_obtained = safe_number(year_data[year_data["Assessment Type"] == "Summative"]["Marks"].sum())
        total_marks_obtained_year = total_formative_obtained + total_summative_obtained
        
        # Calculate totals for max marks: need to sum unique course values (one per course)
        total_formative_max = 0
        total_summative_max = 0
        if "formative" in year_data.columns and "Course Code" in year_data.columns:
            # Group by course code and take first row (all rows for same course have same max marks)
            unique_courses = year_data.groupby("Course Code").first()
            total_formative_max = safe_number(unique_courses["formative"].sum())
        if "summative" in year_data.columns and "Course Code" in year_data.columns:
            unique_courses = year_data.groupby("Course Code").first()
            total_summative_max = safe_number(unique_courses["summative"].sum())

        data.append([
            "",  # Empty first cell
            Paragraph("Total", bold_subject_style),  # Second cell with just "Total"
            safe_number(total_formative_max),
            safe_number(total_summative_max),
            safe_number(total_formative_obtained),
            safe_number(total_summative_obtained),
            Paragraph(str(total_marks_obtained_year), bold_center_style)  # Wrap Total Marks in Paragraph with center alignment
        ])

        table = Table(data, colWidths=[50, 220, 50, 50, 50, 50, 38], repeatRows=2)  # Reduced Sub Code and Total Marks
        
        # Enhanced table style with merged cells for header - reduced font and padding
        enhanced_table_style = TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,-1), "Times-Roman"),
            ("FONTSIZE", (0,0), (-1,-1), 8),  # Reduced from 9
            ("FONTNAME", (0,0), (-1,0), "Times-Bold"),  # Bold first header row
            ("FONTNAME", (0,1), (-1,1), "Times-Bold"),  # Bold second header row
            ("FONTNAME", (0,-1), (-1,-1), "Times-Bold"),  # Bold total row
            ("BACKGROUND", (0,0), (-1,1), colors.lightgrey),  # Background for both header rows
            ("ALIGN", (0,0), (1,1), "LEFT"),  # Left align Sub Code and Subject/Papers
            ("ALIGN", (2,0), (-1,1), "CENTER"),  # Center align header text
            ("ALIGN", (2,2), (5,-1), "CENTER"),  # Center align data columns 2-5
            ("ALIGN", (6,2), (6,-1), "CENTER"),  # Center align Total Marks column (column 6)
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),  # Vertical center alignment
            ("TOPPADDING", (0,0), (-1,-1), 2),  # Reduced padding for compact rows
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),  # Reduced padding for compact rows
            ("LEFTPADDING", (0,0), (-1,-1), 3),
            ("RIGHTPADDING", (0,0), (-1,-1), 3),
            # Merge cells vertically for first two columns
            ("SPAN", (0, 0), (0, 1)),  # Merge "Sub Code" across rows 0 and 1
            ("SPAN", (1, 0), (1, 1)),  # Merge "Subject/Papers" across rows 0 and 1
            # Merge cells horizontally in row 0
            ("SPAN", (2, 0), (3, 0)),  # Merge "Max Marks" cells (columns 2-3) in row 0
            ("SPAN", (4, 0), (5, 0)),  # Merge "Max Obtained" cells (columns 4-5) in row 0
            # Merge cells vertically for Total Marks
            ("SPAN", (6, 0), (6, 1)),  # Merge "Total Marks" across rows 0 and 1
        ])
        
        table.setStyle(enhanced_table_style)

        elements.append(Paragraph(f"<b>Part {year}</b>", left_year_style))
        elements.append(Spacer(1, 15))  # Reduced spacing
        elements.append(table)
        elements.append(Spacer(1, 15))  # Reduced spacing

    # Summary - Calculate across all years
    # Total Marks Obtained = Sum of formative + summative obtained across all years
    total_formative_obtained_all = safe_number(student_data[student_data["Assessment Type"] == "Formative"]["Marks"].sum())
    total_summative_obtained_all = safe_number(student_data[student_data["Assessment Type"] == "Summative"]["Marks"].sum())
    total_marks_obtained_all = total_formative_obtained_all + total_summative_obtained_all
    
    # Total Marks (Max) = Sum of formative + summative max marks across all years (unique courses)
    total_formative_max_all = 0
    total_summative_max_all = 0
    if "formative" in student_data.columns and "Course Code" in student_data.columns:
        unique_courses_all = student_data.groupby("Course Code").first()
        total_formative_max_all = safe_number(unique_courses_all["formative"].sum())
    if "summative" in student_data.columns and "Course Code" in student_data.columns:
        unique_courses_all = student_data.groupby("Course Code").first()
        total_summative_max_all = safe_number(unique_courses_all["summative"].sum())
    
    total_marks_max_all = total_formative_max_all + total_summative_max_all
    
    # Calculate percentage: (obtained / max) * 100
    percentage = 0.0
    if total_marks_max_all > 0:
        percentage = round((total_marks_obtained_all / total_marks_max_all) * 100, 2)
    
    summary_data = [
        ["RESULT", ": Cleared", "", "Total Marks", f": {total_marks_max_all}"],
        ["Percentage", f": {percentage}%", "", "Total Marks Obtained", f": {total_marks_obtained_all}"],
    ]

    colWidths = [0.20 * (width - doc.leftMargin - doc.rightMargin),
                 0.15 * (width - doc.leftMargin - doc.rightMargin),
                 0.20 * (width - doc.leftMargin - doc.rightMargin),
                 0.25 * (width - doc.leftMargin - doc.rightMargin),
                 0.20 * (width - doc.leftMargin - doc.rightMargin)]

    summary_table = Table(summary_data, colWidths=colWidths, hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),  # Reduced from 10
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (4, 0), (4, -1), 'Helvetica-Bold'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),  # Reduced padding
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),  # Reduced padding
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 10))  # Reduced spacing

    # Build
    elements = [x for x in elements if x not in [None, "", []]]
    try:
        doc.build(elements)
        return filename
    except Exception as e:
        st.error(f"❌ PDF build failed: {e}")
        return None

def app():
    fix_streamlit_layout()
    set_compact_theme()
    
    st.header("📄 Marksheet Generation System")
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("📁 Upload your files to generate transcripts with marks (ABC format)")
        
        # Course details file upload (optional)
        course_file = st.file_uploader(
            "📚 Upload Course Details (Excel/CSV) - Optional",
            type=['xlsx', 'xls', 'csv'],
            help="Upload course details file with columns: 'year', 'course code', 'course name', 'formative', 'summative'"
        )
        
             
        # Data file upload
        data_file = st.file_uploader(
            "📊 Upload Student Data (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            help="Upload the student data file with marks information (Formative, Summative, Total Marks columns)"
        )
        
        # Images zip file upload (optional)
        st.markdown("---")
        images_zip = st.file_uploader(
            "📷 Upload Student Photos (ZIP file) - Optional",
            type=['zip'],
            help="Upload a ZIP file containing student photos (named by Student Code/enrollment number). Photos will be added later if not uploaded now."
        )
        
        # Report date input
        report_date = st.date_input(
            "📅 Report Date",
            value=datetime.now().date(),
            help="Select the report generation date"
        )
        
        # Generate button
        if st.button("🚀 Generate Transcripts (Marks)", type="primary"):
            if data_file is not None:
                try:
                    # Create temporary directories
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract images if provided
                        images_dir = os.path.join(temp_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        
                        if images_zip is not None:
                            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                                zip_ref.extractall(images_dir)
                            
                            # Copy images to current directory for processing
                            current_images_dir = os.path.join(os.getcwd(), "images")
                            if os.path.exists(current_images_dir):
                                shutil.rmtree(current_images_dir)
                            shutil.copytree(images_dir, current_images_dir)
                            
                            # Debug: Show what images are available
                            st.info(f"📁 Images extracted to: {images_dir}")
                            if os.path.exists(images_dir):
                                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                st.write(f"📷 Found {len(image_files)} image files: {image_files[:10]}{'...' if len(image_files) > 10 else ''}")
                        else:
                            st.info("ℹ️ No photos uploaded - transcripts will be generated without student photos (you can add photos later)")
                        
                        # Process course details file if provided
                        df_courses = None
                        if course_file is not None:
                            try:
                                if course_file.name.endswith('.csv'):
                                    df_courses = pd.read_csv(course_file, encoding='latin1')
                                else:
                                    df_courses = pd.read_excel(course_file)
                                
                                df_courses.columns = df_courses.columns.str.strip()
                                
                                # Clean course names if needed
                                if "course name" in df_courses.columns:
                                    df_courses["course name"] = df_courses["course name"].apply(clean_course_name)
                                
                                st.success(f"✅ Loaded {len(df_courses)} course details")
                            except Exception as e:
                                st.warning(f"⚠️ Could not load course details file: {str(e)}")
                        
                        #st.write(df_courses)
                        # Process data file
                        if data_file.name.endswith('.csv'):
                            df = pd.read_csv(data_file, encoding='latin1')
                        else:
                            df = pd.read_excel(data_file)
                        
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        
                        # Remove unnamed columns
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        
                        # Process course columns if course file is provided
                        #if df_courses is not None:
                        #    df = process_course(df, df_courses)
                        
                        # Process to long format
                        df_long = process_marks_long_format(df, df_courses)
                        
                        # Drop rows with no marks
                        df_long = df_long.dropna(subset=['Course Code', 'Assessment Type', 'Marks'], how='all')
                        
                        #st.write(df_long.head())
                        # Create output directory
                        output_dir = os.path.join(temp_dir, "transcripts")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Logo path - look for logo.png in logo_dir
                        logo_dir = os.path.join(os.getcwd(), "logo_dir")
                        os.makedirs(logo_dir, exist_ok=True)
                        logo_path = os.path.join(logo_dir, "logo.png")
                        if not os.path.exists(logo_path):
                            st.warning("⚠️ Logo file not found at logo_dir/logo.png")
                            st.info("💡 Please place your logo file as 'logo.png' in the 'logo_dir' folder")
                            logo_path = None
                        else:
                            st.success("✅ Logo found successfully")
                        
                        # Generate PDFs
                        generated_files = []
                        progress_bar = st.progress(0)
                        total_students = len(df_long.groupby("Student Code"))
                        
                        for i, (student_id, student_data) in enumerate(df_long.groupby("Student Code")):
     
                            student_data_year1 = student_data[student_data['year']==1]
                            try:
                                filename = generate_pdf_onepage(
                                    student_id, 
                                    student_data_year1, 
                                    report_date.strftime("%d-%m-%Y"), 
                                    output_dir, 
                                    logo_path,
                                    images_dir,
                                    suffix="_1"
                                )
                                if filename:
                                    generated_files.append(filename)
                            except Exception as e:
                                st.warning(f"Failed to generate transcript for student {student_id}: {str(e)}")
                                import traceback
                                st.write(traceback.format_exc())
                            
                            student_data_year2 = student_data[student_data['year']==2]
                            
                            try:
                                filename = generate_pdf_onepage(
                                    student_id, 
                                    student_data_year2, 
                                    report_date.strftime("%d-%m-%Y"), 
                                    output_dir, 
                                    logo_path,
                                    images_dir,
                                    suffix="_2"
                                )
                                if filename:
                                    generated_files.append(filename)
                            except Exception as e:
                                st.warning(f"Failed to generate transcript for student {student_id}: {str(e)}")
                                import traceback
                                st.write(traceback.format_exc())
                            
                            progress_bar.progress((i + 1) / total_students)
                        
                        # Create final zip file
                        if generated_files:
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for pdf_file in generated_files:
                                    zip_file.write(pdf_file, os.path.basename(pdf_file))
                            
                            zip_buffer.seek(0)
                            
                            # Download button
                            st.success(f"✅ Generated {len(generated_files)} transcripts successfully!")
                            st.download_button(
                                label="📥 Download Transcripts (Marks) (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"transcripts_marks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            
                            # Clean up images after download (if photos were uploaded)
                            if images_zip is not None:
                                current_images_dir = os.path.join(os.getcwd(), "images")
                                if os.path.exists(current_images_dir):
                                    for file in os.listdir(current_images_dir):
                                        try:
                                            os.remove(os.path.join(current_images_dir, file))
                                        except:
                                            pass
                                    st.info("🧹 Student photos cleaned up (logo preserved in logo_dir)")
                            
                        else:
                            st.error("❌ No transcripts were generated successfully.")
                            
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
            else:
                st.warning("⚠️ Please upload the data file. Photos are optional and can be added later.")
        else:
            st.info("📋 Please upload the data file to generate transcripts with marks (ABC format). Photos are optional.")

if __name__ == "__main__":
    app()