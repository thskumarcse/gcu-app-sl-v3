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

"""
this generates transcripts for programs like M.Pharm, M.Tech, etc. 
with two years/4 semesters of study. The result is of 2-page per student.
It is based on CGPA/SGPA calculation.
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

# Helper functions from notebook
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
    extensions = ["jpg", "jpeg", "png"]
    for ext in extensions:
        candidate = os.path.join(photo_dir, f"{enrollment_no}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def expand_student_rows(df):
    """Convert each row (student with multiple subjects) into multiple rows."""
    subject_nums = sorted({col[3] for col in df.columns if col.startswith("SUB") and col[3].isdigit()})
    all_rows = []

    for _, row in df.iterrows():
        for num in subject_nums:
            new_row = {
                "ORG_NAME": row["ORG_NAME"],
                "ORG_PIN": row["ORG_PIN"],
                "ACADEMIC_COURSE_ID": row["ACADEMIC_COURSE_ID"],
                "COURSE_NAME": row["COURSE_NAME"],
                "ADMISSION_YEAR": row["ADMISSION_YEAR"],
                "DEPARTMENT": row["DEPARTMENT"],
                "STREAM": row["STREAM"],
                "SESSION": row["SESSION"],
                "ABC_ACCOUNT_ID": row["ABC_ACCOUNT_ID"],
                "DOB": row["DOB"],
                "GENDER": row["GENDER"],
                "MRKS_REC_STATUS": row["MRKS_REC_STATUS"],
                "CNAME": row["CNAME"],
                "YEAR": row["YEAR"],
                "MONTH": row["MONTH"],
                "SEM": row["SEM"],
                "RROLL": row["RROLL"],
                "REGN_NO": row["REGN_NO"],
                "TOT_GRADE": row["TOT_GRADE"],
                "TOT_CREDIT": row["TOT_CREDIT"],
                "TOT_CREDIT_POINTS": row["TOT_CREDIT_POINTS"],
                "TOT_GRADE_POINTS": row["TOT_GRADE_POINTS"],
                "PERCENT": row["PERCENT"],
                "CGPA": row["CGPA"],
                "SGPA": row["SGPA"],
                "GRAND_TOT_CREDIT_POINTS": row["GRAND_TOT_CREDIT_POINTS"],
                "GRAND_TOT_CREDIT": row["GRAND_TOT_CREDIT"],
                "RESULT": row["RESULT"],
                "REMARKS": row["REMARKS"],
                "DOI": row["DOI"],
                "NCRF_LEVEL": row["NCRF_LEVEL"],
                "SUB_NAME": row[f"SUB{num}NM"],
                "SUB_CODE": row[f"SUB{num}"],
                "GRADE": row[f"SUB{num}_GRADE"],
                "GRADE_POINTS": row[f"SUB{num}_GRADE_POINTS"],
                "CREDIT": row[f"SUB{num}_CREDIT"],
                "CREDIT_POINTS": row[f"SUB{num}_CREDIT_POINTS"]
            }
            all_rows.append(new_row)

    return pd.DataFrame(all_rows)

def draw_header_with_photo(canvas, doc, student_data, logo_path, photo_dir):
    """Draw header with logo and student photo."""
    width, height = A4
    usable_width = width - doc.leftMargin - doc.rightMargin

    enrollment_no = safe_text(student_data.iloc[0].get("RROLL", ""))
    student_name = safe_text(student_data.iloc[0].get("CNAME", ""))
    apaar_id = safe_number_text(student_data.iloc[0].get("ABC_ACCOUNT_ID", ""))
    serial_no = safe_number_text(student_data.iloc[0].get("REGN_NO", ""))
    course_name = safe_text(student_data.iloc[0].get("COURSE_NAME", ""))
    level = safe_text(student_data.iloc[0].get("NCRF_LEVEL", ""))

    # Logo
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
            # Debug: Print when photo is not found
            print(f"⚠️ No photo found for {student_name} ({enrollment_no}) in {photo_dir}")
    except Exception as e:
        print(f"❌ Error drawing photo for {enrollment_no}: {e}")
        pass

    # Titles
    canvas.setFont("Times-Roman", 18)
    canvas.drawCentredString(width / 2, height - 60, "Girijananda Chowdhury University")
    canvas.setFont("Times-Roman", 14)
    canvas.drawCentredString(width / 2, height - 85, "TRANSCRIPT")
    canvas.setFont("Times-Roman", 12)
    canvas.drawCentredString(width / 2, height - 100, course_name)
    canvas.drawCentredString(width / 2, height - 115, "2023-2025")

    # Student data table
    std_data = [
        ["APAAR ID", ":", apaar_id, "Serial Number : " + serial_no],
        ["Enrollment No.", ":", enrollment_no, ""],
        ["Name", ":", student_name, ""],
        ["NCrF/NHEQF Level", ":", level, ""],
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

def draw_footer(canvas, doc, date_value):
    """Draw footer with date and signature."""
    width, height = A4
    canvas.setFont("Helvetica", 10)
    canvas.drawString(50, 60, f"Date : {date_value}")
    canvas.drawRightString(width - 55, 70, "Controller of Examination")
    canvas.drawRightString(width - 40, 60, "Girijananda Chowdhury University")

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
    subject_style.leading = 11
    subject_style.alignment = TA_LEFT

    if len(str(text)) > max_len:  
        subject_style.fontSize = 8
        subject_style.leading = 10
    elif len(str(text)) > max_len * 1.5:  
        subject_style.fontSize = 7
        subject_style.leading = 9
    else:
        subject_style.fontSize = 9
    return Paragraph(str(text), subject_style)

def generate_pdf(student_id, student_data, report_date, output_dir, logo_path, photo_dir):
    """Generate PDF transcript for a student."""
    width, height = A4
    filename = os.path.join(output_dir, f"Transcript_{student_data.iloc[0]['CNAME']}.pdf")
    doc = BaseDocTemplate(filename, pagesize=A4)
    doc.showFooter = True

    # Styles
    styles = getSampleStyleSheet()
    subject_style = styles["Normal"]
    subject_style.fontName = "Times-Roman"
    subject_style.fontSize = 9
    subject_style.leading = 11
    subject_style.alignment = TA_LEFT

    bold_subject_style = ParagraphStyle(
        name="SubjectBold",
        parent=subject_style,
        fontName="Times-Bold",
        fontSize=subject_style.fontSize,
        leading=subject_style.leading,
        alignment=subject_style.alignment
    )

    left_sem_style = ParagraphStyle(
        name="LeftSemesterHeading",
        parent=styles['Heading4'],
        alignment=TA_LEFT,
        leftIndent=-30,
        fontName="Helvetica"
    )

    # Frames
    frame_first = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 140, id='first_frame')
    frame_later = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height + 50, id='later_frame')

    # Page templates
    first_page_template = PageTemplate(
        id='FirstPage',
        frames=[frame_first],
        onPage=lambda c, d: draw_header_with_photo(c, d, student_data, logo_path, photo_dir)
    )
    
    middle_page_template = PageTemplate(
        id='MiddlePages',
        frames=[frame_later],
        onPageEnd=lambda c, d: draw_footer(c, d, report_date)
    )
    
    doc.addPageTemplates([first_page_template, middle_page_template])

    # Table style
    table_style = TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Times-Roman"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (2,1), (-1,-1), "CENTER"),
    ])

    elements = []

    # Group by SEM
    for i, (sem, sem_data) in enumerate(student_data.groupby("SEM")):
        if i == 1:
            elements.append(NextPageTemplate('MiddlePages'))

        # Build table rows
        data = [["Sub Code", "Subject/Papers", "Credit", "Grade", "Grade Points", "Credit Points"]]

        for _, row in sem_data.iterrows():
            if row[["SUB_CODE","SUB_NAME","CREDIT","GRADE","GRADE_POINTS","CREDIT_POINTS"]].isna().all():
                continue

            sub_code = str(row.get("SUB_CODE", "")).replace("/", "_")
            sub_name = format_subject_name(row.get("SUB_NAME", ""))

            data.append([
                escape(sub_code),
                sub_name,
                safe_number(row.get("CREDIT", 0)),
                escape(str(row.get("GRADE", ""))),
                safe_number(row.get("GRADE_POINTS", 0)),
                safe_number(row.get("CREDIT_POINTS", 0))
            ])

        if len(data) == 1:
            continue

        # SGPA row
        total_credits = safe_number(sem_data["CREDIT"].sum())
        total_credit_points = safe_number(sem_data["CREDIT_POINTS"].sum())
        sgpa = round(total_credit_points / total_credits, 2) if total_credits else 0

        total_row = [
            Paragraph("Total", bold_subject_style),
            Paragraph(f"SGPA : {sgpa}", bold_subject_style),
            safe_number(total_credits),
            "",
            "",
            safe_number(total_credit_points)
        ]
        data.append(total_row)

        colWidths = [65, 220, 45, 45, 70, 65]
        heading = Paragraph(f"<b>Semester {sem}</b>", left_sem_style)

        table = Table(data, colWidths=colWidths, repeatRows=1)
        table.setStyle(table_style)

        elements.append(KeepTogether([heading, Spacer(1, 6), table]))
        elements.append(Spacer(1, 12))

    # Summary section
    try:
        cgpa = round(float(student_data["CGPA"].iloc[-1]), 2)
    except Exception:
        cgpa = 0.0

    percentage = round(cgpa * 10, 2)
    total_credit_points = safe_number(student_data["CREDIT_POINTS"].sum())
    total_credit = safe_number(student_data["CREDIT"].sum())
    result = escape(str(student_data.iloc[0].get("RESULT", "")))

    summary_data = [
        ["RESULT", f": {result}", "", "Grand Total Credit Points", f": {total_credit_points}"],
        ["CGPA", f": {cgpa}", "", "Grand Total Credit", f": {total_credit}"],
        ["Percentage", f": {percentage}", "", "", ""]
    ]

    usable_width = width - doc.leftMargin - doc.rightMargin
    colWidths = [0.12 * usable_width, 0.12 * usable_width, 0.30 * usable_width,
                 0.28 * usable_width, 0.28 * usable_width]

    summary_table = Table(summary_data, colWidths=colWidths, hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (4, 0), (4, -1), 'Helvetica-Bold'),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Build
    elements = [x for x in elements if x not in [None, "", []]]
    try:
        doc.build(elements, canvasmaker=NumberedCanvas)
        return filename
    except Exception as e:
        st.error(f"❌ PDF build failed: {e}")
        return None

def generate_pdf_onepage(student_id, student_data, report_date, output_dir, logo_path, photo_dir):
    """Generate single-page PDF transcript."""
    width, height = A4
    filename = os.path.join(output_dir, f"Transcript_{student_data.iloc[0]['CNAME']}_onepage.pdf")
    doc = BaseDocTemplate(filename, pagesize=A4)

    # Styles
    styles = getSampleStyleSheet()
    subject_style = styles["Normal"]
    subject_style.fontName = "Times-Roman"
    subject_style.fontSize = 9
    subject_style.leading = 11
    subject_style.alignment = TA_LEFT

    bold_subject_style = ParagraphStyle(
        name="SubjectBold",
        parent=subject_style,
        fontName="Times-Bold",
        fontSize=subject_style.fontSize,
        leading=subject_style.leading,
        alignment=subject_style.alignment
    )

    left_sem_style = ParagraphStyle(
        name="LeftSemesterHeading",
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

    # Page template
    single_page_template = PageTemplate(
        id="OnePage",
        frames=[frame_main],
        onPage=lambda c, d: draw_header_with_photo(c, d, student_data, logo_path, photo_dir),
        onPageEnd=lambda c, d: draw_footer(c, d, report_date)
    )
    doc.addPageTemplates([single_page_template])

    # Elements
    elements = []

    for sem, sem_data in student_data.groupby("SEM"):
        data = [["Sub Code", "Subject/Papers", "Credit", "Grade", "Grade Points", "Credit Points"]]

        for _, row in sem_data.iterrows():
            if row[["SUB_CODE", "SUB_NAME", "CREDIT", "GRADE", "GRADE_POINTS", "CREDIT_POINTS"]].isna().all():
                continue

            data.append([
                row.get("SUB_CODE", ""),
                safe_paragraph(row.get("SUB_NAME", ""), subject_style),
                row.get("CREDIT", 0) or 0,
                row.get("GRADE", ""),
                row.get("GRADE_POINTS", 0) or 0,
                row.get("CREDIT_POINTS", 0) or 0
            ])

        if len(data) == 1:
            continue

        total_credits = sem_data["CREDIT"].sum()
        total_credit_points = sem_data["CREDIT_POINTS"].sum()
        sgpa = round(total_credit_points / total_credits, 2) if total_credits else 0

        data.append([
            Paragraph("Total", bold_subject_style),
            Paragraph(f"SGPA : {sgpa}", bold_subject_style),
            total_credits, "", "", total_credit_points
        ])

        table = Table(data, colWidths=[65, 220, 45, 45, 70, 65], repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,-1), "Times-Roman"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
            ("FONTNAME", (0,-1), (-1,-1), "Times-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (2,1), (-1,-1), "CENTER"),
        ]))

        elements.append(Paragraph(f"<b>Semester {sem}</b>", left_sem_style))
        elements.append(Spacer(1, 6))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Summary
    try:
        cgpa = round(student_data["CGPA"].iloc[-1], 2)
    except Exception:
        cgpa = 0
    percentage = round(cgpa * 10, 2)
    total_credit_points = int(student_data["CREDIT_POINTS"].sum())
    total_credit = int(student_data["CREDIT"].sum())
    result = student_data.iloc[0].get("RESULT", "")

    summary_data = [
        ["RESULT", f": {result}", "", "Grand Total Credit Points", f": {total_credit_points}"],
        ["CGPA", f": {cgpa}", "", "Grand Total Credit", f": {total_credit}"],
        ["Percentage", f": {percentage}", "", "", ""]
    ]

    colWidths = [0.12 * (width - doc.leftMargin - doc.rightMargin),
                 0.12 * (width - doc.leftMargin - doc.rightMargin),
                 0.30 * (width - doc.leftMargin - doc.rightMargin),
                 0.28 * (width - doc.leftMargin - doc.rightMargin),
                 0.28 * (width - doc.leftMargin - doc.rightMargin)]

    summary_table = Table(summary_data, colWidths=colWidths, hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (4, 0), (4, -1), 'Helvetica-Bold'),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Build
    elements = [x for x in elements if x not in [None, "", []]]
    try:
        doc.build(elements)
        return filename
    except Exception as e:
        st.error(f"❌ PDF build failed: {e}")
        return None

def generate_pdf_auto(student_id, student_data, report_date, output_dir, logo_path, photo_dir):
    """Automatically choose one-page or multipage PDF generation."""
    total_subjects = student_data[["SUB_CODE", "SUB_NAME"]].notna().sum().max()
    
    if total_subjects <= 15:
        return generate_pdf_onepage(student_id, student_data, report_date, output_dir, logo_path, photo_dir)
    else:
        return generate_pdf(student_id, student_data, report_date, output_dir, logo_path, photo_dir)

def app():
    fix_streamlit_layout()
    set_compact_theme()
    
    st.header("📄 Transcript Generation System")
    
    # File upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("📁 Upload your files to generate transcripts")
        
        # Data file upload
        data_file = st.file_uploader(
            "📊 Upload Student Data (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            help="Upload the student data file with course information"
        )
        
        # Images zip file upload
        st.markdown("---")
        images_zip = st.file_uploader(
            "📷 Upload Student Photos (ZIP file)",
            type=['zip'],
            help="Upload a ZIP file containing student photos (named by enrollment number)"
        )
        
        # Report date input
        report_date = st.date_input(
            "📅 Report Date",
            value=datetime.now().date(),
            help="Select the report generation date"
        )
        
        # Generate button
        if st.button("🚀 Generate Transcripts", type="primary"):
            if data_file is not None and images_zip is not None:
                try:
                    # Create temporary directories
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract images
                        images_dir = os.path.join(temp_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        
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
                        
                        # Process data file
                        if data_file.name.endswith('.csv'):
                            df = pd.read_csv(data_file, skiprows=6, encoding='windows-1252')
                        else:
                            df = pd.read_excel(data_file)
                        
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        
                        # Add NCRF_LEVEL if missing
                        if "NCRF_LEVEL" not in df.columns:  
                            df["NCRF_LEVEL"] = ""
                        df["NCRF_LEVEL"] = df["NCRF_LEVEL"].fillna(" ")
                        
                        # Expand student rows
                        df_processed = expand_student_rows(df)
                        df_processed = df_processed.dropna(subset=['SUB_NAME', 'SUB_CODE'], how='all')
                        
                        # Create output directory
                        output_dir = os.path.join(temp_dir, "transcripts")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Logo path - look for logo.png in logo_dir
                        logo_dir = os.path.join(os.getcwd(), "logo_dir")
                        os.makedirs(logo_dir, exist_ok=True)  # Create logo_dir if it doesn't exist
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
                        total_students = len(df_processed.groupby("RROLL"))
                        
                        for i, (student_id, student_data) in enumerate(df_processed.groupby("RROLL")):
                            try:
                                filename = generate_pdf_auto(
                                    student_id, 
                                    student_data, 
                                    report_date.strftime("%d-%m-%Y"), 
                                    output_dir, 
                                    logo_path,
                                    images_dir
                                )
                                if filename:
                                    generated_files.append(filename)
                            except Exception as e:
                                st.warning(f"Failed to generate transcript for student {student_id}: {str(e)}")
                            
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
                                label="📥 Download Transcripts (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"transcripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            
                            # Clean up images after download
                            if os.path.exists(current_images_dir):
                                # Delete all student photos (logo is in separate logo_dir)
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
                st.warning("⚠️ Please upload both data file and images ZIP file.")
        else:
            st.info("📋 Please upload the required files to generate transcripts")

if __name__ == "__main__":
    app()