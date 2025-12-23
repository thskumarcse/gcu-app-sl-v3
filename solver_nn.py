import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def preprocess_weight(weight_str):
    """Preprocess weight string to handle various input formats and convert to proper list-of-list format"""
    try:
        # Remove all spaces and clean the string
        weight_str = weight_str.replace(' ', '')
        
        # Remove trailing punctuation (periods, commas, semicolons, etc.)
        while weight_str and weight_str[-1] in '.,;!?':
            weight_str = weight_str[:-1]
        
        # Case 1: Already in proper format [[a,b,c,d],[e,f,g,h],[i,j,k,l]]
        if weight_str.startswith('[[') and weight_str.endswith(']]'):
            return weight_str
        
        # Case 2: Missing outer brackets [a,b,c,d],[e,f,g,h],[i,j,k,l]
        elif weight_str.startswith('[') and weight_str.endswith(']') and '],[' in weight_str:
            return '[' + weight_str + ']'
        
        # Case 3: Missing all brackets a,b,c,d],[e,f,g,h],[i,j,k,l
        elif '],[' in weight_str and not weight_str.startswith('['):
            return '[[' + weight_str + ']]'
        
        # Case 4: Comma-separated values without brackets a,b,c,d,e,f,g,h,i,j,k,l
        elif ',' in weight_str and not '[' in weight_str:
            # Try to split into groups of 4
            values = [float(x.strip()) for x in weight_str.split(',')]
            if len(values) == 12:  # 3x4 matrix
                # Group into 3 rows of 4 values each
                rows = []
                for i in range(0, 12, 4):
                    rows.append(values[i:i+4])
                # Return without spaces to avoid parsing issues
                return '[[' + ','.join([str(x) for x in rows[0]]) + '],[' + ','.join([str(x) for x in rows[1]]) + '],[' + ','.join([str(x) for x in rows[2]]) + ']]'
            else:
                st.warning(f"⚠️ Found {len(values)} values, expected 12 for a 3×4 matrix")
                return None
        
        # Case 5: Space or other separator separated values
        elif ' ' in weight_str and not '[' in weight_str:
            # Try splitting by spaces
            values = [float(x.strip()) for x in weight_str.split()]
            if len(values) == 12:
                rows = []
                for i in range(0, 12, 4):
                    rows.append(values[i:i+4])
                return str(rows)
            else:
                st.warning(f"⚠️ Found {len(values)} values, expected 12 for a 3×4 matrix")
                return None
        
        else:
            st.warning(f"⚠️ Unrecognized weight format: {weight_str}")
            st.info("Supported formats:")
            st.info("1. [[a,b,c,d],[e,f,g,h],[i,j,k,l]]")
            st.info("2. [a,b,c,d],[e,f,g,h],[i,j,k,l]")
            st.info("3. a,b,c,d],[e,f,g,h],[i,j,k,l")
            st.info("4. a,b,c,d,e,f,g,h,i,j,k,l (12 values)")
            st.info("5. Any of the above with trailing punctuation (.,;!?)")
            return None
            
    except Exception as e:
        st.error(f"Error preprocessing weight string: {e}")
        return None

def parse_weight_matrix(weight_str):
    """Parse weight matrix from string format like '[[0.2,0.5,0.7,0.5],[0.5,0.2,0.8,0.4],[0.7,0.4,0.7,0.4]]'"""
    try:
        # First preprocess the weight string
        processed_weight = preprocess_weight(weight_str)
        if processed_weight is None:
            return None
        
        # Now parse the preprocessed string
        weight_str = processed_weight.strip()
        
        # Handle different formats
        if weight_str.startswith('[') and weight_str.endswith(']'):
            # Remove outer brackets
            weight_str = weight_str[1:-1]
        
        # Split by '],[' to separate rows (spaces already removed)
        rows = weight_str.split('],[')
        
        # Clean each row
        cleaned_rows = []
        for row in rows:
            # Remove brackets and split by comma
            row = row.replace('[', '').replace(']', '')
            values = [float(x.strip()) for x in row.split(',')]
            cleaned_rows.append(values)
        
        weight_matrix = np.array(cleaned_rows)
        
        # Validate dimensions - should be 3x4 for W1
        if weight_matrix.shape != (3, 4):
            st.warning(f"⚠️ Weight matrix has shape {weight_matrix.shape}, expected (3, 4). Please check your input format.")
            st.info("Expected format: 3 rows × 4 columns (for 3 hidden neurons × 4 input features)")
            return None
            
        return weight_matrix
    except Exception as e:
        st.error(f"Error parsing weight matrix: {e}")
        st.info("Expected format: [[a,b,c,d],[e,f,g,h],[i,j,k,l]] where each row has 4 values")
        return None

def forward_pass(input_values, W1, W2):
    """Perform forward pass through the neural network"""
    try:
        # Convert input to numpy array
        X = np.array(input_values).reshape(-1, 1)
        
        # Validate dimensions
        if X.shape[0] != 4:
            st.error(f"Input should have 4 values, got {X.shape[0]}")
            return None
            
        if W1.shape != (3, 4):
            st.error(f"W1 should be (3, 4), got {W1.shape}")
            return None
            
        if W2.shape != (2, 3):
            st.error(f"W2 should be (2, 3), got {W2.shape}")
            return None
        
        # Input to hidden layer
        hidden_input = W1 @ X
        hidden_output = sigmoid(hidden_input)
        
        # Hidden to output layer
        output_input = W2 @ hidden_output
        output = sigmoid(output_input)
        
        return output.flatten()
    except Exception as e:
        st.error(f"Error in forward pass: {e}")
        st.info("Check that your weight matrices have the correct dimensions:")
        st.info("- W1 (Input to Hidden): 3 rows × 4 columns")
        st.info("- W2 (Hidden to Output): 2 rows × 3 columns")
        return None

def parse_student_result(result_str):
    """Parse student's result from string format like '[0.5, 0.8]' or '0.5,0.8'"""
    try:
        if result_str is None or pd.isna(result_str):
            return None
        
        # Remove all spaces and clean the string
        result_str = str(result_str).replace(' ', '')
        
        # Remove trailing punctuation
        while result_str and result_str[-1] in '.,;!?':
            result_str = result_str[:-1]
        
        # Handle different formats
        if result_str.startswith('[') and result_str.endswith(']'):
            result_str = result_str[1:-1]
        
        # Split by comma and convert to float
        values = [float(x.strip()) for x in result_str.split(',')]
        
        if len(values) == 2:  # Should have 2 output values
            return np.array(values)
        else:
            st.warning(f"⚠️ Student result should have 2 values, got {len(values)}: {result_str}")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Error parsing student result '{result_str}': {e}")
        return None

def calculate_accuracy(student_output, correct_output, tolerance=0.01):
    """Calculate accuracy based on how close the student's output is to correct output"""
    try:
        if student_output is None or correct_output is None:
            return 0.0
        
        # Calculate mean squared error
        mse = np.mean((student_output - correct_output) ** 2)
        
        # Convert to accuracy percentage (lower MSE = higher accuracy)
        accuracy = max(0, 100 - (mse * 1000))  # Scale factor to make it more readable
        return min(100, accuracy)
    except:
        return 0.0

def app():
    st.header("🧠 Neural Network Problem Solver")
    st.markdown("---")
    
    # Problem statement
    st.subheader("📋 Problem Statement")
    st.markdown("""
    **Consider a feedforward neural network with the following specifications:**
    
    1. **Input layer**: 4 neurons with input values `[0.5, 0.8, 0.2, 0.6]`
    2. **Hidden layer**: 3 neurons  
    3. **Output layer**: 2 neurons
    4. **Weights**: 
       - W₁ (Input to Hidden): `[[0.w11, 0.w12, 0.w13, 0.w14], 
       [0.w21, 0.w22, 0.w23, 0.w24], 
       [0.w31, 0.w32, 0.w33, 0.w34]]` (3×4 matrix)
       - W₂ (Hidden to Output): `[[0.4, 0.1, 0.3], [0.5, 0.2, 0.4]]` (2×3 matrix)
    5. **Activation function**: Sigmoid
    
    **Task**: Calculate the output at the output layer when the given input values are passed through the NN.
        
    Where, w’s will have the following values:

    - $w_{11}$  : Number of your siblings (including you)
    - $w_{12}$  : Number of characters of your favourite color
    - $w_{13}$  : Number of characters of your favourite dish
    - $w_{14}$  : Number of characters of your favourite game
    - $w_{21}$  : Number of characters of your hobby
    - $w_{22}$  : Number of books (hard copy + ebook) of AI you have
    - $w_{23}$  : number of characters of your first name
    - $w_{24}$  : number of characters of your last name
    - $w_{31}$  : number of characters of your father’s first name
    - $w_{32}$  : number of characters of your father’s last name
    - $w_{33}$  : number of characters of your mother’s first name
    - $w_{34}$  : number of characters of your mother’s last name
    """)
    
    # Calculate correct solution
    st.subheader("✅ Example Solution")
    st.markdown("""W₁ (Input to Hidden) = np.array([[0.4, 0.1, 0.3, 0.2], 
                [0.5, 0.2, 0.4, 0.1], 
                [0.3, 0.4, 0.2, 0.5]])""")
    
    # Given parameters
    input_values = [0.5, 0.8, 0.2, 0.6]
    W1_correct = np.array([[0.4, 0.1, 0.3, 0.2], [0.5, 0.2, 0.4, 0.1], [0.3, 0.4, 0.2, 0.5]])  # 3x4 matrix
    W2_correct = np.array([[0.4, 0.1, 0.3], [0.5, 0.2, 0.4]])  # 2x3 matrix
    
    # Calculate example output using the correct weight matrix (for demonstration)
    example_output = forward_pass(input_values, W1_correct, W2_correct)
    
    if example_output is not None:
        st.success(f"**Example Output**: [{example_output[0]:.6f}, {example_output[1]:.6f}]")
        
        # Show step-by-step calculation
        with st.expander("🔍 Step-by-step Calculation"):
            st.markdown("**Step 1: Input to Hidden Layer**")
            X = np.array(input_values).reshape(-1, 1)
            st.write(f"Input vector X = {X.flatten()}")
            st.write(f"Weight matrix W₁ = \n{W1_correct}")
            
            hidden_input = W1_correct @ X
            st.write(f"Hidden layer input = W₁ × X = {hidden_input.flatten()}")
            
            hidden_output = sigmoid(hidden_input)
            #st.write(f"Hidden layer output (sigmoid) = {hidden_output.flatten()}")
            st.write(f"Hidden layer output (sigmoid) = {np.round(hidden_output.flatten(), 3)}")
            
            st.markdown("**Step 2: Hidden to Output Layer**")
            st.write(f"Weight matrix W₂ = \n{W2_correct}")
            
            output_input = W2_correct @ hidden_output
            st.write(f"Output layer input = W₂ × Hidden output = {np.round(output_input.flatten(), 3)}")
            
            final_output = sigmoid(output_input)
            st.write(f"Final output (sigmoid) = {np.round(final_output.flatten(), 3)}")
    
    st.markdown("---")
    
    # Student submissions section
    st.subheader("📊 Student Submissions Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Student Data (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file with columns: Timestamp, Enrolment Number, Name, email ID, Weight"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display uploaded data
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head(10))
            
            """
            # Check required columns with flexible matching
            st.subheader("📋 Column Analysis")
            st.write("**Available columns in your file:**")
            for i, col in enumerate(df.columns, 1):
                st.write(f"{i}. `{col}`")
            """
            # Flexible column matching
            def find_column(df, possible_names):
                """Find column with flexible matching"""
                for col in df.columns:
                    col_lower = col.lower().strip()
                    for name in possible_names:
                        if name.lower().strip() in col_lower or col_lower in name.lower().strip():
                            return col
                return None
            
            # Map required fields to possible column names
            column_mapping = {
                'timestamp': find_column(df, ['timestamp', 'time', 'date']),
                'enrollment': find_column(df, ['enrolment', 'enrollment', 'roll', 'student id', 'university number']),
                'name': find_column(df, ['name', 'student name', 'full name']),
                'email': find_column(df, ['email', 'email id', 'e-mail', 'mail']),
                'weight': find_column(df, ['weight', 'weights', 'weight matrix', 'matrix']),
                'result': find_column(df, ['result', 'output', 'answer', 'solution'])
            }
            
            # Check if all required columns are found
            missing_columns = [key for key, value in column_mapping.items() if value is None]
            
            if missing_columns:
                st.error(f"❌ Could not find columns for: {missing_columns}")
                st.info("**Looking for columns containing:**")
                st.info("- **Timestamp**: 'timestamp', 'time', or 'date'")
                st.info("- **Enrollment**: 'enrolment', 'enrollment', 'roll', 'student id', or 'university number'")
                st.info("- **Name**: 'name', 'student name', or 'full name'")
                st.info("- **Email**: 'email', 'email id', or 'e-mail'")
                st.info("- **Weight**: 'weight', 'weights', 'weight matrix', or 'matrix'")
                st.info("- **Result**: 'result', 'output', 'answer', or 'solution'")
            else:
                st.success("✅ All required columns found!")
                #st.write("**Column mapping:**")
                #for key, value in column_mapping.items():
                #    st.write(f"- {key.title()}: `{value}`")
                # Process student submissions
                
                #st.subheader("🔍 Processing Student Submissions")
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    try:
                        # Parse weight matrix using mapped column
                        weight_str = str(row[column_mapping['weight']])
                        W1_student = parse_weight_matrix(weight_str)
                        
                        # Parse student's result
                        result_str = str(row[column_mapping['result']])
                        student_result = parse_student_result(result_str)
                        
                        if W1_student is not None:
                            # Calculate what the neural network should output using student's weight matrix
                            calculated_output = forward_pass(input_values, W1_student, W2_correct)
                            
                            if calculated_output is not None:
                                # Compare with student's provided result
                                if student_result is not None:
                                    # Calculate accuracy between calculated output and student's result
                                    accuracy = calculate_accuracy(calculated_output, student_result)
                                    
                                    results.append({
                                        'Timestamp': row[column_mapping['timestamp']],
                                        'Enrolment_Number': row[column_mapping['enrollment']],
                                        'Name': row[column_mapping['name']],
                                        'Email': row[column_mapping['email']],
                                        'Weight_Matrix': weight_str,
                                        'Student_Result_1': student_result[0],
                                        'Student_Result_2': student_result[1],
                                        'Calculated_Output_1': calculated_output[0],
                                        'Calculated_Output_2': calculated_output[1],
                                        'Accuracy_%': accuracy,
                                        'Error_1': abs(student_result[0] - calculated_output[0]),
                                        'Error_2': abs(student_result[1] - calculated_output[1])
                                    })
                                else:
                                    # Student result parsing failed
                                    results.append({
                                        'Timestamp': row[column_mapping['timestamp']],
                                        'Enrolment_Number': row[column_mapping['enrollment']],
                                        'Name': row[column_mapping['name']],
                                        'Email': row[column_mapping['email']],
                                        'Weight_Matrix': weight_str,
                                        'Student_Result_1': 'Parse Error',
                                        'Student_Result_2': 'Parse Error',
                                        'Calculated_Output_1': calculated_output[0],
                                        'Calculated_Output_2': calculated_output[1],
                                        'Accuracy_%': 'N/A',
                                        'Error_1': 'N/A',
                                        'Error_2': 'N/A'
                                    })
                            else:
                                # Forward pass failed
                                results.append({
                                    'Timestamp': row[column_mapping['timestamp']],
                                    'Enrolment_Number': row[column_mapping['enrollment']],
                                    'Name': row[column_mapping['name']],
                                    'Email': row[column_mapping['email']],
                                    'Weight_Matrix': weight_str,
                                        'Student_Result_1': 'Error',
                                        'Student_Result_2': 'Error',
                                        'Calculated_Output_1': 'Error',
                                        'Calculated_Output_2': 'Error',
                                        'Accuracy_%': 'N/A',
                                        'Error_1': 'N/A',
                                        'Error_2': 'N/A'
                                })
                        else:
                            # Weight matrix parsing failed
                            results.append({
                                'Timestamp': row[column_mapping['timestamp']],
                                'Enrolment_Number': row[column_mapping['enrollment']],
                                'Name': row[column_mapping['name']],
                                'Email': row[column_mapping['email']],
                                'Weight_Matrix': weight_str,
                                        'Student_Result_1': 'Error',
                                        'Student_Result_2': 'Error',
                                        'Calculated_Output_1': 'Error',
                                        'Calculated_Output_2': 'Error',
                                        'Accuracy_%': 'N/A',
                                        'Error_1': 'N/A',
                                        'Error_2': 'N/A'
                            })
                    except Exception as e:
                        st.warning(f"Error processing row {idx + 1}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("📈 Analysis Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Submissions", len(results_df))
                
                with col2:
                    # Convert Accuracy_% to numeric for proper filtering
                    results_df['Accuracy_%_numeric'] = pd.to_numeric(results_df['Accuracy_%'], errors='coerce')
                    valid_submissions = len(results_df[results_df['Accuracy_%_numeric'] > 0])
                    st.metric("Valid Submissions", valid_submissions)
                
                with col3:
                    if valid_submissions > 0:
                        avg_accuracy = results_df[results_df['Accuracy_%_numeric'] > 0]['Accuracy_%_numeric'].mean()
                        st.metric("Average Accuracy", f"{avg_accuracy:.2f}%")
                    else:
                        st.metric("Average Accuracy", "N/A")
                
                with col4:
                    if valid_submissions > 0:
                        max_accuracy = results_df[results_df['Accuracy_%_numeric'] > 0]['Accuracy_%_numeric'].max()
                        st.metric("Best Accuracy", f"{max_accuracy:.2f}%")
                    else:
                        st.metric("Best Accuracy", "N/A")
                
                # Detailed results table
                st.subheader("📊 Detailed Results")
                
                # Format the display dataframe
                display_df = results_df.copy()
                display_df['Student_Result'] = display_df.apply(
                    lambda x: f"[{x['Student_Result_1']:.6f}, {x['Student_Result_2']:.6f}]" 
                    if isinstance(x['Student_Result_1'], (int, float)) 
                    else f"[{x['Student_Result_1']}, {x['Student_Result_2']}]", axis=1
                )
                display_df['Calculated_Output'] = display_df.apply(
                    lambda x: f"[{x['Calculated_Output_1']:.6f}, {x['Calculated_Output_2']:.6f}]" 
                    if isinstance(x['Calculated_Output_1'], (int, float)) 
                    else f"[{x['Calculated_Output_1']}, {x['Calculated_Output_2']}]", axis=1
                )
                # Select columns for display and sort by accuracy (highest to lowest)
                display_columns = ['Name', 'Enrolment_Number', 'Student_Result', 'Calculated_Output', 'Accuracy_%']
                
                # Sort by accuracy (highest to lowest), handling non-numeric values
                display_df_sorted = display_df.copy()
                # Convert Accuracy_% to numeric, replacing non-numeric values with 0 for sorting
                display_df_sorted['Accuracy_%_numeric'] = pd.to_numeric(display_df_sorted['Accuracy_%'], errors='coerce').fillna(0)
                display_df_sorted = display_df_sorted.sort_values('Accuracy_%_numeric', ascending=False)
                
                st.dataframe(display_df_sorted[display_columns], use_container_width=True)
                
                # Accuracy distribution
                st.subheader("📊 Accuracy Distribution")
                
                if valid_submissions > 0:
                    # Create accuracy bins
                    accuracy_bins = [0, 20, 40, 60, 80, 90, 95, 100]
                    accuracy_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-90%', '90-95%', '95-100%']
                    
                    valid_results = results_df[results_df['Accuracy_%_numeric'] > 0]
                    accuracy_dist = pd.cut(valid_results['Accuracy_%_numeric'], bins=accuracy_bins, labels=accuracy_labels, include_lowest=True)
                    dist_counts = accuracy_dist.value_counts().sort_index()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.bar_chart(dist_counts)
                    
                    with col2:
                        st.write("**Distribution:**")
                        for label, count in dist_counts.items():
                            st.write(f"{label}: {count} students")
                
                # Download section
                st.subheader("💾 Download Results")
                
                # Prepare download data
                download_df = results_df.copy()
                download_df = download_df.round(6)  # Round to 6 decimal places
                
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    download_df.to_excel(writer, sheet_name='Neural_Network_Results', index=False)
                    
                    # Create summary sheet
                    summary_data = {
                        'Metric': ['Total Submissions', 'Valid Submissions', 'Average Accuracy (%)', 'Best Accuracy (%)', 'Worst Accuracy (%)'],
                        'Value': [
                            len(results_df),
                            valid_submissions,
                            f"{results_df[results_df['Accuracy_%_numeric'] > 0]['Accuracy_%_numeric'].mean():.2f}" if valid_submissions > 0 else "N/A",
                            f"{results_df[results_df['Accuracy_%_numeric'] > 0]['Accuracy_%_numeric'].max():.2f}" if valid_submissions > 0 else "N/A",
                            f"{results_df[results_df['Accuracy_%_numeric'] > 0]['Accuracy_%_numeric'].min():.2f}" if valid_submissions > 0 else "N/A"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Results (Excel)",
                        data=output.getvalue(),
                        file_name=f"neural_network_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download
                    csv_data = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"neural_network_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Top performers
                if valid_submissions > 0:
                    st.subheader("🏆 Top Performers")
                    top_performers = results_df[results_df['Accuracy_%_numeric'] > 0].nlargest(5, 'Accuracy_%_numeric')
                    st.dataframe(top_performers[['Name', 'Enrolment_Number', 'Accuracy_%']], use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please check the file format and try again.")
    
    else:
        st.info("👆 Please upload a CSV or Excel file to analyze student submissions.")
        
        # Show sample format
        st.subheader("📝 Expected File Format")
        sample_data = {
            'Timestamp': ['2024-01-15 10:30:00', '2024-01-15 10:35:00'],
            'Enrolment Number (Write the complete University number)': ['2024001', '2024002'],
            'Name': ['John Doe', 'Jane Smith'],
            'email ID': ['john@university.edu', 'jane@university.edu'],
            'Weight': ['[[0.4,0.1,0.3,0.2],[0.5,0.2,0.4,0.1],[0.3,0.4,0.2,0.5]]', '[0.3,0.2,0.4,0.1],[0.4,0.3,0.2,0.5],[0.2,0.5,0.3,0.4]'],
            'Result': ['[0.8, 0.6]', '[0.7, 0.5]']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        """
        # Test the preprocessing function
        st.subheader("🧪 Test Weight Preprocessing")
        test_input = st.text_input(
            "Test your weight format here:",
            value="[[0.2,0.4,0.4,0.6] , [0.5,0.3,0.5,0.6] , [0.7,0.6,0.7,0.6]]",
            help="Enter a weight matrix in any supported format to see how it gets processed"
        )
        
        if test_input:
            processed = preprocess_weight(test_input)
            if processed:
                st.success(f"✅ Processed successfully!")
                st.write(f"**Original**: `{test_input}`")
                st.write(f"**Processed**: `{processed}`")
                
                # Try to parse it
                parsed = parse_weight_matrix(test_input)
                if parsed is not None:
                    st.success(f"✅ Successfully parsed into {parsed.shape} matrix!")
                    st.write("**Parsed matrix:**")
                    st.write(parsed)
                else:
                    st.error("❌ Failed to parse the processed string")
            else:
                st.error("❌ Failed to preprocess the input")
        """

if __name__ == "__main__":
    app()
