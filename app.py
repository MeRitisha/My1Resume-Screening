import streamlit as st
import os
import tempfile
import pandas as pd
import asyncio
import sys
from backend.resume_parser import parse_resume
from backend.job_matcher import match_resume_to_job
from backend.huggingface_analyzer import analyze_resume_job_match
from backend.visualization import create_match_visualization, create_skills_chart

# Configure page settings
st.set_page_config(
    page_title="AI Resume Matcher", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Windows-specific event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary file and return the path"""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'resume_text': None,
        'job_description': None,
        'match_results': None,
        'extracted_data': None,
        'analysis_complete': False,
        'api_available': bool(os.environ.get("HUGGINGFACE_API_KEY"))
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_upload_section():
    """Display the file upload and job description sections"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Resume")
        uploaded_resume = st.file_uploader(
            "Choose a resume file (PDF or DOCX)", 
            type=["pdf", "docx"],
            key="resume_uploader"
        )
        if uploaded_resume is not None:
            with st.spinner("Parsing resume..."):
                temp_file_path = save_uploaded_file(uploaded_resume)
                if temp_file_path:
                    try:
                        resume_text, extracted_data = parse_resume(temp_file_path)
                        os.unlink(temp_file_path)  # Clean up temp file
                        if resume_text:
                            st.session_state.resume_text = resume_text
                            st.session_state.extracted_data = extracted_data
                            st.success("Resume parsed successfully!")
                            with st.expander("View Extracted Resume Text"):
                                st.text_area(
                                    "Resume Content", 
                                    resume_text, 
                                    height=250,
                                    label_visibility="collapsed"
                                )
                        else:
                            st.error("Failed to extract text from resume. Please try a different file.")
                    except Exception as e:
                        st.error(f"Error parsing resume: {str(e)}")
    
    with col2:
        st.subheader("Enter Job Description")
        job_description = st.text_area(
            "Paste the job description here", 
            height=250,
            key="job_desc_input"
        )
        if job_description:
            st.session_state.job_description = job_description

def perform_analysis():
    """Perform the resume-job matching analysis"""
    if st.session_state.resume_text and st.session_state.job_description:
        with st.spinner("Analyzing resume and job match..."):
            try:
                basic_match_results = match_resume_to_job(
                    st.session_state.resume_text, 
                    st.session_state.job_description,
                    st.session_state.extracted_data
                )
                
                if st.session_state.api_available:
                    try:
                        advanced_match_results = analyze_resume_job_match(
                            st.session_state.resume_text,
                            st.session_state.job_description,
                            basic_match_results
                        )
                        st.session_state.match_results = advanced_match_results
                        st.success("Advanced analysis complete!")
                    except Exception as api_error:
                        st.warning(f"Advanced analysis failed: {str(api_error)}. Using basic matching.")
                        st.session_state.match_results = basic_match_results
                else:
                    st.session_state.match_results = basic_match_results
                    st.info("Using basic matching. Set HUGGINGFACE_API_KEY for advanced AI analysis.")
                
                st.session_state.analysis_complete = True
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.analysis_complete = False
    else:
        st.warning("Please upload a resume and enter a job description.")

def display_results():
    """Display the analysis results"""
    if not st.session_state.analysis_complete:
        return
    
    st.header("Match Results")
    
    # Score and strengths/gaps columns
    score_col, chart_col = st.columns(2)
    
    with score_col:
        st.subheader("Overall Match Score")
        overall_score = st.session_state.match_results.get("overall_score", 0)
        
        # Determine color based on score
        if overall_score >= 70:
            color = "green"
            feedback = "Excellent Match!"
        elif overall_score >= 50:
            color = "orange"
            feedback = "Good Potential Match"
        else:
            color = "red"
            feedback = "Needs Improvement"
        
        st.markdown(f"""
            <div style='text-align: center;'>
                <h1 style='color: {color}; font-size: 72px;'>
                    {overall_score}%
                </h1>
                <p style='font-size: 24px;'>{feedback}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("Key Strengths")
        strengths = st.session_state.match_results.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.markdown(f"✅ **{strength}**")
        else:
            st.info("No specific strengths identified")
        
        st.subheader("Improvement Areas")
        gaps = st.session_state.match_results.get("gaps", [])
        if gaps:
            for gap in gaps:
                st.markdown(f"⚠️ **{gap}**")
        else:
            st.info("No specific gaps identified")
    
    with chart_col:
        st.subheader("Category Scores")
        category_scores = st.session_state.match_results.get("category_scores", {})
        if category_scores:
            try:
                match_fig = create_match_visualization(category_scores)
                if match_fig is not None:
                    st.plotly_chart(match_fig, use_container_width=True)
                else:
                    st.warning("Could not generate category scores visualization")
            except Exception as e:
                st.error(f"Error displaying category scores: {str(e)}")
        else:
            st.info("No category scores available")
        
        st.subheader("Skills Analysis")
        matched_skills = st.session_state.match_results.get("matched_skills", [])
        missing_skills = st.session_state.match_results.get("missing_skills", [])
        if matched_skills or missing_skills:
            try:
                skills_fig = create_skills_chart(matched_skills, missing_skills)
                if skills_fig is not None:
                    st.pyplot(skills_fig)
                else:
                    st.warning("Could not generate skills visualization")
            except Exception as e:
                st.error(f"Error displaying skills analysis: {str(e)}")
        else:
            st.info("No skills analysis available")
    
    # Recommendations section
    st.subheader("Recommendations")
    recommendations = st.session_state.match_results.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. **{rec}**")
    else:
        st.info("No specific recommendations available")
    
    # Download report button
    st.subheader("Export Results")
    if st.button("Generate Full Analysis Report"):
        try:
            df = pd.DataFrame([st.session_state.match_results])
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name="resume_job_match_analysis.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Could not generate report: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()
    
    st.title("📄 AI Resume Scanner & Job Matcher")
    st.markdown("""
    Upload your resume and job description to get an AI-powered analysis of your job match.
    The system will analyze key skills, experience, and qualifications to provide you with a compatibility score.
    """)
    
    display_upload_section()
    
    if st.session_state.resume_text and st.session_state.job_description:
        if st.button("Analyze Match", type="primary"):
            perform_analysis()
    
    if st.session_state.analysis_complete:
        display_results()

if __name__ == "__main__":
    main()