import streamlit as st
import pypdf
from openai import OpenAI  # Groq uses the same library structure!

# Sidebar for the free Groq API Key
api_key = st.sidebar.text_input("Enter Groq API Key (gsk_...)", type="password")

st.title("🛡️ TruthLayer: Automated Fact-Checking Agent")
st.write("Upload your assessment PDF to extract claims, cross-reference data, and catch inaccuracies instantly.")

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def fact_check_text(document_text, groq_key):
    """
    Connects to Groq's free server using Meta's powerful Llama-3 model
    to perform strict fact-checking analysis.
    """
    # Point the client to Groq's official server address
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key
    )
    
    system_prompt = """
    You are an expert Fact-Checking Agent ("Truth Layer"). Your job is to analyze marketing text up to 2026, identify specific quantitative claims (stats, dates, figures), and verify them.
    
    For each major claim found, you must output a beautifully formatted Markdown Table with the exact columns:
    | Claim | Status | Source / Real Fact |
    
    Rules for the 'Status' column:
    - Must be EXACTLY one of these tags: **[Verified]**, **[Inaccurate]**, or **[False]**.
    
    Be extremely critical. Catch "trap" data (e.g., trillions of ChatGPT users or 85% G7 inflation are completely wrong). Provide the accurate real-world context in the final column.
    """
    
    user_prompt = f"Please extract and fact-check the claims in this text:\n\n{document_text}"
    
    # Using llama-3.1-8b-instant which is incredibly fast and free on Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content

# --- UI Layout ---

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please enter your Groq API Key in the sidebar to proceed.")
    else:
        with st.spinner("🔄 Extracting text from PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            
        st.success("📄 PDF uploaded successfully!")
            
        if st.button("🚀 Run Automated Fact-Checking"):
            with st.spinner("🤖 Analyzing claims against truth baseline..."):
                try:
                    report = fact_check_text(raw_text, api_key)
                    st.subheader("📊 Fact-Check Report")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please double-check your Groq API key.")