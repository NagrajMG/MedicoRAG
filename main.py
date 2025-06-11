# Imports
from utils import *
from engine import RAG, init  # Custom RAG pipeline and initializer

# Setup file paths 
root_dir = pathlib.Path(__file__).parent  # Get current directory
pdf_path = root_dir / 'data' / 'en' / 'extra.pdf'  # Path to store uploaded PDF
txt_path = root_dir / 'data' / 'en' / 'extra.txt'  # Path to store extracted text

# Streamlit page configuration 
st.set_page_config(page_title="RAG PDF QA", layout="wide")
st.title("📘 Upload PDF and Ask Questions (RAG)")  # App title

# Initialize session state variable to track whether app is running 
if "app_running" not in st.session_state:
    st.session_state.app_running = False

# Sidebar Controls for Start and Stop 
st.sidebar.subheader("⚙️ App Controls")
if st.sidebar.button("✅ Start App"):
    st.session_state.app_running = True  # Set app state to running
if st.sidebar.button("🛑 Stop App"):
    st.session_state.app_running = False  # Stop app state
    st.warning("App stopped manually.")   # Show warning in UI
    st.stop()  # Immediately halt further execution of the app

# Main Logic: Only runs if app is marked as running 
if st.session_state.app_running:

    # PDF Upload Section 
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")  # Upload control
    if uploaded_file:
        try:
            # Save the uploaded PDF file to disk
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Extract and clean text from PDF pages
            cleaned_content = ""
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text() or ''  # Get text or empty
                    # Remove non-ASCII characters for compatibility
                    cleaned_content += ''.join(c for c in text if ord(c) < 128)

            # Save cleaned text to a .txt file
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(cleaned_content)

            # Initialize document chunks and tokenization
            init()

            # Notify user of success
            st.success("PDF uploaded, cleaned, and processed.")

        except Exception as e:
            # Show error if any issue occurs during upload or processing
            st.error(f"Error processing PDF: {str(e)}")

    # Query Interface 
    st.markdown("### Ask a Question from the Uploaded PDF")  # Instruction
    query = st.text_input("Enter your query here:")  # Text input box

    # Button to trigger RAG-based answer generation
    if st.button("Get Answer"):
        if query.strip() == "":
            # Handle empty input
            st.warning("Please enter a valid question.")
        else:
            # Show spinner during processing
            with st.spinner("Processing with RAG..."):
                response = RAG(query)  # Call RAG pipeline with query

            # Display the model's response
            st.markdown("### 💬 RAG Response")
            st.write(response)
else:
    # Show this message if app hasn't been started
    st.info("Click '✅ Start App' in the sidebar to begin.")
