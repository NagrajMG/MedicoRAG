# Import everything from the utils module (likely includes text_split, hybrid_search, etc.)
from utils import *

# Tokenizer for basic English tokenization (used in tokenized_corpus)
tokenizer = ToktokTokenizer()

# Load environment variables (e.g., API keys)
load_dotenv()
os.getenv("GROQ_API_KEY")  # Loads the Groq API key from the .env file

# Initialize the LLM (Qwen-32B via Groq) with configuration
llm = ChatGroq(
    model="qwen-qwq-32b",  # Model name
    temperature=0.3,       # Lower temp = more deterministic
    max_tokens=None,       # No specific output limit
    timeout=None,          # No timeout cap
    max_retries=2,         # Retry logic for API calls
)

# Define the directory where .txt files are stored
root_dir = pathlib.Path(__file__).parent
folder_path = str(root_dir / 'data' / 'en')

# Initialization function to load, split, and tokenize documents
def init():
    documents = []   # Raw documents
    doc_names = []   # Corresponding file names

    start_time = time.time()  # Start timer
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())   # Load file content
                doc_names.append(filename)      # Store filename

    # Split documents into smaller chunks (for better retrieval granularity)
    chunks = text_split(documents)

    # Tokenize each chunk (lowercased) for later use in hybrid search
    tokenized_corpus = [tokenizer.tokenize(chunk.lower()) for chunk in chunks]

    end_time = time.time()  # End timer
    execution_time = end_time - start_time

    return chunks, tokenized_corpus  # Return processed data

# Run the init process and store the outputs globally
chunks, tokenized_corpus = init()

# Template to use when expecting *only the answer index* (A/B/C/D) with a strict constraint on no extra info
template_for_accuracy = """
query:{query}
COntext:{retrived_document_information}
You are a Medical expert MCQ answer teller. Given the above context, it is your job to tell The correct option.return only the index of correct annswer

Example-
C
D
A

**DONT ANSWER QUESTION IF THERE IS NO RELAVANT CONTEXT AND REPLY THAT YOU DONT HAVE RELAVANT DOCUMENT**
dont give excess information further to it
"""

# Template used for general question answering with contextual constraint
template = """
query:{query}
COntext:{retrived_document_information}
You are a Medical expert MCQ answer teller. Given the above context, it is your job to tell The correct answer.
**DONT ANSWER QUESTION IF THERE IS NO RELAVANT CONTEXT **
"""

# Create a LangChain PromptTemplate for quiz answering
quiz_generation_prompt = PromptTemplate(
    input_variables=["retrived_document_information", "query"],
    template=template
)

# LLMChain to wrap the prompt and model together for easy inference
chain = LLMChain(
    llm=llm,                          # The language model to use
    prompt=quiz_generation_prompt,   # Prompt format
    output_key="answer"              # Expected return key from model output
)

# Main RAG function to perform retrieval-augmented generation
def RAG(query, top_n=10, alpha=0.4):
    # Perform hybrid (semantic + token) search to retrieve top-n relevant chunks
    info = hybrid_search(query, top_n, chunks, tokenized_corpus, alpha)

    # Call the LLM chain with the query and retrieved document context
    response = chain({
        "retrived_document_information": info,
        "query": query
    })

    # Return the predicted answer from the model
    return response.get("answer")
