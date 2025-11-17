import streamlit as st
import requests
import json
import time

# --- Configuration ---
# You MUST set your API Key in the Streamlit Secrets (st.secrets) or environment variables.
# For local testing, you can use st.secrets.GEMINI_API_KEY if you set up a secrets.toml file,
# or uncomment the line below and replace 'YOUR_API_KEY' with your actual key.
# IMPORTANT: For deployment, use Streamlit Secrets or environment variables for security.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback for local testing without secrets.toml (REMOVE FOR PRODUCTION)
    API_KEY = "" # Replace with your key for quick local test if needed

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- JSON Schema for Structured Output ---
# This ensures the AI returns standardized, machine-readable data.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "primaryDiagnosis": {"type": "STRING", "description": "The most likely primary diagnosis based on the symptoms and history."},
        "confidenceScore": {"type": "INTEGER", "description": "Confidence score (1-5, 5 being highest) for the primary diagnosis."},
        "soapNote": {
            "type": "OBJECT",
            "properties": {
                "subjective": {"type": "STRING", "description": "Patient's chief complaint, HPI, and relevant history."},
                "objective": {"type": "STRING", "description": "Objective findings from the doctor's observation/examination."},
                "assessment": {"type": "STRING", "description": "The final clinical assessment/impression."},
                "plan": {"type": "STRING", "description": "Proposed next steps, tests, and management."}
            }
        },
        "differentialDiagnosis": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "A list of 3 most likely alternative diagnoses (DDx), ranked by likelihood."
        }
    }
}

# --- Gemini API Call Function with Backoff ---
def call_gemini_api(conversation):
    """Calls the Gemini API to process the conversation into a structured JSON."""
    
    if not API_KEY:
        st.error("API Key is missing. Please set GEMINI_API_KEY in your Streamlit secrets or environment.")
        return None

    system_prompt = (
        "You are a specialized AI clinical documentation assistant integrated into the Preventify EHR. "
        "Your task is to process a doctor-patient conversation, extract critical information, and format it into a "
        "standardized JSON structure conforming strictly to the provided schema. The output must be valid JSON only."
    )
    user_query = f"Process the following doctor-patient conversation: \"{conversation}\""
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA
        }
    }
    
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload),
                timeout=30 # Set a reasonable timeout
            )
            
            if response.status_code == 429 and i < max_retries - 1:
                # Exponential backoff for rate limiting
                delay = 2 ** i + (time.monotonic() * 0.1)
                st.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            
            response.raise_for_status() # Raise exception for bad status codes
            
            result = response.json()
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if json_text:
                return json.loads(json_text)
            else:
                st.error("AI response content was empty or malformed.")
                return None

        except requests.exceptions.RequestException as e:
            st.error(f"Network or API Error: {e}")
            return None
        except json.JSONDecodeError:
            st.error("The AI returned a non-JSON response. This is often due to a complex prompt or internal error.")
            st.code(response.text) # Show raw response for debugging
            return None
    
    st.error("Failed to get a response after multiple retries.")
    return None

# --- UI Rendering ---

st.set_page_config(
    page_title="Preventify AI Co-Pilot", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5em;
    color: #4361ee;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5em;
}
.stTextArea label, .stButton {
    color: #4361ee !important;
    font-weight: 600;
}
.stCode {
    background-color: #161b22;
    border-radius: 8px;
    padding: 15px;
}
.stAlert {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Preventify AI Co-Pilot Prototype</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#c9d1d9;">Transforming unstructured doctor-patient dialogue into standardized EHR data.</p>', unsafe_allow_html=True)
st.divider()

# Example Conversation to preload
default_conversation = """
Doctor: Good morning. What brings you in today?
Patient: Good morning, Doctor. I've been feeling unusually tired and my joints, especially my knees and wrists, have been aching for about two months now. It's much worse right when I wake up.
Doctor: I see. Can you describe the joint pain? Is it a sharp pain or a dull ache?
Patient: Definitely a dull, persistent ache. And when I wake up, my fingers feel stiff for almost an hour before they loosen up.
Doctor: Stiffness lasting an hour. Important. Any associated swelling or redness?
Patient: No visible swelling, just the internal ache and stiffness.
Doctor: Have you had any recent fevers, unexplained weight loss, or skin rashes?
Patient: No weight loss or fever. But I have been quite stressed lately, and I switched to a completely vegan diet three months ago.
Doctor: That's relevant. We'll need to check your nutrient levels. Any family history of autoimmune conditions like Rheumatoid Arthritis?
Patient: My mother has a history of general arthritis, but nothing severe.
"""

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Doctor-Patient Interaction (Input)")
    conversation_input = st.text_area(
        "Paste or Type Conversation Transcript:",
        default_conversation,
        height=400,
        placeholder="The AI Co-Pilot listens to the conversation in real-time..."
    )

    if st.button("Process with AI Co-Pilot", use_container_width=True):
        if not conversation_input.strip():
            st.error("Please enter a conversation to process.")
        else:
            with st.spinner('AI Co-Pilot is generating standardized notes and diagnoses...'):
                structured_data = call_gemini_api(conversation_input)
            
            # Store the result in session state for display in the second column
            st.session_state['structured_data'] = structured_data

with col2:
    st.subheader("2. AI Output (Standardized EHR Data)")
    
    if 'structured_data' in st.session_state and st.session_state['structured_data'] is not None:
        data = st.session_state['structured_data']

        # --- Display Primary Diagnosis and Confidence ---
        st.info(f"**Primary Diagnosis Suggestion:** **{data.get('primaryDiagnosis', 'N/A')}**")
        
        confidence = data.get('confidenceScore')
        stars = '⭐' * confidence
        st.markdown(f"**Confidence Score:** {stars} ({confidence}/5)")
        st.divider()
        
        # --- Display SOAP Note ---
        st.markdown("### SOAP Note (Structured for EHR)")
        soap = data.get('soapNote', {})
        
        st.markdown(f"**Subjective (S):** {soap.get('subjective', 'N/A')}")
        st.markdown(f"**Objective (O):** {soap.get('objective', 'N/A')}")
        st.markdown(f"**Assessment (A):** {soap.get('assessment', 'N/A')}")
        st.markdown(f"**Plan (P):** {soap.get('plan', 'N/A')}")
        st.divider()

        # --- Display Differential Diagnosis ---
        st.markdown("### Differential Diagnosis (DDx)")
        ddx_list = data.get('differentialDiagnosis', [])
        if ddx_list:
            st.markdown(
                "The AI Co-Pilot provides these alternative diagnoses to ensure all standardized protocols are considered:"
            )
            for i, ddx in enumerate(ddx_list, 1):
                st.write(f"**{i}.** {ddx}")
        else:
            st.write("No differential diagnoses suggested.")

    else:
        st.markdown("<p style='color:#8b949e; margin-top: 15px;'>Click 'Process with AI Co-Pilot' to see the standardized output here.</p>", unsafe_allow_html=True)

st.divider()
st.caption("Powered by Gemini 2.5 Flash and structured JSON schema generation.")
