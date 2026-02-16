import streamlit as st
import sqlite3
import bcrypt
import sys
import importlib.util
from pathlib import Path
import traceback
import time
import os
from dotenv import load_dotenv
import uuid
import json
import base64
from pathlib import Path

# Load environment variables
load_dotenv()

# -------------------------------
# Dynamically import task8 from Milestone-2
try:
    task8_path = Path("Milestone-2/task8.py").resolve()
    spec = importlib.util.spec_from_file_location("task8", task8_path)
    task8 = importlib.util.module_from_spec(spec)
    sys.modules["task8"] = task8
    spec.loader.exec_module(task8)
    create_rag_chain = task8.create_rag_chain
except Exception as e:
    st.error(f"Error loading RAG chain module: {e}")
    st.stop()

# -------------------------------
# API CONFIGURATION
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME]):
    st.error("Missing configuration in .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENV"] = PINECONE_ENV
os.environ["PINECONE_INDEX_NAME"] = PINECONE_INDEX_NAME

# -------------------------------
# PROFILE PICS DIR
PROFILE_PICS_DIR = Path("profile_pics")
PROFILE_PICS_DIR.mkdir(exist_ok=True)

# -------------------------------
# DATABASE
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    password TEXT,
    profile_pic TEXT
)
''')

# Chat sessions table
c.execute('''
CREATE TABLE IF NOT EXISTS chat_sessions (
    chat_id TEXT,
    username TEXT,
    messages TEXT,
    PRIMARY KEY (chat_id, username)
)
''')
conn.commit()

# -------------------------------
# HELPER FUNCTIONS
def save_profile_picture(uploaded_file):
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        path = PROFILE_PICS_DIR / filename
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename
    return None

def add_user(username, name, email, password, profile_pic=None):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c.execute('INSERT INTO users (username, name, email, password, profile_pic) VALUES (?, ?, ?, ?, ?)',
                  (username, name, email, hashed_pw, profile_pic))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    result = c.fetchone()
    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
        return True
    return False

def get_user(username):
    c.execute('SELECT username, name, email, profile_pic FROM users WHERE username=?', (username,))
    return c.fetchone()

def update_user(username, name, email, profile_pic=None):
    try:
        if profile_pic:
            c.execute('UPDATE users SET name=?, profile_pic=? WHERE username=?', 
                      (name, profile_pic, username))
        else:
            c.execute('UPDATE users SET name=? WHERE username=?', (name, username))
        conn.commit()
        return True
    except:
        return False

# -------------------------------
# CHAT HISTORY FUNCTIONS
def save_chat_history(chat_id, username, messages):
    c.execute('''
        INSERT OR REPLACE INTO chat_sessions (chat_id, username, messages)
        VALUES (?, ?, ?)
    ''', (chat_id, username, json.dumps(messages)))
    conn.commit()

def load_user_chats(username):
    c.execute('SELECT chat_id, messages FROM chat_sessions WHERE username=?', (username,))
    rows = c.fetchall()
    chat_sessions = {}
    for chat_id, messages_json in rows:
        chat_sessions[chat_id] = json.loads(messages_json)
    if "default_chat" not in chat_sessions:
        chat_sessions["default_chat"] = [{"role": "assistant", "content": "How can I help you with your legal research today?"}]
    return chat_sessions

# -------------------------------
# SESSION STATE INIT
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.current_page = "Login"
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = "default_chat"

if st.session_state.current_chat_id not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.current_chat_id] = [
        {"role": "assistant", "content": "How can I help you with your legal research today?"}
    ]
st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_chat_id]

# -------------------------------
# CALLBACKS
def switch_to_signup():
    st.session_state.current_page = "Signup"
    st.rerun()

def switch_to_login():
    st.session_state.current_page = "Login"
    st.rerun()

def new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = [
        {"role": "assistant", "content": "Starting a new legal session. What's your question?"}
    ]
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = st.session_state.chat_sessions[new_chat_id]
    st.session_state.current_page = "Home"
    st.rerun()

def switch_chat(chat_id):
    if chat_id in st.session_state.chat_sessions:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_sessions[chat_id]
        st.session_state.current_page = "Home"
        st.rerun()

# -------------------------------
# PAGES
def signup_page():
    st.markdown(
    """<style>
    /* Gradient multicolor background */
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 25%, #fbc2eb 50%, #a1c4fd 75%, #c2e9fb 100%);
        min-height: 100vh;
        background-attachment: fixed;
    }

    /* Text input styling */
    .stTextInput>label {
        color: black !important; /* label text color */
        font-weight: bold;
    }
    .stTextInput>div>div>input, 
    .stTextInput>div>div>textarea, 
    .stTextInput>div>div>div>input {
        background-color: rgba(255,255,255,0.7) !important; /* lighter background */
        color: black !important; /* input text color */
        border-radius: 5px !important;
        padding: 8px;
    }

    /* File uploader styling */
    .stFileUploader>label {
        color: black !important; /* label text color */
        font-weight: bold;
    }
    .stFileUploader>div>div>input {
        background-color: rgba(255,255,255,0.7) !important; /* light background */
        color: black !important;
        border-radius: 5px !important;
        padding: 8px;
    }

    /* Style the Create Account button */
    div.stButton > button:first-child {
        background-color: #F08080;  
        color: white;
        font-weight: bold;
        height: 40px;
        width: 100%;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #FFB38E
        ; /* Darker green on hover */
    }
    </style>""", unsafe_allow_html=True
)

    st.markdown("<h1 style='text-align:center;color:#F08080;'>Create Account</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<div class='profile-card'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        uploaded_file = st.file_uploader("Profile Picture", type=["jpg","png"])
        if uploaded_file: st.image(uploaded_file, width=150)
        
        if st.button("Create Account"):
            if not username or not name or not email or not password:
                st.warning("Fill all required fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                pic = save_profile_picture(uploaded_file)
                if add_user(username, name, email, password, pic):
                    st.success("Account created successfully! Log in now.")
                    st.session_state.current_page = "Login"
                    st.rerun()
                else:
                    st.error("Username exists.")
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Text and Login button side by side
        col_text, col_button = st.columns([3,1])
        with col_text:
            st.markdown("<p style='text-align:right;color:#111111; margin-top:8px;'>Already have an account?</p>", unsafe_allow_html=True)
        with col_button:
            st.button("Login", key="login_link", on_click=switch_to_login)



def login_page():
    st.markdown(
        """
        <style>
        /* Gradient multicolor background */
        .stApp {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 25%, #fbc2eb 50%, #a1c4fd 75%, #c2e9fb 100%);
            min-height: 100vh;
            background-attachment: fixed;
        }


        /* Center content vertically */
        .css-1d391kg {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Text input styling */
        .stTextInput>label {
            color: black !important; /* label text color */
            font-weight: bold;
        }
        .stTextInput>div>div>input, 
        .stTextInput>div>div>textarea, 
        .stTextInput>div>div>div>input {
            background-color: rgba(255,255,255,0.7) !important; /* lighter background */
            color: black !important; /* input text color */
            border-radius: 5px !important;
            padding: 8px;
        }

        /* Button styling (green) */
        div.stButton > button:first-child {
            background-color: #F08080;
            color: white;
            font-weight: bold;
            height: 40px;
            width: 100%;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #FFB38E;
        }

        /* Remove extra margin below heading */
        h1 {
            margin-bottom: 0px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align:center;color:#F08080;'>Login</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='profile-card'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username,password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.current_page = "Home"
                st.session_state.chat_sessions = load_user_chats(username)
                st.session_state.current_chat_id = "default_chat"
                st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_chat_id]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username/password")
        st.markdown("</div>", unsafe_allow_html=True)

        # Text and Sign up button side by side
        col_text, col_button = st.columns([3,1])
        with col_text:
            st.markdown("<p style='text-align:right;color:#111111; margin-top:8px;'>Create account here</p>", unsafe_allow_html=True)
        with col_button:
            st.button("Sign up", key="signup_link", on_click=switch_to_signup)




# -------------------------------
# PROFILE PAGE
def profile_page():
    set_profile_background("image copy.png")

    # NEW FIX: Inject CSS for button hover color
    st.markdown("""
    <style>
    /* Target all st.button elements */
    div.stButton > button:hover {
        background-color: #F08080 !important; /* Light Coral background on hover */
        border-color: #F08080 !important;
        color: white !important; /* Ensure text is readable on hover */
    }
    </style>
    """, unsafe_allow_html=True)
    # -------------------------------

    user = get_user(st.session_state.username)
    if not user: 
        st.error("User not found.")
        return

    # FIX 1: Change header color to BLACK
    st.markdown(f"<h1 style='text-align:center;color:#000000;'>Profile - {user[1]}</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        # Profile picture
        if user[3]:
            path = PROFILE_PICS_DIR / user[3]
            if path.exists():
                st.image(str(path), width=150, caption=user[1])
        else:
            st.image(f"https://api.dicebear.com/7.x/initials/svg?seed={user[1]}", width=150)

        # Display info
        # FIX: Use st.markdown with explicit inline styling for profile details
        st.markdown(f"<p style='color:#000000;'><b>Username:</b> {user[0]}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#000000;'><b>Name:</b> {user[1]}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#000000;'><b>Email:</b> {user[2]}</p>", unsafe_allow_html=True) 

        # Toggle edit profile
        if "show_edit" not in st.session_state:
            st.session_state.show_edit = False
        
        # This button's hover state is controlled by the CSS injected above
        if st.button("✏️ Edit Profile"):
            st.session_state.show_edit = not st.session_state.show_edit
            st.rerun()

        if st.session_state.show_edit:
            # FIX: Change "Edit Your Profile" subheader color to black
            st.markdown("<h2 style='color:#000000'>Edit Your Profile</h2>", unsafe_allow_html=True) # or use <h3 style='color:#000000;'>
            
            # FIX: CSS to make labels black AND change Browse Files button color to #F08080
            st.markdown("""<style>div[data-testid*="stTextInput"] label, .stFileUploader label {color: #000000 !important;} .stFileUploader > div > button {background-color: #F08080; color: white !important;}</style>""", unsafe_allow_html=True)
            
            edit_name = st.text_input("Full Name", value=user[1])
            new_pic = st.file_uploader("Change Profile Picture", type=["jpg", "png"], key="edit_profile_pic")
            
            if new_pic:
                st.image(new_pic, width=150)

            # This button's hover state is controlled by the CSS injected above
            if st.button("Save Profile Changes"):
                pic_file = save_profile_picture(new_pic) if new_pic else None
                if update_user(st.session_state.username, edit_name, user[2], pic_file):
                    st.success("✅ Profile updated successfully!")
                    st.session_state.show_edit = False
                    st.rerun()
                else:
                    st.error("❌ Update failed.")

            # ---------------- Change Password Section ----------------
            if "show_password_change" not in st.session_state:
                st.session_state.show_password_change = False

            # This button's hover state is controlled by the CSS injected above
            if st.button("🔒 Change Password"):
                st.session_state.show_password_change = not st.session_state.show_password_change
                st.rerun()

            if st.session_state.show_password_change:
                # FIX: Change "Change Password" subheader color to black
                st.markdown("<h2 style='color:#000000'>Change Password</h2>", unsafe_allow_html=True) # or use <h3 style='color:#000000;'>
                
                # FIX: st.text_input label text is now handled by the updated CSS in set_profile_background
                current_pw = st.text_input("Current Password", type="password", key="current_pw")
                new_pw = st.text_input("New Password", type="password", key="new_pw")
                confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")

                # This button's hover state is controlled by the CSS injected above
                if st.button("Update Password"):
                    if not current_pw or not new_pw or not confirm_pw:
                        st.warning("Please fill all password fields.")
                    elif new_pw != confirm_pw:
                        st.error("New passwords do not match.")
                    else:
                        c.execute("SELECT password FROM users WHERE username=?", (st.session_state.username,))
                        db_pw = c.fetchone()[0]
                        if bcrypt.checkpw(current_pw.encode(), db_pw.encode()):
                            hashed_new_pw = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
                            c.execute("UPDATE users SET password=? WHERE username=?", (hashed_new_pw, st.session_state.username))
                            conn.commit()
                            st.success("✅ Password changed successfully!")
                            st.session_state.show_password_change = False
                            st.rerun()
                        else:
                            st.error("❌ Current password is incorrect.")

                            
def set_profile_background(image_file):
    """Set a local image as background for the profile page with brighter text and styled buttons"""
    image_path = Path(__file__).parent / image_file
    if not image_path.exists():
        st.error(f"Background image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            opacity: 0.9;
            background-attachment: fixed;
        }}
        
        /* Main Content Area Styling */
        .stApp > .main {{
            border-radius: 10px;
            padding: 20px;
        }}

        /* FIX 2: Force ALL text, including labels and subheaders, to BLACK in the main content area */
        .stApp > .main h1, 
        .stApp > .main h2, 
        .stApp > .main h3, 
        .stApp > .main .stMarkdown p, 
        .stApp > .main div[data-testid*="stText"] p,
        .stApp > .main div[data-testid*="stText"] p b, 
        .stApp > .main p,
        .stApp > .main p b,
        /* NEW FIX: Target labels for text inputs and file uploaders */
        .stApp > .main label {{ 
            color: #000000 !important; /* Set text to black */
            text-shadow: none !important; 
            background-color: transparent !important;
        }}
        
        /* FIX 3: Apply #F08080 ONLY to Profile page main content buttons */
        .stApp > .main div.stButton > button:first-child {{
            color: white !important;
            font-weight: bold;
            background-color: #F08080 !important; /* Requested button color */
            border: 1px solid #F08080 !important;
            border-radius: 5px;
            height: 35px;
            width: 100%;
        }}
        .stApp > .main div.stButton > button:first-child:hover {{
            background-color: #FFB38E !important; /* Darker hover for #F08080 */
            border-color: #C07070 !important;
            color: white !important;
        }}
        
        /* ======================================= */
        /* Sidebar Styling: Ensure White Text and Default Buttons */
        
        /* Ensure Sidebar Text (Title, Chat History, Pages) remains WHITE */
        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] .stMarkdown {{
            color: white !important;
            text-shadow: none !important;
        }}
        
        /* Reset Sidebar button styling (to maintain the dark theme appearance) */
        section[data-testid="stSidebar"] div.stButton > button:first-child {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            font-weight: normal; 
        }}
        section[data-testid="stSidebar"] div.stButton > button:first-child:hover {{
            background-color: #FFB38E !important;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
    )
# -------------------------------
# HOME PAGE


def set_background_image(image_file):
    """Set a local image as Streamlit background with dark chat overlay"""
    image_path = Path(__file__).parent / image_file
    if not image_path.exists():
        st.error(f"Background image not found: {image_path}")
        return
    
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: 60% auto;
;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Dark semi-transparent overlay behind main content */
        .stApp > .main {{
            background-color: rgba(0, 0, 0, 0.6); /* light black */
            border-radius: 10px;
            padding: 20px;
        }}

        /* Chat text white */
        .stChatMessage p {{
            color: white !important;
        }}

        /* Optional: make chat bubbles slightly darker for contrast */
        .stChatMessage {{
            background-color: rgba(0, 0, 0, 0.7) !important;
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def home_page():
    # Set background image
    set_background_image("image copy.png")

    st.markdown("<h1 style='text-align:center;color:#F08080;'>LegalBot - AI Legal Assistant</h1>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)

    if prompt := st.chat_input("Ask a question about the constitution of India..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    @st.cache_resource(show_spinner=False)
                    def load_rag_chain_cached(): 
                        return create_rag_chain()
                    rag_chain = load_rag_chain_cached()
                    result = rag_chain.invoke({"input": prompt, "chat_history": chat_history, "context": ""})
                    result_text = result.get("output") or result.get("answer") or str(result) if isinstance(result, dict) else str(result)
                    st.markdown(result_text)
                    st.session_state.messages.append({"role": "assistant", "content": result_text})
                    save_chat_history(st.session_state.current_chat_id, st.session_state.username, st.session_state.messages)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.error(traceback.format_exc())




# -------------------------------
# APP NAVIGATION
if not st.session_state.logged_in:
    if st.session_state.current_page == "Login": login_page()
    elif st.session_state.current_page == "Signup": signup_page()
    else: login_page()
else:
    user_data = get_user(st.session_state.username)
    username, name, email, profile_pic_filename = user_data if user_data else (st.session_state.username, st.session_state.username, None, None)
    with st.sidebar:
        st.title("LegalBot ⚖")
        if profile_pic_filename:
            path = PROFILE_PICS_DIR / profile_pic_filename
            if path.exists(): st.image(str(path), use_container_width=True)
        else:
            st.markdown(f'<img src="https://api.dicebear.com/7.x/initials/svg?seed={name}" class="sidebar-avatar" />', unsafe_allow_html=True)
            
            # 🆕 Add welcome text just below profile picture
        st.markdown(
            f'<p style="text-align:center; color:white; font-size:16px; margin-top:10px;">Welcome, <b>{name}</b>!</p>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.button("➕ Start New Chat", use_container_width=True, on_click=new_chat)
        st.markdown("### Chat History")
        sorted_chat_ids = ["default_chat"] + [cid for cid in reversed(st.session_state.chat_sessions) if cid != "default_chat"]
        for chat_id in sorted_chat_ids:
            messages = st.session_state.chat_sessions[chat_id]
            first_msg = next((m['content'] for m in messages if m['role'] == "user"), "Untitled Chat")
            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
            st.button(title, key=f"chat_btn_{chat_id}", use_container_width=True, on_click=switch_chat, args=(chat_id,))
        st.markdown("---")
        page_options = ["Home", "Profile"]
        page = st.radio("Pages", page_options, index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0)
        st.session_state.current_page = page
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.current_chat_id = "default_chat"
            st.session_state.chat_sessions = {}
            st.session_state.current_page = "Login"
            st.success("Logged out successfully.")
            st.rerun()

    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Profile":
        profile_page()