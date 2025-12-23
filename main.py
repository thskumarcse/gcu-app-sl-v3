import streamlit as st
from streamlit_option_menu import option_menu
from utility import detect_screen_width, get_authorized_pages_for_role
import login
import os

def _is_rerun_exc(ex):
    """Check if an exception is a Streamlit rerun exception that should be re-raised."""
    try:
        if getattr(ex, "is_fragment_scoped_rerun", False):
            return True
    except Exception:
        pass
    tname = type(ex).__name__
    if "Rerun" in tname or "rerun" in tname.lower():
        return True
    try:
        if "RerunData" in repr(ex) or "rerun" in repr(ex).lower():
            return True
    except Exception:
        pass
    try:
        mod = getattr(type(ex), "__module__", "")
        if mod and "streamlit" in mod:
            return True
    except Exception:
        pass
    return False

# Import modules with error handling
try:
    import hr_attendance
    import exam_transcript, exam_marksheet, exam_admitcard, exam_results, exam_results_all
    
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# --- Custom CSS for violet theme ---
custom_styles = {
    "container": {"padding": "5!important", "background-color": "#4a0072"},
    "icon": {"color": "white", "font-size": "15px"},
    "nav-link": {
        "color": "white",
        "font-size": "14px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#6a008c"
    },
    "nav-link-selected": {"background-color": "#2c6c9a", "font-weight": "normal"},
    "menu-title": {"color": "white", "font-size": "16px", "font-weight": "bold"}
}

# --- Configuration ---
# Production mode - always False for deployed version
DEV_MODE = os.getenv('DEV_MODE', 'False').lower() == 'true'

# App configuration
APP_CONFIG = {
    "title": "GCU Management System",
    "version": "1.0.0",
    "description": "Galgotias College University Management System"
}

# --- Page Rendering Logic ---
def render_page(page_name, role):
    """Render the correct page based on menu selection and user role."""
    page_map = {
        "Attendance": hr_attendance.app,
        "Transcript": exam_transcript.app,
        "Mark Sheet": exam_marksheet.app,
        "Admit Card": exam_admitcard.app,
        "Results": exam_results.app,
        "All Programs Results": exam_results_all.app,
        
    }

    authorized_pages = get_authorized_pages_for_role(role)
    if page_name not in authorized_pages:
        st.warning(f"⚠️ You do not have access to the '{page_name}' page.")
        return

    if page_name in page_map:
        try:
            page_map[page_name]()
        except Exception as e:
            # If this is a Streamlit rerun-control exception, re-raise so
            # Streamlit can handle the rerun instead of showing a raw error.
            if _is_rerun_exc(e):
                raise
            st.error(f"Error loading {page_name} page: {e}")
            st.info("Please try refreshing the page or contact support if the issue persists.")
    else:
        st.info("Please select an option from the sidebar.")


# --- Main App ---
def main():
    # Set page config
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    detect_screen_width()

    # Production mode indicator (only show in dev mode)
    if DEV_MODE:
        st.sidebar.markdown("""
        <div style='background-color: #ffeb3b; color: #000; padding: 5px; border-radius: 3px; text-align: center; margin-bottom: 10px;'>
            <strong>🔧 DEVELOPMENT MODE</strong>
        </div>
        """, unsafe_allow_html=True)

    # --- Authentication ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if DEV_MODE:
        # 🔹 Bypass login during development (unless explicitly logged out)
        if not st.session_state.get("dev_logged_out", False):
            st.session_state.authenticated = True
            st.session_state["role"] = "admin"
            st.session_state["current_user"] = {
                "Name": "Development User",
                "Designation": "System Administrator",
                "Department": "IT Department"
            }
        else:
            # Show login page even in dev mode if logged out
            st.session_state.authenticated = login.login()
            if st.session_state.authenticated:
                st.session_state.dev_logged_out = False  # Reset logout flag
                st.rerun()
            st.stop()
    else:
        # Production mode - always require authentication
        if not st.session_state.authenticated:
            st.session_state.authenticated = login.login()
            if st.session_state.authenticated:
                st.rerun()
            st.stop()

    # Get user role
    role = st.session_state.get("role", "guest")

    # --- Menus ---
    all_menus = {
        "HR Dept": {"icon": "people", "submenu": {"Attendance": "Attendance", "Feedback": "Feedback"}},
        "Examinations": {"icon": "book", "submenu": {
            "Transcript": "Transcript", "Mark Sheet": "Mark Sheet", "Admit Card": "Admit Card", "Results": "Results", "All Programs Results": "All Programs Results"}},
        "Mentoring": {"icon": "clipboard", "submenu": {
            "Mentor-Mentee": "Mentor-Mentee", "Data Input": "Data Input", "Reports": "Reports"}}
    }

    # Role-based menu filtering
    filtered_menus = {}
    if role == "admin":
        filtered_menus = all_menus
    elif role == "mentor_admin":
        # Mentor-admin has access to both Mentoring and Examinations
        filtered_menus = {
            "Mentoring": all_menus["Mentoring"],
            "Examinations": all_menus["Examinations"]
        }
    elif role == "hod":
        # HOD has access to Mentoring module
        filtered_menus = {"Mentoring": all_menus["Mentoring"]}
    elif role == "coordinator":
        # Coordinator has access to Mentoring module
        filtered_menus = {"Mentoring": all_menus["Mentoring"]}
    elif role == "mentor":
        # Mentor has access to Mentoring module
        filtered_menus = {"Mentoring": all_menus["Mentoring"]}
    elif role == "exam":
        filtered_menus = {"Examinations": all_menus["Examinations"]}
    elif role == "hr":
        filtered_menus = {"HR Dept": all_menus["HR Dept"]}
    else:
        st.error(f"No modules available for role '{role}'. Please contact admin.")
        st.stop()

    if not filtered_menus:
        st.error("No modules available for your role. Please contact admin.")
        st.stop()

    # Ensure valid active module & selected page
    if "active_module" not in st.session_state or st.session_state["active_module"] not in filtered_menus:
        st.session_state["active_module"] = list(filtered_menus.keys())[0]
        st.session_state["selected_page"] = list(
            filtered_menus[st.session_state["active_module"]]["submenu"].keys()
        )[0]

    if "selected_page" not in st.session_state or st.session_state["selected_page"] not in get_authorized_pages_for_role(role):
        st.session_state["selected_page"] = list(
            filtered_menus[st.session_state["active_module"]]["submenu"].keys()
        )[0]

    # Sidebar menu
    with st.sidebar:
        st.markdown("<h2 style='color: #8b00a3; text-align: center;'>GCU</h2>", unsafe_allow_html=True)
        
        # User info display
        if st.session_state.get("current_user"):
            user_info = st.session_state["current_user"]
            st.markdown(f"""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <p style='margin: 0; font-weight: bold; color: #4a0072;'>{user_info.get('Name', 'Unknown')}</p>
                <p style='margin: 0; font-size: 12px; color: #666;'>{user_info.get('Designation', '')}</p>
                <p style='margin: 0; font-size: 12px; color: #666;'>{user_info.get('Department', '')}</p>
            </div>
            """, unsafe_allow_html=True)

        options, icons = [], []
        for main_item, data in filtered_menus.items():
            options.append(main_item)
            icons.append(data["icon"])
            if main_item == st.session_state["active_module"]:
                for sub_item, _ in data["submenu"].items():
                    options.append("    " + sub_item)
                    icons.append(None)

        try:
            default_index = options.index("    " + st.session_state["selected_page"])
        except ValueError:
            default_index = options.index(st.session_state["active_module"])

        selected_option = option_menu(
            menu_title=None,
            options=options,
            icons=icons,
            default_index=default_index,
            styles=custom_styles
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Logout button - simplified
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                # Clear all session state and redirect to login
                st.session_state.clear()
                if DEV_MODE:
                    st.session_state.dev_logged_out = True  # Set flag for dev mode
                st.rerun()

    # Handle menu navigation
    processed_option = selected_option.strip()
    if processed_option in filtered_menus:
        if processed_option != st.session_state["active_module"]:
            st.session_state["active_module"] = processed_option
            first_sub_page = list(filtered_menus[processed_option]["submenu"].keys())[0]
            st.session_state["selected_page"] = first_sub_page
            st.rerun()
    elif processed_option != st.session_state["selected_page"]:
        st.session_state["selected_page"] = processed_option
        st.rerun()

    # Render selected page
    render_page(st.session_state["selected_page"], role)


# --- Run the app ---
if __name__ == "__main__":
    main()