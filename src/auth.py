import streamlit as st


def enforce_enterprise_auth() -> str:
    """Gates access if REQUIRE_AUTH is true in Streamlit secrets. Returns active user ID."""
    require_auth = st.secrets.get("REQUIRE_AUTH", False)
    if not require_auth:
        return "guest_enterprise_user"

    if not getattr(st, "user", None) or not st.user.is_logged_in:
        st.warning("🔒 Enterprise SSO Login Required to access Voice Studio.")
        if st.button("Sign in with Corporate SSO", type="primary"):
            st.login()
        st.stop()

    with st.sidebar:
        st.caption(f"Signed in as: **{st.user.email}**")
        if st.button("Log out"):
            st.logout()

    return st.user.email
