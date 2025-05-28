import streamlit as st
from core.palette import get_palette_choices, load_palette
from core.translations import translations
from streamlit_option_menu import option_menu

def t(key, lang):
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("fr", key))

def get_text_color(r, g, b):
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111" if luminance > 180 else "#fff"

def main():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"
    lang = st.session_state["lang"]
    
    selected = option_menu(
        None,
        ["Pixelizer", t("menu_gestion", lang)],
        icons=["palette2", "gear"],
        menu_icon="cast",
        default_index=1,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#23242a"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "color": "white",
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "border-radius": "6px",
            },
            "nav-link-selected": {"background-color": "#ffd600", "color": "#111"},
        },
    )

    if selected == "Pixelizer":
        st.switch_page("app.py")  # Ou le chemin relatif selon ton projet

    

    st.title(t("filter_palette", lang))
    st.info(t("info_palette_filter", lang))

    palette_choices = get_palette_choices()
    palette_choice = st.selectbox(t("choose_palette", lang), palette_choices)

    palette_df = load_palette(palette_choice, st)
    if palette_df is not None:
        if "ignored_colors" not in st.session_state or st.session_state.get("palette_filter_name", "") != palette_choice:
            st.session_state.ignored_colors = set()
            st.session_state.palette_filter_name = palette_choice

        # Actions de masse
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,2])
        with col_btn1:
            if st.button(t("all_on", lang)):
                st.session_state.ignored_colors = set()
        with col_btn2:
            if st.button(t("all_off", lang)):
                st.session_state.ignored_colors = set(palette_df['Code'].astype(str))
        with col_btn3:
            if st.button(t("reset_palette", lang)):
                st.session_state.ignored_colors = set()

        st.markdown("### " + t("active_colors", lang))

        cols_per_row = 10
        total_colors = len(palette_df)
        rows = (total_colors + cols_per_row - 1) // cols_per_row

        for row_idx in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row_idx * cols_per_row + col_idx
                if idx >= total_colors:
                    continue
                row = palette_df.iloc[idx]
                color_code = row['Code']
                try:
                    r, g, b = int(row['R']), int(row['G']), int(row['B'])
                except Exception:
                    r, g, b = 128, 128, 128
                checked = color_code not in st.session_state.ignored_colors

                with cols[col_idx]:
                    col_cb, col_sq = st.columns([1,2])
                    with col_cb:
                        cb = st.checkbox("", value=checked, key=f"palette_color_{idx}_filter")
                        if not cb:
                            st.session_state.ignored_colors.add(color_code)
                        else:
                            st.session_state.ignored_colors.discard(color_code)
                    with col_sq:
                        txt_color = get_text_color(r, g, b)
                        st.markdown(
                            f"""
                            <div style='
                                width: 38px; height: 38px; border-radius: 7px;
                                background: rgb({r},{g},{b});
                                border: 2.2px solid #444;
                                display: flex; align-items: center; justify-content: center;
                                box-shadow: 0 1px 5px #0003;
                                font-size: 15px; font-weight: 700;
                                color: {txt_color};
                                letter-spacing: 0.5px;
                                text-shadow: 1px 1px 2px #0008;
                                margin-top: 2px; margin-bottom: 6px;
                            '>
                                {color_code}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.success(t("choices_saved", lang))

        if st.button(t("back_pixelizer", lang)):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
