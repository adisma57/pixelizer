import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import math
from collections import Counter
from streamlit_option_menu import option_menu
from core.translations import translations
from core.palette import get_palette_choices, load_palette
from core.image_processing import (
    resize_and_contrast, convert_image_to_palette,
    diminish_colors, draw_block_grid, split_grid, upscale_for_pixel_perfect
)
from core.exporters import export_to_excel_multi, export_png_and_zip
from utils.streamlit_helpers import display_palette_preview

def t(key, lang):
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("fr", key))

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64

def display_palette_preview_only_selected(palette_df, ignored_colors):
    selected_df = palette_df[~palette_df['Code'].isin(ignored_colors)].reset_index(drop=True)
    html = """
    <div style="display: flex; flex-wrap: wrap; gap: 8px 6px; align-items: flex-start; margin-bottom: 6px;">
    """
    for _, row in selected_df.iterrows():
        r, g, b = int(row['R']), int(row['G']), int(row['B'])
        color_code = row['Code']
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        txt_color = "#111" if luminance > 180 else "#fff"
        html += f"""<div style='
            width: 27px; height: 27px; border-radius: 6px;
            background: rgb({r},{g},{b});
            border: 2px solid #fff4; box-shadow: 0 1px 7px #0005;
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; font-weight: 700;
            letter-spacing:0.3px; color:{txt_color}; text-shadow: 1px 1px 2px #0007;
            margin: 0;
        '>{color_code}</div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    return len(selected_df)

def main():
    st.set_page_config(page_title="Pixelizer", layout="wide", initial_sidebar_state="collapsed")

    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"
    lang = st.session_state["lang"]

    # Menu principal avec traduction
    selected = option_menu(
        None,
        ["Pixelizer", t("menu_gestion", lang)],
        icons=["palette2", "gear"],
        menu_icon="cast",
        default_index=0,
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

    if selected == t("menu_gestion", lang):
        st.switch_page("pages/palette_manager.py")

    # Titre et langue
    col_title, col_flag1, col_flag2 = st.columns([0.96, 0.02, 0.02])
    with col_title:
        st.markdown(
            f"<h1 style='margin-bottom:0.15em; margin-top:0.2em'>{t('title', lang)}</h1>",
            unsafe_allow_html=True
        )
    with col_flag1:
        st.markdown("<div style='height: 0.8em'></div>", unsafe_allow_html=True)
        if st.button("🇫🇷", key="lang_fr_btn"):
            st.session_state["lang"] = "fr"
            st.rerun()
    with col_flag2:
        st.markdown("<div style='height: 0.8em'></div>", unsafe_allow_html=True)
        if st.button("🇬🇧", key="lang_en_btn"):
            st.session_state["lang"] = "en"
            st.rerun()
    lang = st.session_state["lang"]

    # Gestion synchronisée de la palette
    palette_choices = get_palette_choices()
    if "current_palette" not in st.session_state:
        st.session_state["current_palette"] = palette_choices[0]

    palette_choice = st.selectbox(
        t("choose_palette", lang),
        palette_choices,
        index=palette_choices.index(st.session_state["current_palette"])
    )

    if palette_choice != st.session_state["current_palette"]:
        st.session_state["current_palette"] = palette_choice
        for key in [
            "code_grid_full", "result_img_full", "rgb_to_code", "alpha_arr", "nb_colors_full",
            "block_size", "input_width", "slider_width", "ignored_colors", "palette_filter_name"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # --- Chargement de la palette et application du filtre ---
    palette_df = load_palette(st.session_state["current_palette"], st)
    if palette_df is not None:
        if "ignored_colors" not in st.session_state or st.session_state.get("palette_filter_name", "") != st.session_state["current_palette"]:
            st.session_state.ignored_colors = set()
            st.session_state.palette_filter_name = st.session_state["current_palette"]

        filtered_palette_df = palette_df[~palette_df['Code'].isin(st.session_state.ignored_colors)].reset_index(drop=True)
        if len(filtered_palette_df) == 0:
            st.error(t("need_one_color", lang))
            filtered_palette_df = palette_df.copy()

        # --- Aperçu palette filtrée uniquement ---
        st.subheader(t("palette_overview", lang))
        if len(st.session_state.ignored_colors) > 0:
            st.markdown(
                f"<span style='background:#FFD600;padding:4px 12px 4px 12px;border-radius:8px;font-size:15px;font-weight:700;color:#444;margin-left:10px;'>{t('palette_preview_custom', lang)}</span>",
                unsafe_allow_html=True
            )

        with st.container():
            display_palette_preview_only_selected(palette_df, st.session_state.ignored_colors)

        # --- Upload de l'image ---
        uploaded_file = st.file_uploader(t("upload_img", lang), type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img_orig = Image.open(uploaded_file)
            if 'last_image_name' not in st.session_state or st.session_state.last_image_name != uploaded_file.name:
                st.session_state.input_width = img_orig.width
                st.session_state.last_image_name = uploaded_file.name

            if img_orig.mode != "RGBA":
                img_orig = img_orig.convert("RGBA")

            # --- Options d'ajustement image (gris, postérisation, contraste) ---
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                use_gray = st.checkbox(t("gray", lang), value=False, key="gray_toggle")
            with col_btn2:
                use_poster = st.checkbox(t("posterize", lang), value=False, key="poster_toggle")
            with col_btn3:
                use_contrast = st.checkbox(t("contrast", lang), value=False, key="contrast_toggle")

            img_mod = img_orig.copy()
            # -- Mode gris --
            if use_gray:
                if img_mod.mode == "RGBA":
                    r, g, b, a = img_mod.split()
                    gray = Image.merge("RGB", (r, g, b)).convert("L")
                    img_mod = Image.merge("RGBA", (gray, gray, gray, a))
                else:
                    gray = img_mod.convert("L")
                    img_mod = Image.merge("RGB", (gray, gray, gray))
            # -- Posterisation --
            if use_poster:
                n_levels = st.slider(t("levels", lang), min_value=2, max_value=12, value=4)
                arr = np.array(img_mod)
                if img_mod.mode == "RGBA":
                    rgb = arr[..., :3]
                    alpha = arr[..., 3:]
                else:
                    rgb = arr
                    alpha = None

                if use_gray or (img_mod.mode in ("L", "RGB") and np.all(rgb[..., 0] == rgb[..., 1])):
                    val = rgb[..., 0]
                    bins = np.linspace(0, 256, n_levels + 1, dtype=int)
                    digitized = np.digitize(val, bins) - 1
                    poster = (bins[digitized] + bins[digitized+1]) // 2
                    new_rgb = np.stack([poster]*3, axis=-1).astype(np.uint8)
                else:
                    bins = np.linspace(0, 256, n_levels + 1, dtype=int)
                    digitized = np.digitize(rgb, bins) - 1
                    poster = (bins[digitized] + bins[digitized+1]) // 2
                    new_rgb = poster.astype(np.uint8)

                if alpha is not None:
                    new_arr = np.concatenate([new_rgb, alpha], axis=-1)
                    img_mod = Image.fromarray(new_arr, "RGBA")
                else:
                    img_mod = Image.fromarray(new_rgb, "RGB")

            # -- Contraste (avant resize) --
            if use_contrast:
                contrast_value = st.slider(
                    t("contrast_slider", lang),
                    min_value=0.5,
                    max_value=2.5,
                    value=1.0,
                    step=0.05
                )
            else:
                contrast_value = 1.0

            # --- Ajustement de la taille (redimensionnement) ---
            st.markdown(f"### {t('resize_title', lang)}")
            if "initial_width" not in st.session_state:
                st.session_state.initial_width = img_mod.width
            if "input_width" not in st.session_state:
                st.session_state.input_width = img_mod.width

            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns([1, 1, 1, 1, 3])
            if col_b1.button(t("btn_init", lang)):
                st.session_state.input_width = st.session_state.initial_width
            if col_b2.button(t("btn_half", lang)):
                st.session_state.input_width = max(8, st.session_state.initial_width // 2)
            if col_b3.button(t("btn_quarter", lang)):
                st.session_state.input_width = max(8, st.session_state.initial_width // 4)
            if col_b4.button(t("btn_eighth", lang)):
                st.session_state.input_width = max(8, st.session_state.initial_width // 8)
            with col_b5:
                width = st.number_input(
                    t("width_label", lang),
                    min_value=8,
                    max_value=5000,
                    step=1,
                    key="input_width"
                )

            ratio = img_mod.height / img_mod.width
            new_height = int(width * ratio)
            st.write(f"{t('new_size', lang)} : {width} x {new_height}")

            resized_img = resize_and_contrast(img_mod, width, contrast_value)

            # --- Affichage images avant/après ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:600;'>{t('original_img', lang)}</div>", unsafe_allow_html=True)
                img_disp = upscale_for_pixel_perfect(img_orig)
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center;'>
                        <img src='data:image/png;base64,{pil_to_base64(img_disp)}'
                            style='border:2px solid #888; border-radius:12px; box-shadow:0 0 6px #aaa; image-rendering:pixelated; display:block;max-height:480px'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:600;'>{t('modified_img', lang)}</div>", unsafe_allow_html=True)
                img_mod_disp = upscale_for_pixel_perfect(resized_img)
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center;'>
                        <img src='data:image/png;base64,{pil_to_base64(img_mod_disp)}'
                            style='border:2px solid #888; border-radius:12px; box-shadow:0 0 6px #aaa; image-rendering:pixelated; display:block;max-height:480px'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --- Export/conversion ---
            st.markdown(f"### {t('export_title', lang)}")
            colb1, colb2, colb3, colb4, colb5 = st.columns([1,1,1,2,1])
            if "block_size" not in st.session_state:
                st.session_state.block_size = 50
            with colb1:
                if st.button("19 x 19"):
                    st.session_state.block_size = 19
            with colb2:
                if st.button("25 x 25"):
                    st.session_state.block_size = 25
            with colb3:
                if st.button("50 x 50"):
                    st.session_state.block_size = 50
            with colb4:
                custom_size = st.number_input(t("custom_block_size", lang), min_value=5, max_value=100, value=st.session_state.block_size, step=1)
                if custom_size != st.session_state.block_size:
                    st.session_state.block_size = custom_size

            block_size = st.session_state.block_size
            st.write(f"{t('block_size_label', lang)} : {block_size} x {block_size}")

            col_left, col_center, col_right = st.columns([2,1,2])
            with col_center:
                convert_clicked = st.button(t("convert_btn", lang))

            if convert_clicked:
                with st.spinner(t("convert_in_progress", lang)):
                    rgb_palette = filtered_palette_df.dropna(subset=['R', 'G', 'B'])[['R', 'G', 'B']].to_numpy().astype(int)
                    code_grid, result_img, alpha_arr, rgb_to_code = convert_image_to_palette(
                        resized_img, filtered_palette_df, rgb_palette, filtered_palette_df
                    )
                st.session_state.code_grid_full = code_grid
                st.session_state.result_img_full = result_img
                st.session_state.rgb_to_code = rgb_to_code
                st.session_state.alpha_arr = alpha_arr
                codes_present_full = [c for c in code_grid.flatten() if c not in ('', None)]
                st.session_state.nb_colors_full = len(set(codes_present_full))
                st.success(t("convert_success", lang))

            if (
                "code_grid_full" in st.session_state and
                st.session_state.code_grid_full is not None
            ):
                code_grid_full = st.session_state.code_grid_full
                result_img_full = st.session_state.result_img_full
                rgb_to_code = st.session_state.rgb_to_code
                alpha_arr = st.session_state.alpha_arr
                nb_colors_full = st.session_state.nb_colors_full

                codes_present_full = [c for c in code_grid_full.flatten() if c not in ('', None)]
                nb_colors_init = len(set(codes_present_full))
                nb_colors = st.slider(
                    t("current_nb_colors", lang),
                    min_value=1,
                    max_value=nb_colors_init,
                    value=nb_colors_init,
                    step=1
                )
                code_grid = code_grid_full.copy()
                result_img = result_img_full.copy()
                for _ in range(nb_colors_init - nb_colors):
                    code_grid, result_img, ok = diminish_colors(
                        code_grid, result_img, rgb_to_code
                    )
                    if not ok:
                        break
                st.write(f"{nb_colors} {t('current_nb_colors', lang)}")

                result_img_pil = Image.fromarray(result_img.astype('uint8'))
                img_with_grid = result_img_pil.copy()
                img_with_grid = draw_block_grid(img_with_grid, block_size, color=(255,0,0), width=1)
                b64_img_grid = pil_to_base64(img_with_grid)
                st.markdown(
                    f"""
                    <div style='width:100%; max-width:500px; margin:auto;'>
                        <img src='data:image/png;base64,{b64_img_grid}'
                        style='width:100%; height:auto; image-rendering:pixelated; border:2px solid #888;'/>
                    </div>
                    <div style='font-size:12px; color:#d00;'>{t('summary_blocks', lang)} ({block_size} x {block_size})</div>
                    """,
                    unsafe_allow_html=True
                )

                code_list = code_grid.flatten()
                counts = Counter([code for code in code_list if code != '' and code is not None])
                sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
                df_récap = pd.DataFrame(sorted_counts, columns=[t("convert_btn", lang), "Quantité"])
                n_blocks_rows = math.ceil(code_grid.shape[0] / block_size)
                n_blocks_cols = math.ceil(code_grid.shape[1] / block_size)
                n_blocks_total = n_blocks_rows * n_blocks_cols

                st.markdown(f"**{t('total_blocks', lang)} {n_blocks_total}**")
                st.subheader(t("recap", lang))
                st.dataframe(df_récap, hide_index=True)

                code_blocks = split_grid(code_grid, block_size)
                img_blocks = split_grid(result_img, block_size)
                alpha_blocks = split_grid(alpha_arr, block_size)
                st.subheader(t("export_blocks", lang))
                code_list_all = code_grid.flatten()
                counts_all = Counter([code for code in code_list_all if code != '' and code is not None])
                sorted_counts_all = sorted(counts_all.items(), key=lambda x: -x[1])

                rgb_to_code = {
                    str(row['Code']): {'rgb': (int(row['R']), int(row['G']), int(row['B']))}
                    for _, row in filtered_palette_df.iterrows()
                }

                # Initialisation session state si besoin
                if "export_excel_ready" not in st.session_state:
                    st.session_state.export_excel_ready = False
                if "export_excel" not in st.session_state:
                    st.session_state.export_excel = None
                if "export_png_ready" not in st.session_state:
                    st.session_state.export_png_ready = False
                if "export_png" not in st.session_state:
                    st.session_state.export_png = None

                code_blocks = split_grid(code_grid, block_size)
                img_blocks = split_grid(result_img, block_size)
                alpha_blocks = split_grid(alpha_arr, block_size)
                code_list_all = code_grid.flatten()
                counts_all = Counter([code for code in code_list_all if code != '' and code is not None])

                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    if st.button(t("prepare_excel", lang)):
                        with st.spinner(t("export_excel_prep", lang)):
                            excel_bytes = export_to_excel_multi(
                                code_blocks,
                                img_blocks,
                                alpha_blocks,
                                rgb_to_code,
                                block_size,
                                counts_all,
                                code_grid.shape
                            )
                            st.session_state.export_excel = excel_bytes
                            st.session_state.export_excel_ready = True
                    if st.session_state.export_excel_ready:
                        st.download_button(
                            label=t("download_excel", lang),
                            data=st.session_state.export_excel,
                            file_name="pixelizer_export.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                with col_export2:
                    if st.button(t("prepare_png", lang)):
                        with st.spinner(t("export_png_prep", lang)):
                            zip_png = export_png_and_zip(
                                code_grid,
                                result_img,
                                alpha_arr,
                                rgb_to_code,
                                block_size
                            )
                            st.session_state.export_png = zip_png
                            st.session_state.export_png_ready = True
                    if st.session_state.export_png_ready:
                        st.download_button(
                            label=t("download_png", lang),
                            data=st.session_state.export_png,
                            file_name="pixelizer_export.zip",
                            mime="application/zip"
                        )

if __name__ == "__main__":
    main()
