import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from collections import Counter
import math

from core.translations import translations
from core.palette import get_palette_choices, load_palette
from core.image_processing import (
    resize_and_contrast, convert_image_to_palette,
    diminish_colors, draw_block_grid, split_grid, upscale_for_pixel_perfect
)
from core.exporters import export_to_excel_multi, export_png_and_zip
from utils.streamlit_helpers import display_palette_preview

st.set_page_config(page_title="Pïxelizer", layout="wide")

def t(key, lang):
    return translations.get(key, {}).get(lang, translations.get(key, {}).get("fr", key))

def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64

def main():
    # -- LANG SELECTOR TOP RIGHT --
    if "lang" not in st.session_state:
        st.session_state["lang"] = "fr"
    lang = st.session_state["lang"]

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

    palette_choices = get_palette_choices()
    palette_choice = st.selectbox(t("choose_palette", lang), palette_choices)

    # PATCH: Reset state on palette change
    if "last_palette" not in st.session_state:
        st.session_state.last_palette = palette_choice
    if palette_choice != st.session_state.last_palette:
        for key in [
            "code_grid_full", "result_img_full", "rgb_to_code", "alpha_arr", "nb_colors_full",
            "block_size", "input_width", "slider_width"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_palette = palette_choice
        st.rerun()

    palette_df = load_palette(palette_choice, st)
    if palette_df is not None:
        st.subheader(t("palette_preview", lang))
        display_palette_preview(palette_df, st)

        uploaded_file = st.file_uploader(t("upload_img", lang), type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img_orig = Image.open(uploaded_file)
            if 'last_image_name' not in st.session_state or st.session_state.last_image_name != uploaded_file.name:
                st.session_state.input_width = img_orig.width
                st.session_state.last_image_name = uploaded_file.name
                

            if img_orig.mode != "RGBA":
                img_orig = img_orig.convert("RGBA")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                use_gray = st.checkbox(t("gray", lang), value=False, key="gray_toggle")
            with col_btn2:
                use_poster = st.checkbox(t("posterize", lang), value=False, key="poster_toggle")
            with col_btn3:
                use_contrast = st.checkbox(t("contrast", lang), value=False, key="contrast_toggle")

            img_mod = img_orig.copy()
            if use_gray:
                if img_mod.mode == "RGBA":
                    r, g, b, a = img_mod.split()
                    gray = Image.merge("RGB", (r, g, b)).convert("L")
                    img_mod = Image.merge("RGBA", (gray, gray, gray, a))
                else:
                    gray = img_mod.convert("L")
                    img_mod = Image.merge("RGB", (gray, gray, gray))

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

            # Slider contraste AVANT Redimensionner
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

            # Affichage images haut
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:600;'>{t('original_img', lang)}</div>", unsafe_allow_html=True)
                img_disp = upscale_for_pixel_perfect(img_orig)
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center;'>
                        <img src='data:image/png;base64,{pil_to_base64(img_disp)}'
                            style='border:2px solid #888; border-radius:12px; box-shadow:0 0 6px #aaa; image-rendering:pixelated; display:block;'/>
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
                            style='border:2px solid #888; border-radius:12px; box-shadow:0 0 6px #aaa; image-rendering:pixelated; display:block;'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ============= SUITE DE L'APPLI =============
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
                    rgb_palette = palette_df.dropna(subset=['R', 'G', 'B'])[['R', 'G', 'B']].to_numpy().astype(int)
                    code_grid, result_img, alpha_arr, rgb_to_code = convert_image_to_palette(
                        resized_img, palette_df, rgb_palette, palette_df
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
                    for idx, row in palette_df.iterrows()
                    if not pd.isnull(row['R']) and not pd.isnull(row['Code'])
                }
                
                with st.spinner(t("export_excel_prep", lang)):
                    excel_file = export_to_excel_multi(
                        code_blocks, img_blocks, alpha_blocks, rgb_to_code, block_size, 
                        sorted_counts_all, code_grid.shape
                    )
                st.download_button(
                    label=t("download_excel", lang),
                    data=excel_file,
                    file_name="perler_export_blocs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                with st.spinner(t("export_png_prep", lang)):
                    zip_png = export_png_and_zip(code_grid, result_img, alpha_arr, rgb_to_code, block_size)
                st.download_button(
                    label=t("download_zip", lang),
                    data=zip_png,
                    file_name="perler_blocs.zip",
                    mime="application/zip"
                )
        else:
            st.info(t("load_img_info", lang))
    else:
        st.info(t("load_img_info", lang))


if __name__ == "__main__":
    main()
