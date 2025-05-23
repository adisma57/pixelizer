def display_palette_preview(palette_df, st, cell_size=32):
    colors_html = ""
    for idx, row in palette_df.iterrows():
        try:
            color = '#{:02X}{:02X}{:02X}'.format(int(row['R']), int(row['G']), int(row['B']))
        except:
            continue
        txt_col = "#fff" if int(row['R'])*0.299 + int(row['G'])*0.587 + int(row['B'])*0.114 < 128 else "#000"
        colors_html += f"""
            <div style='
                width:{cell_size}px; height:{cell_size}px; background:{color};
                display:inline-block; margin:5px 4px 0 0; border-radius:8px; border:1.5px solid #888; vertical-align:top; position:relative; text-align:center;'>
                <span style='position:absolute; left:0; right:0; bottom:2px; font-size:0.7em; color:{txt_col}; font-family:monospace;'></span>
            </div>
        """
    st.markdown(colors_html, unsafe_allow_html=True)
