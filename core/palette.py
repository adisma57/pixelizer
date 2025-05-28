import pandas as pd

def get_palette_choices():
    return ["Artkal Mini C", "Artkal Midi S", "Hama", "Perler", "DMC"]

def load_palette(choice, st):
    palettes = {
        "Artkal Mini C": "artkal_c_mini.csv",
        "Hama": "hama.csv",
        "Artkal Midi S": "artkal_s_midi.csv",
        "Perler": "perler.csv",
        "DMC" : "DMC.csv"
    }
    if choice != "Importer mon CSV...":
        palette_filename = palettes[choice]
        return pd.read_csv(palette_filename)
    else:
        uploaded_palette = st.file_uploader("Uploader votre palette CSV", type=['csv'])
        if uploaded_palette:
            return pd.read_csv(uploaded_palette)
    return None


def filter_palette(palette_df, ignored_codes):
    """
    Retourne une version filtrée de la palette, sans les codes spécifiés.
    """
    return palette_df[~palette_df["Code"].isin(ignored_codes)].reset_index(drop=True)
