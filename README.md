# 🎨 Pixelizer

**Pixelizer** est une application Streamlit permettant de convertir n’importe quelle image en patron optimisé pour perler beads (Artkal, Hama, etc.),  
avec export Excel et PNG (numérotation des couleurs, grilles, gestion des palettes, etc.).

![pixelizer banner](docs/banner.png) <!-- Ajoute une image si tu veux un visuel ! -->

---

## 🚀 Fonctionnalités principales

- Conversion d’images en “grille perles” adaptée à la taille et la palette choisie
- Affichage pixel-art net (pas de flou)
- Sélection de palettes (Artkal Mini, Artkal Midi, Hama, etc.)
- Affichage du code couleur dans chaque case
- Export Excel avec :
    - Cellules colorées + code couleur
    - Grille (lignes fines, noires tous les 10, rouges tous les blocs)
    - Taille de police, couleur de police adaptée au contraste
- Export PNG avec la même grille
- Multilingue (FR/EN)
- Reset intelligent des paramètres lors d’un nouveau chargement d’image

---

## 🛠️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/adisma57/pixelizer.git
cd pixelizer
