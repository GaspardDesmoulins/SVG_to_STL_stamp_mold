# SVG_to_STL_stamp_mold

Outil Python pour convertir un SVG en moule 3D imprimable (tampons silicone).

## Fonctionnalités principales

- Support de paths/groupes avec transformations imbriquées, rect/ellipse convertis en paths
- Mise à l’échelle automatique pour respecter une dimension cible
- Création de base + bordure (paramètres dans `settings.py`)
- Gravure en creux des motifs (profondeur paramétrable)
- Deux modes de gravure:
  - Classique: loft avec dépouille puis extrusion si nécessaire
  - Étagé (raster): par couches en « escaliers » pour une robustesse accrue sur des formes complexes
- Exports de debug:
  - STL de base du moule
  - SVG de résumé par étape automatiquement générés, et un résumé final
  - STL intermédiaires par groupe (si activé via API)

## Ce qui a changé depuis l’origine

- Normalisation complète des SVG avant conversion:
  - Conversion des `rect` et `ellipse` en `path` équivalents
  - Aplatissement des transformations imbriquées (matrix, translate, scale)
  - Centrage automatique et viewBox propre
  - Propagation des attributs hérités (fill, stroke)
- Pipeline de gravure plus robuste:
  - Groupement outer/inners par inclusion (fill-rule: evenodd)
  - Loft avec dépouille (angle par défaut 15°), fallback en extrusion simple
- Nouveau mode « gravure étagée (raster) »:
  - Rasterisation des contours, croissance/érosion par couche, extrusion de dalles (0,1 mm par défaut)
  - Paramètres API: `layer_thickness_mm`, `pixel_size_mm`, `growth_per_layer_px`
  - Exporte un STL des dalles de chaque groupe si `export_steps=True`
- Base + bordure en anneau générées proprement, export STL de la base disponible
- Génération automatique de SVG de résumé par étape et final, avec statut gravé/non gravé
- Outils de vérification (rasterisation + IoU) et de recalage affine (ICP)
- Suite de tests unitaires couvrant les rectangles, ellipses, transform, summary SVG, ICP

## Dépendances

- Python 3.11 (recommandé)
- cadquery, svgpathtools, numpy, scipy, shapely, matplotlib, cairosvg, pillow, tqdm

### Installation rapide (Conda, Windows PowerShell)

```powershell
# Créez et activez l’environnement
conda create -n stl_mold -y -c conda-forge python=3.11
conda activate stl_mold

# Installez les libs principales via conda-forge (cadquery tire ses dépendances lourdes)
mamba install -y -c conda-forge cadquery svgpathtools numpy scipy shapely matplotlib pillow cairosvg tqdm

# IMPORTANT (Windows): installer le binding Python de nlopt pour CadQuery
python -m pip install -U pip
python -m pip install nlopt
```

Remarques:

- Sur Windows, même si un paquet `nlopt` conda est présent, l’import Python peut échouer. Le wheel `pip install nlopt` résout le problème d’import (utilisé par `cadquery.occ_impl.sketch_solver`).
- Les runtimes Visual C++ nécessaires sont généralement installés via conda (vs2015_runtime, vc14_runtime).

## Structure du projet

- `main.py` : point d’entrée principal (CLI)
- `moule_svg_cadquery.py` : cœur du pipeline SVG → STL (base, bordure, gravure, summary SVG)
- `utils.py` : fonctions SVG/transformations (flatten, ICP, rasterisation…), regroupements outer/inner
- `settings.py` : paramètres globaux (épaisseurs, marges, profondeurs, taille max)
- `svgs/` : exemples de SVG sources
- `stls/` : exemples de STL générés
- `tests/` : tests unitaires (unittest)
- `debug_<nom_svg>/` : répertoire temporaire de debug (SVG normalisé, base STL, résumés SVG, rasters…)

## Utilisation (CLI)

```powershell
python .\main.py --svg .\svgs\le_chat.svg --size 40 --output .\stls\moule_chat.stl --keep-debug-files --stepped
```

Options actuelles:

- `--svg` chemin du SVG source
- `--size` taille max du motif (mm) [par défaut: `settings.MAX_DIMENSION`]
- `--output` chemin du STL de sortie
- `--keep-debug-files` conserve le dossier `debug_<svg>` avec les fichiers intermédiaires
- `--stepped` active la gravure étagée (raster) au lieu du mode classique (loft/extrusion)
- `--no-interactive` drapeau prévu pour désactiver les interactions; actuellement non utilisé par le CLI

Notes:

- Les SVG de résumé par étape sont générés automatiquement dans `debug_<svg>/` pendant la gravure; le résumé final est également écrit. Ces fichiers de debug ne sont conservés que si `--keep-debug-files` est fourni.
- Les STL intermédiaires (par groupe et dalles) ne sont exportés que si `export_steps=True` (réglable via l’API, pas exposé en CLI pour l’instant).
- Le STL de base (avant gravure) est toujours exporté dans le répertoire de debug, mais il n’est conservé que si `--keep-debug-files` est fourni.

### Exemple de configuration de débogage VS Code

```json
{
  "name": "Moule depuis SVG (CLI)",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/main.py",
  "args": [
    "--svg", "${workspaceFolder}/svgs/le_chat.svg",
    "--size", "40",
    "--output", "${workspaceFolder}/stls/moule_chat.stl",
    "--keep-debug-files"
  ],
  "console": "integratedTerminal"
}
```

## API Python (résumé)

- `generate_cadquery_mold(svg_file, max_dim, base_thickness=..., border_height=..., border_thickness=..., engrave_depth=..., margin=..., export_base_stl=True, base_stl_name="moule_base.stl", export_steps=False, keep_debug_files=False, engraving_mode="classic"|"stepped", layer_thickness_mm=0.1, pixel_size_mm=0.1, growth_per_layer_px=1)`
  - Retourne `(mold_solid, engraved_indices, shape_history)`
  - Crée `debug_<svg>/`; si `keep_debug_files=False`, ce dossier est supprimé à la fin
  - Génère `step_<k>_summary.svg` à chaque étape + un `summary_<svg>_final.svg`
  - Si `export_steps=True`:
    - Exporte aussi les STL intermédiaires: `step_<k>.stl` et `group_<k>_slabs.stl` (en mode étagé)
  - `engraving_mode`:
    - `classic`: loft avec dépouille (15° par défaut), fallback extrusion
    - `stepped`: gravure par couches (0,1 mm par défaut), grille raster `pixel_size_mm` et croissance par couche `growth_per_layer_px`

## Tests

Les tests utilisent `unittest` et couvrent: aplatissement des transforms, conversion rect/ellipse, summary SVG, ICP affine, similarité raster (IoU).

```powershell
python -m unittest -v
```

### Exécution des tests dans VS Code

1) Sélectionner l’interpréteur Python

- Command Palette → “Python: Select Interpreter” → choisissez l’interpréteur de l’env `stl_mold` (ex: `C:\Users\<vous>\.conda\envs\stl_mold\python.exe`).

1) Paramètres de découverte unittest (déjà présents dans `.vscode/settings.json`)

```jsonc
{
  "python.testing.unittestEnabled": true,
  "python.testing.unittestArgs": ["-v", "-s", "./tests", "-p", "test_*.py"],
  "python.testing.pytestEnabled": false
}
```

1) Panneau Testing

- Ouvrir Testing → “Refresh” → “Run All Tests”.

Astuce CLI équivalente (depuis la racine du projet, env activé):

```powershell
python -m unittest discover -v -s .\tests -p test_*.py
```

### Dépannage “tests non découverts / ImportError”

- Vérifiez que l’interpréteur sélectionné dans VS Code est bien l’env `stl_mold`.
- Installez le binding Python de `nlopt` si CadQuery se plaint: `python -m pip install nlopt`.
- Assurez-vous que `numpy`, `cairosvg`, `shapely`, etc. sont bien installés dans le même env.
- Les warnings `ResourceWarning: unclosed file` émis par CairoSVG sont bénins pour les tests.

## Aspects avancés et debug

- Debug visuel: `debug_<svg>/` peut contenir le SVG normalisé, le STL de base, les SVG de résumé par étape et final, des rasters PNG.
- Similarité raster: calcul IoU entre SVG normalisé et résumé final (voir tests).
- ICP affine: outils pour comparer des polygones ou aligner deux formes (utils.py).
- Mode étagé: utilise `scipy.ndimage` pour la dilatation/érosion si disponible; sinon, un fallback pur NumPy est utilisé (plus lent).

## Limitations connues

- Transformations supportées dans le parseur: `matrix(...)`, `translate(...)`, `scale(...)` (pas de `rotate(...)` direct)
- Les arcs sont approximés par des segments lors de certains traitements
- Textes et images raster ne sont pas pris en charge
- Certains SVG très complexes/malfichus peuvent nécessiter un nettoyage manuel

## Auteurs et licence

Projet initial par Gaspard Desmoulins. Licence MIT.
