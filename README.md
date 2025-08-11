# SVG_to_STL_stamp_mold

Outil Python pour convertir un SVG en moule 3D imprimable (tampons silicone).

## Fonctionnalités principales

- Support de paths/groupes avec transformations imbriquées, rect/ellipse convertis en paths
- Mise à l’échelle automatique pour respecter une dimension cible
- Création de base + bordure (paramètres dans `settings.py`)
- Gravure en creux des motifs (profondeur paramétrable)
- Exports de debug (base STL, SVG de résumé par étape et final) via l’API

## Ce qui a changé depuis l’origine

- Normalisation complète des SVG avant conversion:
  - Conversion des `rect` et `ellipse` en `path` équivalents
  - Aplatissement des transformations imbriquées (matrix, translate, scale)
  - Centrage automatique et viewBox propre
  - Propagation des attributs hérités (fill, stroke)
- Pipeline de gravure plus robuste:
  - Groupement outer/inners par inclusion (fill-rule: evenodd)
  - Loft avec dépouille (angle par défaut 15°), fallback en extrusion simple
- Base + bordure en anneau générées proprement, export STL de la base disponible
- Génération de SVG de résumé par étape et final (via l’API), avec statut gravé/non gravé
- Outils de vérification (rasterisation + IoU) et de recalage affine (ICP)
- Suite de tests unitaires couvrant les rectangles, ellipses, transform, summary SVG, ICP

## Dépendances

- Python 3.11 (recommandé)
- cadquery, svgpathtools, numpy, scipy, shapely, matplotlib, cairosvg, pillow

### Installation rapide (Conda, Windows PowerShell)

```powershell
# Créez et activez l’environnement
conda create -n stl_mold -y -c conda-forge python=3.11
conda activate stl_mold

# Installez les libs principales via conda-forge (cadquery tire ses dépendances lourdes)
mamba install -y -c conda-forge cadquery svgpathtools numpy scipy shapely matplotlib pillow cairosvg

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
python .\main.py --svg .\svgs\le_chat.svg --size 40 --output .\stls\moule_chat.stl --keep-debug-files
```

Options actuelles:

- `--svg` chemin du SVG source
- `--size` taille max du motif (mm) [par défaut: `settings.MAX_DIMENSION`]
- `--output` chemin du STL de sortie
- `--keep-debug-files` conserve le dossier `debug_<svg>` avec les fichiers intermédiaires
- `--no-interactive` drapeau prévu pour désactiver les interactions; actuellement non utilisé par le CLI

Notes:

- L’export des étapes intermédiaires (summary SVG par étape) est disponible via l’API (`export_steps=True` dans `generate_cadquery_mold`). Le CLI ne l’expose pas encore.
- Le STL de base (avant gravure) est exporté dans le répertoire de debug quand `keep-debug-files` est utilisé.

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

- `generate_cadquery_mold(svg_file, max_dim, ..., export_steps=False, keep_debug_files=False)`
  - Retourne `(mold_solid, engraved_indices, shape_history)`
  - Crée `debug_<svg>/`; si `keep_debug_files=False`, ce dossier est supprimé à la fin
  - Si `export_steps=True`, écrit `step_<k>_summary.svg` à chaque étape + un `summary_<svg>_final.svg`

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

## Limitations connues

- Transformations supportées dans le parseur: `matrix(...)`, `translate(...)`, `scale(...)` (pas de `rotate(...)` direct)
- Les arcs sont approximés par des segments lors de certains traitements
- Textes et images raster ne sont pas pris en charge
- Certains SVG très complexes/malfichus peuvent nécessiter un nettoyage manuel

## Auteurs et licence

Projet initial par Gaspard Desmoulins. Licence MIT.
