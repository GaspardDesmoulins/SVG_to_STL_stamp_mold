# SVG_to_STL_stamp_mold

Outil Python pour convertir un fichier SVG en moule 3D imprimable (par exemple pour la fabrication de tampons en silicone).

## Présentation du projet

Ce projet permet de générer automatiquement un moule 3D à partir d'un dessin vectoriel (SVG). Il est particulièrement adapté à la création de tampons, d'emporte-pièces ou de moules personnalisés pour la fabrication artisanale.

Pipeline principal :

1. **Lecture et normalisation du SVG** : nettoyage, héritage des attributs, gestion des transformations, support des paths et ellipses.
2. **Extraction des contours** : conversion des chemins SVG en polygones, simplification, gestion des sous-chemins.
3. **Génération du moule** : création de la base, ajout d'une bordure, gravure des motifs.
4. **Export STL** : sauvegarde du moule final au format STL, prêt à être imprimé en 3D.

## Fonctionnalités principales

- Support des fichiers SVG complexes (paths, groupes, ellipses, transformations imbriquées)
- Simplification automatique des contours (Ramer-Douglas-Peucker)
- Mise à l'échelle automatique pour respecter une dimension cible
- Ajout d'une marge et d'une bordure autour du motif
- Gravure des motifs en creux (profondeur paramétrable)
- Export des étapes intermédiaires pour le debug (SVG, STL)
- Mode interactif pour gérer les cas ambigus

## Dépendances

- Python 3.10+
- [CadQuery](https://github.com/CadQuery/cadquery)
- [svgpathtools](https://github.com/mathandy/svgpathtools)
- numpy, scipy, shapely

Installer l'environnement recommandé (exemple avec conda) :

```bash
conda env create -f conda_list -n moule_svg
conda activate moule_svg
```

## Structure du projet

- `main.py` : point d'entrée principal (CLI)
- `moule_svg_cadquery.py` : cœur du pipeline SVG → STL
- `settings.py` : paramètres globaux (épaisseurs, marges, etc.)
- `stls/` : STL générés
- `svgs/` : SVG sources
- `debug_*/` : dossiers de debug avec SVG normalisés, STL intermédiaires, etc.
- `tests/` : tests unitaires

## Utilisation

### Génération d'un moule

```bash
python main.py --svg ./<nom_fichier_entrée>.svg --size 40 --output ./stls/nom_fichier_sortie.stl --export-steps
```

**Options principales :**

- `--svg` : chemin du fichier SVG source
- `--size` : taille maximale du motif (mm)
- `--output` : chemin du fichier STL de sortie
- `--export-steps` : export des étapes intermédiaires (debug)

### Exemple de configuration de débogage VS Code

```json
{
  "name": "Débogueur Python : Fichier chat STL",
  "type": "debugpy",
  "request": "launch",
  "program": "${file}",
  "args": [
    "--svg",
    "le_chat.svg",
    "--size",
    "40",
    "--output",
    "moule_chat.stl",
    "--export-steps"
  ],
  "console": "integratedTerminal"
}
```

## Aspects avancés et debug

- **Debug visuel** : chaque exécution crée un dossier `debug_<nom_svg>/` avec :
  - SVG normalisé
  - STL de la base, étapes de gravure, SVG de résumé
- **Gestion des erreurs** : le script tente de corriger ou d’ignorer les polygones problématiques (mode interactif ou automatique)
- **Tests** : des tests unitaires sont disponibles dans `tests/` pour valider la chaîne SVG → STL

## Limitations connues

- Certains SVG très complexes ou mal formés peuvent nécessiter un nettoyage manuel
- Les transformations SVG avancées (hors `matrix()`) sont partiellement supportées
- Les textes et images raster ne sont pas convertis

## Auteurs et licence

Projet initial par Gaspard Desmoulins. Licence MIT.
