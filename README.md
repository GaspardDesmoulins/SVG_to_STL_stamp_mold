# SVG_to_STL_stamp_mold

A python utility to convert a SVG into a 3d printable silicon stamp mold

## Génération d'un moule

- Installer l'environnement conda
- lancer la commande

`python main.py --svg ./<nom_fichier_entrée>.svg --size 40 --output ./svgs/nom_fichier_sortie.stl --export-steps`

## Vs Code debug configuration

```JSON
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
