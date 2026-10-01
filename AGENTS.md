# Environnement pour agents LLM

## Objectif

Les tests nécessitent CadQuery et ses bibliothèques natives. Utilisez l'environnement Conda défini dans `environment.yml` plutôt que le Python système.

## Préparer l'environnement

Depuis la racine du dépôt :

```bash
conda env create -f environment.yml
conda run -n svg-to-stl-stamp-mold python -c "import cadquery, cairosvg, numpy, scipy, shapely, svgpathtools"
```

Pour mettre à jour un environnement existant :

```bash
conda env update -n svg-to-stl-stamp-mold -f environment.yml --prune
```

## Valider une modification

Exécutez la suite complète dans l'environnement :

```bash
conda run -n svg-to-stl-stamp-mold python -m unittest discover -v -s tests -p "test_*.py"
```

Les tests de génération écrivent des répertoires `debug_*`; ils sont ignorés par Git. Ne modifiez pas les STL d'exemple sauf si la tâche le demande.

## Agents Copilot cloud

`.github/workflows/copilot-setup-steps.yml` installe cet environnement avant le démarrage d'un agent Copilot et exécute la suite de tests. Le workflow ne sera utilisé par les agents qu'après fusion sur la branche par défaut.

Si le solveur Conda échoue, ne remplacez pas arbitrairement CadQuery par une installation `pip` : ses dépendances OpenCascade sont fournies de manière cohérente par conda-forge. Corrigez les contraintes dans `environment.yml`, puis recréez l'environnement.
