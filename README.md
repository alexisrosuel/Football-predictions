# Football-predictions

Ce projet a pour but d'utiliser des données historiques de rencontres de football, et d'appliquer une modélisation Bayésienne Hierarchique afin de proposer une prédiction sur l'issue du match.

En particulier, les paramètres utilisés sont le nom de chaque équipe, leur durée de récupération depuis leur dernier match, ainsi que leur dynamique (historique des 5 derniers matchs).

2 résultats sont renvoyés sous forme de graphique :
- la probabilité de victoire / défaite / match nul des équipes
- la probabilité de réalisation du score final (0-0, 1-0, etc.). Le score le plus probable est aussi affiché
