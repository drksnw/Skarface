# Skarface

## Cahier des charges
Le but du projet est de pouvoir ajouter des effets, comme des images de masques ou de lunettes par exemple, sur des visages en temps réel. Ceci se fera à travers une application réalisée en Python utilisant le flux de la caméra de l'ordinateur.

Pour ce faire, il va falloir récupérer l'image provenant de la webcam, analyser cette dernière et récupérer les coordonnées où se trouvent les différents visages et leurs yeux. Ensuite il va falloir adapter l'image de notre masque par rapport aux données relatives aux visages puis la positionner par-dessus. 

L'application va permettre à l'utilisateur de choisir parmi plusieurs effets via un panel situé à côté du retour de la webcam. En appuyant sur une touche spécifique, il va pouvoir sauvegarder le rendu !

Exemple de l'analyse de la webcam :
![image](https://cloud.githubusercontent.com/assets/14583100/23750220/37369af2-04cc-11e7-8566-732cc09ad32f.png)
