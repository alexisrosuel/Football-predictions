# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import theano
import pymc3 as pm
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from matplotlib import gridspec


def define_model(num_teams):
    with pm.Model() as model:
        # shape parameter des lois de student et de la loi normale pour le bruit 
        hyper_lambda_team = pm.Uniform('hyper_lambda_team', lower=0, upper=5, shape=1)
        hyper_lambda_coeff = pm.Uniform('hyper_lambda_coeff', lower=0, upper=5, shape=1)  
        
        # Paramètres spécifiques à chaque équipe  
        attaque = pm.StudentT("attaque", nu=2, mu=0, lam=hyper_lambda_team, shape=num_teams)
        defense = pm.StudentT("defense", nu=2, mu=0, lam=hyper_lambda_team, shape=num_teams)
        
        # Paramètre lié aux conditions du match
        coeff_home = pm.StudentT("coeff_home", nu=2, mu=0, lam=hyper_lambda_coeff, shape=1)
        coeff_recup = pm.StudentT("coeff_recup", nu=2, mu=0, lam=hyper_lambda_coeff, shape=1)
        coeff_dynamique = pm.StudentT("coeff_dynamique", nu=2, mu=0, lam=hyper_lambda_coeff, shape=1)
        coeff_perf = pm.StudentT("coeff_perf", nu=2, mu=0, lam=hyper_lambda_coeff, shape=1)
        
        # et un intercept
        intercept = pm.Normal("intercept", mu=0, sd=hyper_lambda_coeff, shape=1)
        
        home_theta  = pm.math.exp(intercept + 
                                  coeff_home + 
                                  coeff_recup * (model_input_conditions[0] - model_input_conditions[1]) + 
                                  coeff_dynamique * (model_input_conditions[2] - model_input_conditions[3]) +
                                  coeff_perf * (attaque[model_input_team[0]] - defense[model_input_team[1]])
                                  )
        away_theta  = pm.math.exp(intercept + 
                                  coeff_recup * (model_input_conditions[1] - model_input_conditions[0]) +
                                  coeff_dynamique * (model_input_conditions[3] - model_input_conditions[2]) +
                                  coeff_perf * (attaque[model_input_team[1]] - defense[model_input_team[0]])
                                 )
    
        # likelihood of observed data 
        home_points = pm.Poisson('home_points', mu=home_theta, observed=model_output[0])
        away_points = pm.Poisson('away_points', mu=away_theta, observed=model_output[1])
    
    return model

def get_info():
    home_team = int(input('Id de l\'equipe jouant a domicile : '))
    away_team = int(input('Id de l\'equipe jouant a l\'exterieur : '))
    
    home_recuperation = int(input('Duree de recuperation en jours (de 1 a 7) de l\'equipe a domicile : '))
    away_recuperation = int(input('Duree de recuperation en jours (de 1 a 7) de l\'equipe a l\'exterieur : '))
    
    home_dynamique = raw_input('Dynamique (historique des 5 derniers matchs, ex: GGNPP) de l\'equipe a domicile : ')
    away_dynamique = raw_input('Dynamique (historique des 5 derniers matchs, ex: GGNPP) de l\'equipe a l\'exterieur : ')
    
    return (home_team, away_team, home_recuperation, away_recuperation, home_dynamique, away_dynamique)
    
def calcul_score_dynamique(dynamique):
    score = 0
    for i in range(5):
        score = score + (1 if dynamique[i] == 'G' else -1 if dynamique[i] == 'P' else 0) * (float(i+1)/float(5))
    return score

def simulation(model):
    # on charge la trace qui contient les distributions des parametres
    trace = pm.backends.text.load('C:\\Users\\Alexis\\Documents\\Football-predictions\\Trace', model=model)
    
    print(trace)
    
    # on simule le match désiré
    ppc = pm.sample_ppc(trace, model=model, samples=100)    
    
    return ppc

def declare_shared_variable(home_team, away_team, home_recuperation, away_recuperation, home_dynamique, away_dynamique):
    model_input_team = theano.shared(np.array([home_team, away_team]))
    model_input_conditions = theano.shared(np.array([home_recuperation, away_recuperation,
                                                     home_dynamique, away_dynamique])
                                            )
    model_output = theano.shared(np.array([0,0])) #on met n'importe quoi ici, c'est juste pour définir le modèle
    
    return model_input_team, model_input_conditions, model_output

"""Pour chaque dataframe, on détermine l'équipe qui gagne"""
def vainqueur(row):
    if row[0] > row[1]:
        return 'H'
    elif row[0] < row[1]:
        return 'A'
    else:
        return 'N'

def analyse(prediction):
    """On rassemble dans des dataframe"""
    away_prediction = pd.DataFrame(prediction['away_points'])
    home_prediction = pd.DataFrame(prediction['home_points'])  
    
    prediction_score = home_prediction.merge(away_prediction, how='inner', left_index=True, right_index=True)
    
    prediction_score.columns = ['home_prediction', 'away_prediction']
    
    prediction_score['prediction_vainqueur'] = prediction_score.apply(vainqueur, axis=1) 
    
    return prediction_score

def plot_resultats(prediction_score):    
    """Et on ajoute le score le plus probable"""    
    home_score, away_score = prediction_score.groupby(["home_prediction", "away_prediction"]).size().idxmax()
    resultat = prediction_score.prediction_vainqueur.value_counts().idxmax(), home_score, away_score
    
    df_affichage = pd.DataFrame([prediction_score.prediction_vainqueur.value_counts()*100/len(prediction_score)])
    
    
    if 'H' not in df_affichage:
        df_affichage['H'] = 0.0
    if 'A' not in df_affichage:
        df_affichage['A'] = 0.0
    if 'N' not in df_affichage:
        df_affichage['N'] = 0.0
        
    df_affichage = df_affichage.transpose()
        
    df_affichage.index = [equipes[home_team] if e=='H' else equipes[away_team] if e=='A' else 'Nul' for e in df_affichage.index]
    
    fig = plt.figure(figsize=(16,12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4]) 
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax2.set_title('score le plus probable ' + str(home_score) + '-' + str(away_score))
    seaborn.heatmap(df_affichage, vmin=0, vmax=100, annot=True, cmap='RdBu_r', ax=ax1)
    seaborn.kdeplot(data=prediction_score.home_prediction, data2=prediction_score.away_prediction, shade=True, ax=ax2)
    plt.show()
    

if __name__ == '__main__':
    print('Bienvenue dans le programme de prédiction de matchs de football')
    print('Vous allez être ammené à renseigner plusieurs paramètres de conditions du match (quelles équipes, durée de récupération de chacune, etc.)')
    print('Voici les indices correspondant à chacune des équipes')
    
    equipes = np.array(['Everton', 'Arsenal', 'Bolton Wanderers', 'Middlesbrough',
       'Sunderland', 'West Ham United', 'Hull City', 'Chelsea',
       'Manchester United', 'Aston Villa', 'Blackburn Rovers', 'Liverpool',
       'Tottenham Hotspur', 'West Bromwich Albion', 'Fulham',
       'Newcastle United', 'Stoke City', 'Manchester City',
       'Wigan Athletic', 'Portsmouth', 'Wolverhampton Wanderers',
       'Birmingham City', 'Burnley', 'Blackpool', 'Queens Park Rangers',
       'Swansea City', 'Norwich City', 'Reading', 'Southampton',
       'Crystal Palace', 'Cardiff City', 'Leicester City', 'Bournemouth',
       'Watford'], dtype=object)
    
    num_teams = len(equipes) 
    
    while(True):
    
        for i in range(len(equipes)):
            print(str(i) + ' --> ' + equipes[i])
            
        home_team, away_team, home_recuperation, away_recuperation, home_dynamique, away_dynamique = get_info()
    
        # On calcule l'équivalent numérique de la dynamique (modèle empirique perso)
        home_dynamique = calcul_score_dynamique(home_dynamique)
        away_dynamique = calcul_score_dynamique(away_dynamique)
        
        # On déclare les 'shared variable' theano (on doit le faire AVANT de définir le modèle pymc)
        model_input_team, model_input_conditions, model_output = declare_shared_variable(home_team, away_team, home_recuperation, away_recuperation, home_dynamique, away_dynamique)
         
        # on défini le modèle (issu des notebooks python)
        model = define_model(num_teams)
        
        # on simule à partir de notre modèle
        prediction = simulation(model)
    
        # on analyse les résultats
        resultat = analyse(prediction)
        
        # On affiche les résultats
        plot_resultats(resultat)