import drrdTools as dr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os




def exp_dataframe(PREFIX:str, ABOVE:int, LAST_SESSION:int, RATS:list):
    
    plt.style.use('ggplot')
    sns.set(style='ticks')

    DATA_PATH = os.path.realpath(f'../../data/raw/{PREFIX}')+'/'
    OUTPUT_PATH = os.path.realpath('../../data/processed/')+'/'
    ALL_SESSIONS = np.arange(1,LAST_SESSION+1)

    def find_frac_above(rat:int, session:int, D, above= ABOVE):
        return [rat, session,\
                        len(D[D[:,0]>above]) / len(D), len(D), above]


    # --- main --- 
    frac_above = []
    all_data   = []


    for rat in RATS: 
        for session in ALL_SESSIONS:
            print(rat, session)
            D = dr.drrd(prefix= PREFIX, animalID=rat, sessions=[session],
                        dataPath=DATA_PATH, plotFlag=True, events_to_eliminate=(5,9))
            
            frac_above.append( find_frac_above(rat, session, D) )
            
            if type(all_data) == list:
                all_data = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                                    'valid','criterion','session', 'group'])
                all_data.loc[:,'rat'] = rat
                all_data.loc[:,'trial'] = np.arange(1,len(all_data)+1)
                
            else:
                thisD = pd.DataFrame(D, columns= ['duration','iti','reinforced',\
                                                'valid','criterion','session', 'group'])
                thisD.loc[:,'rat'] = rat
                thisD.loc[:,'trial'] = np.arange(1,len(thisD)+1)
                
                all_data = pd.concat((all_data,thisD))


    df = all_data[['rat', 'group', 'session','trial','duration', 'iti', 'reinforced', 'criterion' ]]
    df = df.astype({'rat':int,'session':int,'reinforced':int})
    df.to_csv(os.path.realpath('../../data/processed/')+'/'+f'{PREFIX}_tentative.csv', index=False)

    return(df)

    #dffrac = pd.DataFrame(frac_above, columns=['rat','group','session','frac','n_trials','above'])
    #dffrac.to_csv(os.path.realpath('../../data/processed/')+'/'+ f'{PREFIX}_frac_tentative.csv', index=False)

    #plt.figure()
    #sns.boxplot(x='group',y='frac', data= dffrac)
    #plt.ylabel(f'fraction above {ABOVE}s')

