import numpy as np
import pandas as pd

raw_data = pd.read_csv('!BoreliozAnaliza.csv')

# drop the last two columns as they are probably entry errors. !!!!!need to check!!!!!
data = raw_data.drop(['Unnamed: 64', 'Unnamed: 65'], axis='columns')


# need to check colnames as they are in Polish
colnames = 'record_time,blood,infection,residence,start_treat'.split(',')
treatments = 'doxy,ilads,buhner,cowden,liposomal,other_herbs,vitaminD,supp,oil,kambo,plasma,sugar-free,gluten-free,dairy-free,bioresonance,antimicrobial,oxygen,cannabis,binaural'.split(',')
colnames.extend(treatments)
stimulants = 'tobacco,alcohol,coffee,marijuana,other_stim'.split(',')
colnames.extend(stimulants)
colnames.extend(['num_antibiotics', 'method_antibiotics'])
symptoms = 'depression,unreal,irritability,sleep,conc,stupor,memory,stiffness,breath,fatigue,numb,strength,ache,arthralgia,headache,facial_muscle,abdominal,gastro_reflux,eye,ringing,light,sound,toothache,rash,hair,chest_pain,bladder,libido,weight,fever,Parkinson,muscle_decay'.split(',')
colnames.extend(symptoms)
colnames.append('effective')
data.columns = colnames


def one_hot_encode(df, colname):
    '''
    input: Dataframe, column name
    output: Given dataframe with the column one-hot encoded and original col removed
    '''
    dummy = pd.get_dummies(df[colname])
    dummy.columns = [colname+'_'+x for x in list(dummy)]
    df = pd.concat([df, dummy], axis='columns')
    df = df.drop(columns=[colname])
    return df
    

# =====
# blood
blood_dict = {'A RH+ 32% Polaków':'A+', '0RH+ 32% Polaków':'O+', 'B RH+ 15% Polaków':'B+', '0 RH- 6% Polaków':'O-', 
              'AB RH- 1% Polaków':'AB-', 'AB RH+ 7% Polaków':'AB+', '':'B-', 'A RH- 6% Polaków':'A-', 'nie wiem':'TBD'}
data['blood'] = data['blood'].map(blood_dict)
data = one_hot_encode(data, 'blood')


# =====
# infection
# Acinomyces (promieniowiec) & leptospiroza has no patients with it in the initial survey
infection_dict = {'Borelioza':'lyme', 'Bartonella':'bartonella', 'Mykoplazma pneumoniae':'m_pneumoniae', 'Babesia':'babesia',
                  'Chlamydia pneumoniale':'c_pneumoniae', 'Yersinia':'yersinia', 'Pasożyty':'parasite', 'Candidoza':'candida',
                  'Toxoplazmoza':'toxoplasmosis', 'Toksyny':'toxin', 'Brucelloza':'brucellosis', 'Anaplazma':'anaplasma',
                  'Lamblia':'lamblia', 'Chlamydia trachomatis':'c_trachomatis', 'Cytomegalia - CMV':'CMV', 
                  'szpitalne/oporne MRSA i VRSA':'mrsa', 'Ureaplazma':'ureaplasma', 'Acinomyces (promieniowiec)':'acinomyces',
                  'vir.Herpes zoster':'herpes', 'vir.Epstein-Barr':'epstein-barr', 'Haemophilus influenzae':'h_influenzae',
                  'leptospiroza':'leptospirosis', 'Staphylococcus aureus':'staph', 'Mykoplazma ureyacelum':'m_ureyacelum'}

infection_df = pd.DataFrame()
# for each response, split all the infections and one-hot encode manually
for row in data['infection']:
    if isinstance(row, str):
      row_df = pd.get_dummies(row.split(';')).sum().to_frame().transpose()
    else:
      # for row with nan, get a dummy
      row_df = pd.DataFrame({'Borelioza':[0]})
    infection_df = infection_df.append(row_df, ignore_index = True, sort=True)
    
infection_df = infection_df.fillna(0)
infection_df = infection_df.astype(int)
# translate
new_colnames = ['infection_'+infection_dict[x] for x in list(infection_df)]
infection_df.columns = new_colnames
# add infections to data
data = pd.concat([data, infection_df], axis='columns')
data = data.drop(columns=['infection'])


# =====
# residence
# Big city (more than 300k residents) => 3, Small city (less than 300k residents) => 2, Countryside => 1
residence_dict = {'Wieś':'country', 'Małe miasto (do 300 tysięcy mieszkańców)':'small_city',
                  'Duże miasto (powyżej 300 tysięcy mieszkańców)':'big_city'}
data['residence'] = data['residence'].map(residence_dict)
data = one_hot_encode(data, 'residence')


# =====
# start_treat
# the responses are problematic in that there are overlaps
# need to think about month translation whether to do 1, 2, 3 or 1, 4, 8, etc.
start_treat_dict = {'1-3 miesiąc':1, '4-7 miesiąc':4, '8-12 miesiąc':8, '12-18 miesiąc':12, '18 -24 miesiąc':18,
                    '2-3 rok':24, '3-6 rok':36, '6+ rok':72}
data['start_treat'] = data['start_treat'].map(start_treat_dict)


# =====
# treatments
treatment_dict = {'1-3 m':1, '4-7 m':4, '8-12 m':8, '13-18 m':13, '19 -24 m':19, 
                  '2-3 lat':24, '4-5 lat':48, '6+ lat':72}

# for multiple checked boxes, choose longest duration
def longest(string):
    '''
    input: some string or nan
    output: longest duration or nan
    '''
    if isinstance(string, str):
      return treatment_dict[max([(treatment_dict[x], x) for x in string.split(';')])[1]]
    else:
      # may be nan
      return string

data[treatments] = data[treatments].applymap(longest).fillna(0)


# =====
# stimulants
# rarely: <1/month => 1, moderately: 1-2/week => 2, often: everyday => 3
stimulant_dict = {'rzadko (raz na miesiąc lub rzadziej)':1, 'średnio (1-2 razy w tygodniu)':2,
                 'często (codziennie)':3}

for stimulant in stimulants: 
    data[stimulant] = data[stimulant].map(stimulant_dict).fillna(0)

# =====
# num_antiobiotics
num_anti_dict = {'1-3':1, '4-6':4, '7-9':7, '10+':10}
data['num_antibiotics'] = data['num_antibiotics'].map(num_anti_dict)


# =====
# method_antiobiotics
# oral, daily; oral, pulsed; oral& intra, pulsed; oral & intra, daily; intra, daily; intra, pulsed 
method_anti_dict = {'doustne  ciągłe':'oral_daily', 'doustne pulsacyjnie':'oral_pulse',
                    'dożylnie ciągłe':'intra_daily', 'dożylnie pulsacyjnie':'intra_pulse',
                    'doustne i dożylnie ciągłe':'oral_intra_daily', 'doustne i dożylnie pulsacyjne':'oral_intra_pulse',
                    'dożylnie mieszane':'oral_daily'}
data['method_antibiotics'] = data['method_antibiotics'].map(method_anti_dict)
data = one_hot_encode(data, 'method_antibiotics')


# =====
# symptoms
# currently, past, current & past
# should I encode current & past to divide into currently and past?
symptom_dict = {'Występowały':'current', 'Występują':'past', 'Występowały;Występują':'continuing'}
for symptom in symptoms:
    data[symptom] = data[symptom].map(symptom_dict)
    data = one_hot_encode(data, symptom)

data.to_csv('/Users/rosaria/Documents/DataScience/FightLyme/LymeSurveyPreproc.csv')