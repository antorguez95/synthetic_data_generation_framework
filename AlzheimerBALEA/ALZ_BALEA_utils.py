# Copyright (C) 2022 Antonio Rodriguez
# 
# This file is part of synthetic_data_generation_framework.
# 
# synthetic_data_generation_framework is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# synthetic_data_generation_framework is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with synthetic_data_generation_framework.  If not, see <http://www.gnu.org/licenses/>.

# Dependencies 
import os 
import pandas as pd
import numpy as np

from  sdg_utils import Positive, Binary 

import openpyxl

from sklearn.preprocessing import OneHotEncoder

from typing import Tuple, List

def prepare_ALZ_BALEA(dataset_path : str = "", filename : str = "") ->  Tuple[pd.DataFrame, 
                                                        pd.DataFrame, pd.DataFrame, List, str] :
    """Read the Alzheimer-Balea dataset from a .xlsx file and suit it to be processed 
    as a pd.DataFrame. It returns tha dataset dataframe and strings associated to 
    it to easy its management.

    Args:
    -----
            dataset_path: path where ALZ-BALEA dataset is stored. Set by default.
            filename : file name of the .csv containing the dataset. Set by default.

    Returns:
    --------
            data: dataframe containing the whole dataset
            X : dataframe containing the dataset features
            Y : dataframe containing only the target variable
            cols_names: list of strings containing feature names. 
            y_tag: string containing target variable name.
    """

    # Go to dataset path
    os.chdir(dataset_path)

    # Open the Excel file 
    bd = openpyxl.load_workbook(filename)

    # Load the useful information of the Excel file 
    sheet = bd['Tesis definitiva']
    
    # Convert sheet into DataFrame 
    data = pd.DataFrame(sheet.values)
    
    # Fix data properly in the dataframe 
    tags = data.iloc[0]
    data.columns = tags

    # Drop useless rows and columns 
    data = data.drop(index = [0,86,87,88,89,90]) 
    data.reset_index(drop=True, inplace=True)
    data = data.drop(['Grupo','ID','Pfeiffer','Comentario', 'HiperEspectMIC', 'HiperEspectMAC'], axis=1)

    # Replace "NULL" values by a integer to be handled later 
    data = data.replace(to_replace = "#NULL!", value = 99999)

    # Store features' and target variable's names 
    cols_names_prev = data.columns
    y_tag = cols_names_prev[len(cols_names_prev)-1]
    cols_names = cols_names_prev[0:cols_names_prev.size]
    
    # Substitute value of 6 (non-existant) in AntNeurol by 99999, as if it was "NULL"
    data['AntNeurolog'][61] = 99999

    # Drop those rows which has the output 'TNeurolog' equals to 99999 (NULL)
    data = data[data['TNeurocog'] < 2]

    # Save X, Y, feature names and Y name 
    y_tag = cols_names[len(cols_names)-1]
    cols_names = cols_names[0:len(cols_names)-1]
    X = data[cols_names]
    Y = data[y_tag]
    
    return data, X, Y, cols_names, y_tag

def numerical_conversion(data : np.array, features : str, y_col : str) -> Tuple[pd.DataFrame, 
                                            pd.DataFrame, pd.DataFrame]:
    """Fix all Alzheimer-Balea database features data types to its original type after
    KNNImputer is used, since this functions returns only a floating points ndarray. 
    For more, check sklearn documentation of this function at
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html. After 
    fixing datatypes, an ndarray to pd.DataFrame conversion is performed. Notice that this
    operation is only done in the fields that were not originally floats.

    Args:
    -----
            data: data returned by KNN Imputer (float data type).
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe containing the whole dataset after imputation
            X : dataframe containing the dataset features after imputation
            y : dataframe containing only the target variable after imputation
    """
    # From ndarray to pd.DataFrame
    names = features.insert(len(features), y_col)
    data = pd.DataFrame(data, columns = names)
    
    # Fixing necessary datatypes to int (including categorical variables)
    data['Edad'] = data['Edad'].astype(int)
    data['Sexo'] = data['Sexo'].astype(int)
    data['EstCivil'] = data['EstCivil'].astype(int)
    data['ActLaboral'] = data['ActLaboral'].astype(int)
    data['FormatConvivencia'] = data['FormatConvivencia'].astype(int)
    data['NivelEducativo'] = data['NivelEducativo'].astype(int)
    data['ActIntelectual'] = data['ActIntelectual'].astype(int)
    data['RelSociales'] = data['RelSociales'].astype(int)
    data['AntPsiquia'] = data['AntPsiquia'].astype(int)
    data['AntCardiologi'] = data['AntCardiologi'].astype(int)
    data['AntNeurolog'] = data['AntNeurolog'].astype(int)
    data['AntRenal'] = data['AntRenal'].astype(int)
    data['AntPulmonar'] = data['AntPulmonar'].astype(int)
    data['AntDemencia'] = data['AntDemencia'].astype(int)
    data['Tabaco'] = data['Tabaco'].astype(int)
    data['Alcohol'] = data['Alcohol'].astype(int)
    data['HTAanterior'] = data['HTAanterior'].astype(int)
    data['TAS'] = data['TAS'].astype(int)
    data['TAD'] = data['TAD'].astype(int)
    data['DM'] = data['DM'].astype(int)
    data['Barthel'] = data['Barthel'].astype(int)
    data['Hb'] = data['Hb'].astype(float)
    data['VCM'] = data['VCM'].astype(float)
    data['HCM'] = data['HCM'].astype(float)
    data['Plaquetas'] = data['Plaquetas'].astype(int)
    data['Leucocitos'] = data['Leucocitos'].astype(float)
    data['VCM'] = data['VCM'].astype(float)
    data['Neutrófilos'] = data['Neutrófilos'].astype(float)
    data['Linfocitos'] = data['Linfocitos'].astype(float)
    data['Monocitos'] = data['Monocitos'].astype(float)
    data['Glucosa'] = data['Glucosa'].astype(int)
    data['Creatinina'] = data['Creatinina'].astype(float)
    data['FG'] = data['FG'].astype(float)
    data['Na'] = data['Na'].astype(int)
    data['K'] = data['K'].astype(float)
    data['ALT'] = data['ALT'].astype(float)
    data['Colesterol'] = data['Colesterol'].astype(int)
    data['LDL'] = data['LDL'].astype(int)
    data['TNeurocog'] = data['TNeurocog'].astype(int)

    # Separate X and Y 
    X = data[features]
    y = data[[y_col]]    
     
    return data, X, y

def general_conversion (data : pd.DataFrame) -> pd.DataFrame :
    """Fix all Alzheimer-Balea database features data types to its original type.
    Categorical variables are set as "object" type. A DataFrame with the original 
    datatypes of this database is returned.

    Args:
    -----
            data: dataset with datatypes not corresponding to the original ones.
            features: list of strings containing the feature names of the dataset. 
            y_col: target variable (i.e., Y) name 

    Returns:
    --------
            data: dataframe with the original datatypes 
    """
    data['Edad'] = data['Edad'].astype(int)
    data['Sexo'] = data['Sexo'].astype('object')
    data['EstCivil'] = data['EstCivil'].astype('object')
    data['ActLaboral'] = data['ActLaboral'].astype('object')
    data['FormatConvivencia'] = data['FormatConvivencia'].astype('object')
    data['NivelEducativo'] = data['NivelEducativo'].astype('object')
    data['ActIntelectual'] = data['ActIntelectual'].astype('object')
    data['RelSociales'] = data['RelSociales'].astype('object')
    data['AntPsiquia'] = data['AntPsiquia'].astype('object')
    data['AntNeurolog'] = data['AntNeurolog'].astype('object')
    data['AntRenal'] = data['AntRenal'].astype('object')
    data['AntPulmonar'] = data['AntPulmonar'].astype('object')
    data['AntDemencia'] = data['AntDemencia'].astype('object')
    data['Tabaco'] = data['Tabaco'].astype('object')
    data['Alcohol'] = data['Alcohol'].astype('object')
    data['AntPsiquia'] = data['AntPsiquia'].astype('object')
    data['AntCardiologi'] = data['AntCardiologi'].astype('object')
    data['HTAanterior'] = data['HTAanterior'].astype(int)
    data['TAS'] = data['TAS'].astype(int)
    data['TAD'] = data['TAD'].astype(int)
    data['DM'] = data['DM'].astype(int)
    data['Barthel'] = data['Barthel'].astype(int)
    data['Hb'] = data['Hb'].astype(float)
    data['VCM'] = data['VCM'].astype(float)
    data['HCM'] = data['HCM'].astype(float)
    data['Plaquetas'] = data['Plaquetas'].astype(int)
    data['Leucocitos'] = data['Leucocitos'].astype(float)
    data['VCM'] = data['VCM'].astype(float)
    data['Neutrófilos'] = data['Neutrófilos'].astype(float)
    data['Linfocitos'] = data['Linfocitos'].astype(float)
    data['Monocitos'] = data['Monocitos'].astype(float)
    data['Glucosa'] = data['Glucosa'].astype(int)
    data['Creatinina'] = data['Creatinina'].astype(float)
    data['FG'] = data['FG'].astype(float)
    data['Na'] = data['Na'].astype(int)
    data['K'] = data['K'].astype(float)
    data['ALT'] = data['ALT'].astype(float)
    data['Colesterol'] = data['Colesterol'].astype(int)
    data['LDL'] = data['LDL'].astype(int)
    data['TNeurocog'] = data['TNeurocog'].astype(int)
    
    return data

def replacement(data : pd.DataFrame) -> pd.DataFrame :
    """This function replaces the numerical values corresponding to categories in 
    the Alzheimer-Balea database by its correspondant category. It returns a DataFrame
    after this replacement.

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    data['Sexo']  = data['Sexo'].replace([1,2],['Varon', 'Mujer'])
    data['EstCivil']  = data['EstCivil'].replace([1,2,3,4],['Solteria', 'Matrimonio', 
                                                                        'Viudedad', 'Divorcio'])
    data['ActLaboral']  = data['ActLaboral'].replace([1,2,3,4],['Cualificado', 'No cualificado', 
                                                                        'Empresario', 'Ama de casa'])
    data['FormatConvivencia']  = data['FormatConvivencia'].replace([1,2,3,4],['Convivencia Conyuge', 'Convivencia Solo', 
                                                                        'Convivencia Hijos_Familiares', 'Convivencia Residencia'])
    data['NivelEducativo']  = data['NivelEducativo'].replace([1,2,3,4],['Sin estudios', 'Estudios primarios', 
                                                                        'Estudios medios', 'Estudios universitarios'])
    data['ActIntelectual']  = data['ActIntelectual'].replace([1,2,3],['+10 libros/año', '5-10 libros/año', 
                                                                        'Lectura escasa'])
    data['RelSociales']  = data['RelSociales'].replace([1,2,3],['Relsociales Buenas', 'Relsociales Normales',
                                                                             'Relsociales Inexistentes'])
    data['AntPsiquia']  = data['AntPsiquia'].replace([1,2,3],['Psiquia Depresion', 'Psiquia Otros antecedentes', 'Psiquia Sin antecedentes'])
    data['AntCardiologi']  = data['AntCardiologi'].replace([1,2,3,4],['Cardio Isquemica', 'Cardio FA',
                                                                            'Cardio otras','Cardio Sin antecedentes'])
    data['AntNeurolog'] = data['AntNeurolog'].replace([1,2,3,4,5],['Neuro Ictus', 'Neuro Cefalea', 'Neuro Traumatismo',
                                                                            'Neuro Epilepsia','Neuro Sin antecedentes'])
    data['AntRenal'] = data['AntRenal'].replace([1,2,3],['Renal ERC', 'Renal Otras', 'Renal Sin Patologias'])
    data['AntPulmonar'] = data['AntPulmonar'].replace([1,2,3,4],['Pulmonar EPOC', 'Pulmonar Cancer', 
                                                                             'Pulmonar Otras','Pulmonar Sin Patologias'])
    data['AntDemencia'] = data['AntDemencia'].replace([1,2,3],['Demencia Padres', 'Demencia otros', 
                                                                             'Demencia Sin Antecedentes'])
    data['Tabaco'] = data['Tabaco'].replace([1,2],['Ex/Fumador', 'No Fumador'])
    data['Alcohol'] = data['Alcohol'].replace([1,2,3,4],['Alcohol Elevado', 'Alcohol Moderado',
                                                                           'Alcohol leve', 'Alcohol nunca'])
    data['HTAanterior'] = data['HTAanterior'].replace([1,2],[1,0]) 
    data['DM'] = data['DM'].replace([1,2],[1,0])
    
    return data 

def one_hot_enc(data : pd.DataFrame) -> pd.DataFrame :
    """This function performs One-Hot Encoding in the Alzheimer-Balea database. Since this
    database is really small, validation and train sets are even smaller. Hence, sometimes 
    columns full of 0s must be manually added, because a certain value of a feature does not 
    appear in the dataset subset. This is the case of category 'Empresarios' of feature "ActLaboral". 

    Args:
    -----
            data: dataset with categories represented by numbers.

    Returns:
    --------
            data: dataframe with the categories represented by their correspondant string  
    """
    
    # One-hot Encoder declaration 
    enc = OneHotEncoder(handle_unknown='ignore')
    # Sexo
    cats = ['Varon', 'Mujer']
    data[['Sexo']] = data[['Sexo']].astype('category')
    sexo = pd.DataFrame(enc.fit_transform(data[['Sexo']]).toarray())
    sexo.columns = enc.categories_
    for name in cats:
        if name not in sexo:
            sexo[name] = 0
    sexo = sexo[['Varon', 'Mujer']]
    sexo.reset_index(drop=True, inplace=True)
    # Estado Civil
    cats = ['Solteria', 'Matrimonio', 'Viudedad', 'Divorcio']
    data[['EstCivil']] = data[['EstCivil']].astype('category')
    civil = pd.DataFrame(enc.fit_transform(data[['EstCivil']]).toarray())
    civil.columns = enc.categories_
    for name in cats:
        if name not in civil:
            civil[name] = 0 
    civil = civil[['Solteria', 'Matrimonio', 'Viudedad', 'Divorcio']]        
    civil.reset_index(drop=True, inplace=True)
    # Actividad laboral
    cats = ['Cualificado', 'No cualificado','Empresario', 'Ama de casa']
    data[['ActLaboral']] = data[['ActLaboral']].astype('category')
    laboral = pd.DataFrame(enc.fit_transform(data[['ActLaboral']]).toarray())
    laboral.columns = enc.categories_
    for name in cats:
        if name not in laboral:
            laboral[name] = 0    
    laboral = laboral[['Cualificado', 'No cualificado','Empresario', 'Ama de casa']]
    laboral.reset_index(drop=True, inplace=True)
    # Formato Convivencia 
    cats = ['Convivencia Conyuge', 'Convivencia Solo','Convivencia Hijos_Familiares',
            'Convivencia Residencia']
    data[['FormatConvivencia']] = data[['FormatConvivencia']].astype('category')
    convi = pd.DataFrame(enc.fit_transform(data[['FormatConvivencia']]).toarray())
    convi.columns = enc.categories_
    for name in cats:
        if name not in convi:
            convi[name] = 0    
    convi = convi[['Convivencia Conyuge', 'Convivencia Solo','Convivencia Hijos_Familiares',
            'Convivencia Residencia']]
    convi.reset_index(drop=True, inplace=True)
    # Nivel educativo 
    cats = ['Sin estudios', 'Estudios primarios','Estudios medios', 'Estudios universitarios']
    data[['NivelEducativo']] = data[['NivelEducativo']].astype('category')
    educ = pd.DataFrame(enc.fit_transform(data[['NivelEducativo']]).toarray())
    educ.columns = enc.categories_
    for name in cats:
        if name not in educ:
            educ[name] = 0    
    educ = educ[['Sin estudios', 'Estudios primarios','Estudios medios', 'Estudios universitarios']]
    educ.reset_index(drop=True, inplace=True)
    # Acitividad intelectual  
    cats = ['+10 libros/año', '5-10 libros/año', 'Lectura escasa']
    data[['ActIntelectual']] = data[['ActIntelectual']].astype('category')
    intec = pd.DataFrame(enc.fit_transform(data[['ActIntelectual']]).toarray())
    intec.columns = enc.categories_
    for name in cats:
        if name not in intec:
            intec[name] = 0    
    intec = intec[['+10 libros/año', '5-10 libros/año', 'Lectura escasa']]
    intec.reset_index(drop=True, inplace=True)
    # Relaciones sociales
    cats = ['Relsociales Buenas', 'Relsociales Normales','Relsociales Inexistentes']
    data[['RelSociales']] = data[['RelSociales']].astype('category')
    rels = pd.DataFrame(enc.fit_transform(data[['RelSociales']]).toarray())
    rels.columns = enc.categories_
    for name in cats:
        if name not in rels:
            rels[name] = 0    
    rels = rels[['Relsociales Buenas', 'Relsociales Normales','Relsociales Inexistentes']]
    rels.reset_index(drop=True, inplace=True)
    # Psiquiatria 
    cats = ['Psiquia Depresion', 'Psiquia Otros antecedentes', 'Psiquia Sin antecedentes']
    data[['AntPsiquia']] = data[['AntPsiquia']].astype('category')
    psiq = pd.DataFrame(enc.fit_transform(data[['AntPsiquia']]).toarray())
    psiq.columns = enc.categories_
    for name in cats:
        if name not in psiq:
            psiq[name] = 0    
    psiq = psiq[['Psiquia Depresion', 'Psiquia Otros antecedentes', 'Psiquia Sin antecedentes']]
    psiq.reset_index(drop=True, inplace=True)
    # Cardiologia 
    cats = ['Cardio Isquemica', 'Cardio FA','Cardio otras','Cardio Sin antecedentes']
    data[['AntCardiologi']] = data[['AntCardiologi']].astype('category')
    card = pd.DataFrame(enc.fit_transform(data[['AntCardiologi']]).toarray())
    card.columns = enc.categories_
    for name in cats:
        if name not in card:
            card[name] = 0    
    card = card[['Cardio Isquemica', 'Cardio FA','Cardio otras','Cardio Sin antecedentes']]
    card.reset_index(drop=True, inplace=True)
    # Neurologia
    cats = ['Neuro Ictus', 'Neuro Cefalea', 'Neuro Traumatismo','Neuro Epilepsia','Neuro Sin antecedentes']
    data[['AntNeurolog']] = data[['AntNeurolog']].astype('category')
    neur = pd.DataFrame(enc.fit_transform(data[['AntNeurolog']]).toarray())
    neur.columns = enc.categories_
    for name in cats:
        if name not in neur:
            neur[name] = 0    
    neur = neur[['Neuro Ictus', 'Neuro Cefalea', 'Neuro Traumatismo','Neuro Epilepsia',
                 'Neuro Sin antecedentes']]
    neur.reset_index(drop=True, inplace=True)
    # Renal
    cats = ['Renal ERC', 'Renal Otras', 'Renal Sin Patologias']
    data[['AntRenal']] = data[['AntRenal']].astype('category')
    ren = pd.DataFrame(enc.fit_transform(data[['AntRenal']]).toarray())
    ren.columns = enc.categories_
    for name in cats:
        if name not in ren:
            ren[name] = 0    
    ren = ren[['Renal ERC', 'Renal Otras', 'Renal Sin Patologias']]
    ren.reset_index(drop=True, inplace=True)
    # Pulmonar
    cats = ['Pulmonar EPOC', 'Pulmonar Cancer','Pulmonar Otras','Pulmonar Sin Patologias']
    data[['AntRenal']] = data[['AntRenal']].astype('category')
    pulm = pd.DataFrame(enc.fit_transform(data[['AntPulmonar']]).toarray())
    pulm.columns = enc.categories_
    for name in cats:
        if name not in pulm:
            pulm[name] = 0    
    pulm = pulm[['Pulmonar EPOC', 'Pulmonar Cancer','Pulmonar Otras','Pulmonar Sin Patologias']]
    pulm.reset_index(drop=True, inplace=True)
    # Demencia 
    cats = ['Demencia Padres', 'Demencia otros', 'Demencia Sin Antecedentes']
    data[['AntDemencia']] = data[['AntDemencia']].astype('category')
    dem = pd.DataFrame(enc.fit_transform(data[['AntDemencia']]).toarray())
    dem.columns = enc.categories_
    for name in cats:
        if name not in dem:
            dem[name] = 0    
    dem = dem[['Demencia Padres', 'Demencia otros', 'Demencia Sin Antecedentes']]
    dem.reset_index(drop=True, inplace=True)
    # Tabaco 
    cats = ['Ex/Fumador', 'No Fumador']
    data[['Tabaco']] = data[['Tabaco']].astype('category')
    tab = pd.DataFrame(enc.fit_transform(data[['Tabaco']]).toarray())
    tab.columns = enc.categories_
    for name in cats:
        if name not in tab:
            tab[name] = 0    
    tab = tab[['Ex/Fumador', 'No Fumador']]
    tab.reset_index(drop=True, inplace=True)
    # Alcohol 
    cats = ['Alcohol Elevado', 'Alcohol Moderado','Alcohol leve', 'Alcohol nunca']
    data[['Alcohol']] = data[['Alcohol']].astype('category')
    alc = pd.DataFrame(enc.fit_transform(data[['Alcohol']]).toarray())
    alc.columns = enc.categories_
    for name in cats:
        if name not in alc:
            alc[name] = 0    
    alc = alc[['Alcohol Elevado', 'Alcohol Moderado','Alcohol leve', 'Alcohol nunca']]
    alc.reset_index(drop=True, inplace=True)
    
    # Drop column to add it at the end 
    affected = data[['TNeurocog']]
    affected.reset_index(drop=True, inplace=True)
    
    # Drop original categorical columns
    data = data.drop(['Sexo', 'EstCivil', 'ActLaboral','FormatConvivencia',
                                'NivelEducativo', 'ActIntelectual', 'RelSociales','AntPsiquia',
                                'AntCardiologi','AntNeurolog','AntRenal','AntPulmonar',
                                'AntDemencia','Tabaco','Alcohol','TNeurocog'], axis=1)
    data.reset_index(drop=True, inplace=True)
    
    data = pd.concat([data, sexo, civil, laboral,
                                  convi, educ, intec,
                                  rels, psiq, card, neur, ren,
                                  pulm, dem, tab, alc,
                                  affected],axis=1)
    return data

# Dictionary to specify fields of synthetic data for Alzheimer-Balea database
alz_fields = {
    'Edad' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Sexo' : {
        'type' : 'categorical'
    },
    'EstCivil' : {
        'type' : 'categorical'
    },
    'ActLaboral' : {
        'type' : 'categorical'
    },
    'FormatConvivencia' : {
        'type' : 'categorical'
    },
    'NivelEducativo' : {
        'type' : 'categorical'
    },
    'ActIntelectual' : {
        'type' : 'categorical'
    },
    'RelSociales' : {
        'type' : 'categorical'
    },
    'AntPsiquia' : {
        'type' : 'categorical'
    },
    'AntCardiologi' : {
        'type' : 'categorical'
    },
    'AntNeurolog' : {
        'type' : 'categorical'
    },
    'AntRenal' : {
        'type' : 'categorical'
    },
    'AntPulmonar' : {
        'type' : 'categorical'
    },
    'AntDemencia' : {
        'type' : 'categorical'
    },
    'Tabaco' : {
        'type' : 'categorical'
    },
    'Alcohol' : {
        'type' : 'categorical'
    },
    'HTAanterior' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'TAS' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'TAD' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    }, 
    'DM' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Barthel' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },
    'Hb' : {
        'type' : 'numerical',
        'subtype' : 'float'
    }, 
    'VCM' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'HCM' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Plaquetas' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Leucocitos' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Neutrófilos' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Linfocitos' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Monocitos' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Glucosa' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Creatinina' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'FG' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'Na' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'K' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'ALT' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'LDL' : {
        'type' : 'numerical',
        'subtype' : 'float'
    },
    'TNeurocog' : {
        'type' : 'numerical',
        'subtype' : 'integer'
    },   
 }

# Custom variable constraints to generate synthetic data 
constraints = [ 
                #Positive('Edad',handling_strategy='reject_sampling'),
                #Binary('HTAanterior',handling_strategy='reject_sampling'),
                #Positive('TAS',handling_strategy='reject_sampling'), 
                #Positive('TAD',handling_strategy='reject_sampling'),
                #Binary('DM',handling_strategy='reject_sampling'),
                # Positive('Barthel',handling_strategy='reject_sampling'),
                # Positive('Hb',handling_strategy='reject_sampling'),
                # Positive('VCM',handling_strategy='reject_sampling'),
                # Positive('HCM',handling_strategy='reject_sampling'),
                # Positive('Plaquetas',handling_strategy='reject_sampling'),
                # Positive('Leucocitos',handling_strategy='reject_sampling'),
                # Positive('Neutrófilos',handling_strategy='reject_sampling'),
                # Positive('Linfocitos',handling_strategy='reject_sampling'),
                # Positive('Monocitos',handling_strategy='reject_sampling'),
                # Positive('Glucosa',handling_strategy='reject_sampling'),
                # Positive('Creatinina',handling_strategy='reject_sampling'),
                # Positive('FG',handling_strategy='reject_sampling'),
                # Positive('Na',handling_strategy='reject_sampling'),
                # Positive('K',handling_strategy='reject_sampling'),
                # Positive('ALT',handling_strategy='reject_sampling'),
                # Positive('Colesterol',handling_strategy='reject_sampling'),
                # Positive('LDL',handling_strategy='reject_sampling'),
                #Binary('TNeurocog',handling_strategy='reject_sampling'),
                ]

# Distributions for each field (all set to univariate)
alz_distributions = {
    'Grupo' : 'univariate',
    'Edad' : 'univariate', 
    'EstCivil' : 'univariate',
    'Actlaboral' : 'univariate', 
    'FormatConvivencia' : 'univariate', 
    'NivelEducativo' : 'univariate', 
    'ActIntelectual' : 'univariate', 
    'RelSociales' : 'univariate', 
    'AntCardiologi' : 'univariate', 
    'AntNeurolog' : 'univariate',
    'AntRenal' : 'univariate', 
    'AntPulmonar' : 'univariate',
    'AntDemencia' : 'univariate',
    'Tabaco' : 'univariate',
    'Alcohol' : 'univariate',
    'HTAanterior' : 'univariate',
    'TAS' : 'univariate',
    'TAD' : 'univariate',
    'DM' : 'univariate',
    'Barthel' : 'univariate',
    'Hb' : 'univariate',
    'VCM' : 'univariate',
    'HCM' : 'univariate',
    'Plaquetas' : 'univariate',
    'Leucocitos' : 'univariate',
    'Neutrófilos' : 'univariate',
    'Linfocitos' : 'univariate',
    'Monocitos' : 'univariate',
    'Glucosa' : 'univariate',
    'Creatinina' : 'univariate',
    'FG' : 'univariate',
    'Na' : 'univariate',
    'K' : 'univariate',
    'ALT' : 'univariate',
    'Colesterol' : 'univariate',
    'LDL' : 'univariate',
    'TNeurocog' : 'univariate',   
    }  

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################

# Path where directories are stored
DICT_PATH = r"C:\Users\aralmeida\OneDrive - Universidad de Las Palmas de Gran Canaria\Doctorado\codigo\synthetic_data_generation_framework\AlzheimerBALEA\results"

# Dataset name 
dataset_name = 'ALZ-BALEA'

# Variables needed to handle dictionaries corresponding to the number of generated data samples  
sizes_keys = ["quarter", "half", "unit", "double", "quadruple", "only-synth"]

# Balancing Methods 
balance1 = "NC"
balance2 = "Borderline"

# Augmentation methods
augmen1 = "CTGAN"
augmen2 = "GC"

# Best and worst combinations
best_worst = ['Borderline + Sep. + GC', 'NC + CTGAN'] # might be wrong

# Best combination
best_method = 'Borderline + Sep. + GC' # might be wrong

# Name of the used Machine Learning models and associated colors 
models = ['SVM','RF', 'XGB', 'KNN']
model_colors = ['b','r','k','g']

# Studied metrics
mets = ["PCD","MMD","KLD"]

# Strings containing combinations of SDG (Synthetic Data Generators) 
comb1 = ("%s + %s") % (balance1, augmen1)
comb2 = ("%s + %s") % (balance1, augmen2)
comb3 = ("%s + %s") % (balance2, augmen1)
comb4 = ("%s + %s") % (balance2, augmen2)
comb5 = ("%s + Sep. + %s") % (balance1, augmen1)
comb6 = ("%s + Sep. + %s") % (balance1, augmen2)
comb7 = ("%s + Sep. + %s") % (balance2, augmen1)
comb8 = ("%s + Sep. + %s") % (balance2, augmen2)
comb9 = "%s" % (augmen1)
comb10 = "Sep. + %s" % (augmen1)

# Split CTGAN and Gaussian Copula methods to plot them separately
ctgan_combinations = [comb1, comb3, comb5, comb7]
gc_combinations = [comb2, comb4, comb6, comb8]

# Chosen colors for each synthetic data generation combinations
ctgan_colors = ["k","r","g","b"]
gc_colors = ["c","m","y","orange"]

################################################################################
#              CONSTANTS TO HANDLE/STORE/VISUALIZE OBTAINED RESULTS            #
################################################################################