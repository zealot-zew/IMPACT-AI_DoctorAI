import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import pickle


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings('ignore')


df = pd.read_csv('C:/Users/Gaurav Sharma/Desktop/ML hackathon/AI-ML-hackathon-24-main/AI-ML-hackathon-24-main/Training.csv')

x = df.iloc[:,:131]
y = df.iloc[:,131]

regressor = RandomForestRegressor(n_estimators=41, random_state=0, oob_score=True)


regressor.fit(x, y)

test_df = pd.read_csv('C:/Users/Gaurav Sharma/Desktop/ML hackathon/AI-ML-hackathon-24-main/AI-ML-hackathon-24-main/Testing.csv')


x_test = test_df.iloc[:,:131]
y_test = test_df.iloc[:,131]


disease_codes = {
    "Fungal infection": 1,
    "Allergy": 2,
    "GERD": 3,
    "Chronic cholestasis": 4,
    "Drug Reaction": 5,
    "Peptic ulcer diseae": 6,
    "AIDS": 7,
    "Diabetes": 8,
    "Gastroenteritis": 9,
    "Bronchial Asthma": 10,
    "Hypertension": 11,
    "Migraine": 12,
    "Cervical spondylosis": 13,
    "Paralysis (brain hemorrhage)": 14,
    "Jaundice": 15,
    "Malaria": 16,
    "Chicken pox": 17,
    "Dengue": 18,
    "Typhoid": 19,
    "hepatitis A": 20,
    "Hepatitis B": 21,
    "Hepatitis C": 22,
    "Hepatitis D": 23,
    "Hepatitis E": 24,
    "Alcoholic hepatitis": 25,
    "Tuberculosis": 26,
    "Common Cold": 27,
    "Pneumonia": 28,
    "Dimorphic hemmorhoids(piles)": 29,
    "Heart attack": 30,
    "Varicose veins": 31,
    "Hypothyroidism": 32,
    "Hyperthyroidism": 33,
    "Hypoglycemia": 34,
    "Osteoarthristis": 35,
    "Arthritis": 36,
    "(vertigo) Paroymsal Positional Vertigo": 37,
    "Acne": 38,
    "Urinary tract infection": 39,
    "Psoriasis": 40,
    "Impetigo": 41
}

# Using an explicit loop to iterate and replace (less efficient)
for index, row in enumerate(y_test):
    if row in disease_codes:  # Check if the disease name exists in the dictionary
        y_test.iloc[index] = disease_codes[row]
    else:
      y_test.iloc[index] = 0


y_pred = regressor.predict(x_test)

from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix

y_pred = y_pred.astype(int)
y_test_array = y_test.astype(int)

precision = precision_score(y_test_array, y_pred, average='weighted')

# prompt: get me the precision and accuracy also along with the complete matrix

from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix

# Calculate precision score
precision = precision_score(y_test_array, y_pred, average='weighted')

# Calculate accuracy score
accuracy = accuracy_score(y_test_array, y_pred)

# Generate classification report
report = classification_report(y_test_array, y_pred)

# Generate confusion matrix
matrix = confusion_matrix(y_test_array, y_pred)

print("Precision Score:", precision)
print("Accuracy Score:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", matrix)


tester=x_test.iloc[4]
# Reshape 'tester' to be 2-dimensional
tester_reshaped = tester.values.reshape(1, -1)



# Now make a prediction with the reshaped data
prediction_of_test = regressor.predict(tester_reshaped)

# Print the prediction
print(prediction_of_test)


def get_key(val):
    for i in val:
        value_aprox=i
    value_aprox=int(round(value_aprox))

    for key, value in disease_codes.items():
        if value_aprox == value:
            return key

    return "disease not present in database"

print(get_key(prediction_of_test))

# Assuming this list includes all the symptoms your model was trained on
all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
    'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
    'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness',
    'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine',
    'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
    'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    # Add all other symptoms here
]

print(len(all_symptoms))

"""Case 2"""

migraine_symptoms = ['headache','nausea','vomiting']

all_symptoms = {symptom: 0 for symptom in all_symptoms}

# Set the Migraine-associated symptoms to 1 (present)
for symptom in migraine_symptoms:
    all_symptoms[symptom] = 1

# Convert the dictionary to a DataFrame
test_df = pd.DataFrame([all_symptoms])

# Reshape the DataFrame to match the expected input shape for the model
test_sample = test_df.values.reshape(1, -1)


#prediction_of_test = regressor.predict(test_sample)

# Print the prediction


model_1=regressor
prediction_of_test = model_1.predict(test_sample)

#print(prediction_of_test)

print(get_key(prediction_of_test))

pickle.dump(model_1,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(get_key(model.predict(test_sample)))