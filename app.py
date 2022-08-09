import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from prediction import get_prediction, ordinal_encoder, label_encoder

with open('Models/model_XGB.pkl', 'rb') as f:
    model = pickle.load(f)
# model = pickle.load(r'Model/model_XGB.pkl')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚗🚧🚗", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_sex = ['Male','Female','Unknown']
options_education = ['Elementary school','Junior high school',  'High school', 'Above high school','Illiterate', 'Writing & reading','Unknown']
options_owner = ['Owner','Governmental', 'Organization', 'Other']
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_service_year = ['5-10yrs','2-5yrs', 'Above 10yr', 'Unknown', '1-2yr',
       'Below 1yr']
options_road_type = ['Tangent road with flat terrain',
 'Tangent road with mild grade and flat terrain',
 'Steep grade downward with mountainous terrain',
 'Tangent road with mountainous terrain and', 'Escarpments',
 'Steep grade upward with mountainous terrain', 'Sharp reverse curve',
 'Gentle horizontal curve' 'Tangent road with rolling terrain']
options_junc_type = ['No junction', 'Y Shape', 'Crossing', 'Other', 'Unknown', 'O Shape', 'T Shape'
 'X Shape']
options_road_surface = ['Asphalt roads','Earth roads', 
 'Asphalt roads with some distress', 'Gravel roads', 'Other']
options_road_condition =['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
options_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit']
options_weather_condition = ['Normal', 'Raining', 'Raining and Windy', 'Windy', 'Cloudy', 'Snow', 
 'Fog or mist', 'Other', 'Unknown']
options_collision_type = ['Vehicle with vehicle collision',
 'Collision with roadside-parked vehicles', 'Collision with animals',
 'Collision with roadside objects', 'Collision with pedestrians',
 'With Train', 'Rollover', 'Fall from vehicles',  'Other', 'Unknown']
options_vehicle_movement = ['Going straight', 'U-Turn', 'Waiting to go', 'Moving Backward', 'Reversing',
 'Turnover', 'Parked', 'Stopping', 'Getting off',
 'Overtaking','Entering a junction','Other','Unknown']
options_casualty_class = ['Driver or rider', 'Passenger', 'Pedestrian']
options_casualty_severity = ['Slight', 'Serious', 'Fatal']
options_causaly_map = {'Slight': 1, 'Serious': 2, 'Fatal': 3, 'Unknown': None}
options_pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
 'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
 'Walking along in carriageway, back to traffic',
 'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)',
 'Walking along in carriageway, facing traffic',
 'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle', 'Unknown or other']
features = ['day_of_week', 'driver_age', 'driver_sex', 'educational_level',
       'driving_experience', 'vehicle_type', 'vehicle_owner', 'service_year',
       'accident_area', 'lanes', 'road_allignment', 'junction_type',
       'surface_type', 'road_surface_conditions', 'light_condition',
       'weather_condition', 'collision_type', 'vehicles_involved',
       'casualties', 'vehicle_movement', 'casualty_class', 'casualty_sex',
       'casualty_age', 'casualty_severity', 'pedestrian_movement',
       'accident_cause', 'hour', 'minute']
# ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚗🚧🚗</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

       st.subheader("Enter the input for following features:")
       day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
       accident_time = st.time_input('Select Accident Time')
       hour = accident_time.hour
       minute = accident_time.minute
#  hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
#  minute = st.slider("Pickup Minute:",0,60,value=00,format="%d")
       driver_age = st.selectbox("Select Driver Age: ", options=options_age)
       driver_sex = st.radio("Select Driver Sex: ",options=options_sex)
       educational_level = st.selectbox("Select Educational Level: ",options=options_education)
       driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
       vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
       vehicle_owner = st.selectbox("Select Vehicle Owner: ", options=options_owner)
       service_year = st.selectbox("Select Vehicle Service Year: ", options=options_service_year)
       accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
       lanes = st.selectbox("Select Lanes: ", options=options_lanes)
       road_allignment = st.selectbox("Select Road Allignment: ", options=options_road_type)
       junction_type = st.selectbox("Select Junction Type: ", options=options_junc_type)
       surface_type = st.selectbox("Select Road Surface Type: ", options=options_road_surface)
       road_surface_conditions = st.selectbox("Select Road Surface Conditions: ", options=options_road_condition)
       light_condition = st.selectbox("Select Light Conditions: ", options=options_light_condition)
       weather_condition = st.selectbox("Select Weather Conditions: ", options=options_weather_condition)
       collision_type = st.selectbox("Select Collision Type: ", options=options_collision_type)
       vehicles_involved = st.slider("Pickup Hour: ", 1, 7, value=0, format="%d")
       casualties = st.slider("Hour of Accident: ", 1, 8, value=0, format="%d")
       vehicle_movement = st.selectbox("Select Vehicle Movement: ", options=options_vehicle_movement)
       casualty_class = st.selectbox("Select Casualty Class: ", options=options_casualty_class)
       casualty_sex = st.selectbox("Select Casualty Gender: ", options=options_sex)
       if casualty_sex == "Unknown":
              casualty_sex = None
       casualty_age = st.selectbox("Select Casualty Age: ", options=options_age)
       if casualty_age == "Unknown":
              casualty_age = None
       casualty_severity = st.selectbox("Select Casualty Severity: ", options=options_casualty_severity)
       # casualty_severity = options_causaly_map[casualty_severity]
       pedestrian_movement = st.selectbox("Select Pedestrian Movement: ", options=options_pedestrian_movement)
       accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
              
       submit = st.form_submit_button("Predict")


    if submit:
       day_of_week = ordinal_encoder(day_of_week, options_day)
       accident_cause = ordinal_encoder(accident_cause, options_cause)
       vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
       driver_age =  ordinal_encoder(driver_age, options_age)
       accident_area =  ordinal_encoder(accident_area, options_acc_area)
       driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
       lanes = ordinal_encoder(lanes, options_lanes)
       driver_sex = ordinal_encoder(driver_sex,options_sex)
       educational_level = ordinal_encoder(educational_level,options_education)
       vehicle_owner = ordinal_encoder(vehicle_owner,options_owner)
       service_year = ordinal_encoder(service_year,options_service_year)
       road_allignment = ordinal_encoder(road_allignment,options_road_type)
       junction_type = ordinal_encoder(junction_type,options_junc_type)
       surface_type = ordinal_encoder(surface_type,options_road_surface)
       road_surface_conditions = ordinal_encoder(road_surface_conditions,options_road_condition)
       light_condition = ordinal_encoder(light_condition,options_light_condition)
       weather_condition = ordinal_encoder(weather_condition,options_weather_condition)
       collision_type = ordinal_encoder(collision_type,options_collision_type)
       vehicle_movement = ordinal_encoder(vehicle_movement,options_vehicle_movement)
       casualty_class = ordinal_encoder(casualty_class,options_casualty_class)
       casualty_sex = ordinal_encoder(casualty_sex, options_sex)
       casualty_age = ordinal_encoder(casualty_age, options_age)
       casualty_severity = ordinal_encoder(casualty_severity, options_casualty_severity)
       pedestrian_movement = ordinal_encoder(pedestrian_movement, options_pedestrian_movement)

        

       data = np.array([day_of_week,driver_age,driver_sex,educational_level,driving_experience,vehicle_type,vehicle_owner,service_year
,accident_area,lanes,road_allignment,junction_type,surface_type,road_surface_conditions,light_condition
,weather_condition,collision_type,vehicles_involved,casualties,vehicle_movement,casualty_class,casualty_sex
,casualty_age,casualty_severity,pedestrian_movement,accident_cause,hour,minute]).reshape(1,-1)

       pred = get_prediction(data=data, model=model)
       if pred[0] == 1:
              severity = 'Slight'
       elif pred[0] == 2:
              severity = 'Serious'
       elif pred[0] == 3:
              severity = 'Fatal'
       else:
              severity = 'Unknown'

              
       st.markdown(f"The predicted severity is:<h3 style='color:blue;'> {severity}</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()