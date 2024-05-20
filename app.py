import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import pickle
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from xgboost import XGBClassifier

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('Data/bg.jpeg')

# Streamlit app
def main():
    st.title("Data Analysis App")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        Data = pd.read_csv(uploaded_file)
        tab1, tab2,tab3, tab4 = st.tabs(["Raw Data","Descriptive statistics", "EDA",'Model'])
        with tab1:
            st.subheader("Raw Data")
            st.dataframe(Data)
        with tab2:
            st.subheader("Descriptive statistics")
            st.write(Data.describe())
        with tab3:
            cat_columns = st.multiselect('Select Categoricalvaribales',Data.columns,default=['accident_severity','first_road_class','light_conditions','weather_conditions','date','day_of_week',
                                                                                             'time','junction_detail','vehicle_manoeuvre','lsoa_of_accident_location','vehicle_type','sex_of_driver',
                                                                                            'road_surface_conditions', 'urban_or_rural_area', 'age_band_of_driver', 'vehicle_leaving_carriageway'])
            Data[cat_columns] = Data[cat_columns].astype('category') 
            num_cols = Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            st.subheader("Plots")
            option = st.selectbox(label='Plot Type',options=['Correlation plot','Frequency plot','Bar Plot','Box Plot','Scatter Plot'])
            if option == 'Correlation plot':
                plt.figure(figsize=(10, 7))
                sns.heatmap(Data[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
                st.pyplot(plt)
            elif option == 'Frequency plot':
                col = st.selectbox(label='Select column',options=num_cols)
                if col:
                    bins = st.slider('Select number of bins:', min_value=5, max_value=50, value=10)
                    plt.figure(figsize=(10, 6))
                    sns.histplot(Data[col], bins=bins, kde=False)
                    plt.title(f'Frequency Plot for {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    st.pyplot(plt)
                else:
                    st.warning('No category columns')
            elif option == 'Bar Plot':
                col1 = st.selectbox(label='Select X',options=cat_columns,index=1)
                col2 = st.selectbox(label='Select Hue',options=[None]+cat_columns)
                plt.figure(figsize=(10, 6))
                sns.countplot(data=Data, x=col1,hue=col2)
                plt.title(f'Bar Plot for {col1}')
                plt.ylabel('Count')
                st.pyplot(plt)            
            elif option == 'Box Plot':
                col1 = st.selectbox(label='Select X',options=cat_columns,index=1)
                col2 = st.selectbox(label='Select Y',options=num_cols)
                col3 = st.selectbox(label='Select Hue',options=[None]+cat_columns)
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=Data, x=col1, y=col2,hue=col3)
                plt.title(f'Box Plot of {col2} by {col1}')
                plt.ylabel(col2)
                plt.xlabel(col1)
                st.pyplot(plt)
            elif option == 'Scatter Plot':
                available_columns = Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                x_axis = st.selectbox('Select the X-axis:', available_columns)
                y_axis = st.selectbox('Select the Y-axis:', available_columns, index=1 if len(available_columns) > 1 else 0)
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=Data, x=x_axis, y=y_axis)
                plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
                st.pyplot(plt) 
        with tab4:
            result = {
                    'Label': ['Fatal', 'Serious', 'Slight', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                    'Precision': [0.05, 0.28, 0.80, None, 0.38, 0.68],
                    'Recall': [0.10, 0.29, 0.78, None, 0.39, 0.67],
                    'F1-score': [0.07, 0.29, 0.79, 0.6652, 0.38, 0.67],
                    'Support': [673, 10226, 37687,48586, 48586, 48586]
                }

            df = pd.DataFrame(result)
            st.subheader("Ensemble model summary")
            st.dataframe(df)
            
            with open('ensemble_model.pkl', 'rb') as f:
                ensemble_model = pickle.load(f,fix_imports=True)
            
            sample = {
                "number_of_vehicles": 0,
                "number_of_casualties": 0,
                "speed_limit": 0.0,
                "age_of_vehicle": 0,
                "first_road_class_1": 0.0,
                "first_road_class_2": 0.0,
                "first_road_class_3": 0.0,
                "first_road_class_4": 0.0,
                "first_road_class_5": 0.0,
                "first_road_class_6": 0.0,
                "light_conditions_1": 0.0,
                "light_conditions_4": 0.0,
                "light_conditions_5": 0.0,
                "light_conditions_6": 0.0,
                "light_conditions_7": 0.0,
                "weather_conditions_1": 0.0,
                "weather_conditions_2": 0.0,
                "weather_conditions_3": 0.0,
                "weather_conditions_4": 0.0,
                "weather_conditions_5": 0.0,
                "weather_conditions_6": 0.0,
                "weather_conditions_7": 0.0,
                "weather_conditions_8": 0.0,
                "weather_conditions_9": 0.0,
                "road_surface_conditions_1": 0.0,
                "road_surface_conditions_2": 0.0,
                "road_surface_conditions_3": 0.0,
                "road_surface_conditions_4": 0.0,
                "road_surface_conditions_5": 0.0,
                "road_surface_conditions_9": 0.0,
                "urban_or_rural_area_1": 0.0,
                "urban_or_rural_area_2": 0.0,
                "urban_or_rural_area_3": 0.0,
                "age_band_of_driver_1": 0.0,
                "age_band_of_driver_2": 0.0,
                "age_band_of_driver_3": 0.0,
                "age_band_of_driver_4": 0.0,
                "age_band_of_driver_5": 0.0,
                "age_band_of_driver_6": 0.0,
                "age_band_of_driver_7": 0.0,
                "age_band_of_driver_8": 0.0,
                "age_band_of_driver_9": 0.0,
                "age_band_of_driver_10": 0.0,
                "age_band_of_driver_11": 0.0,
                "vehicle_leaving_carriageway_0": 0.0,
                "vehicle_leaving_carriageway_1": 0.0,
                "vehicle_leaving_carriageway_2": 0.0,
                "vehicle_leaving_carriageway_3": 0.0,
                "vehicle_leaving_carriageway_4": 0.0,
                "vehicle_leaving_carriageway_5": 0.0,
                "vehicle_leaving_carriageway_6": 0.0,
                "vehicle_leaving_carriageway_7": 0.0,
                "vehicle_leaving_carriageway_8": 0.0,
                "vehicle_leaving_carriageway_9": 0.0
            }
            input_df = pd.DataFrame(sample, index=[0])

            st.subheader("Traffic Incident Data Input")
            
            first_road_class_map = {'A':1,'B':2,'C':3,'D':4,'M':5,'Unknown':6}
            road_surface_conditions_map ={
                                        'Dry': 1,
                                        'Wet or damp': 2,
                                        'Snow': 3,
                                        'Frost or ice': 4,
                                        'Flood over': 5,
                                        'unknown': 9
                                    }
            light_conditions_map = {
                            'Daylight': 1,
                            'Darkness - lights lit': 4,
                            'Darkness - lights unlit': 5,
                            'Darkness - no lighting': 6,
                            'Darkness - lighting unknown': 7,
                        }
            urban_or_rural_area_map = {'Urban':1,'Rural':2,'unknown':3}
            weather_conditions_map  = {
                            'Fine no high winds': 1,
                            'Raining no high winds': 2,
                            'Snowing no high winds': 3,
                            'Fine + high winds': 4,
                            'Raining + high winds': 5,
                            'Snowing + high winds': 6,
                            'Fog or mist': 7,
                            'Unknown': 9
                        }
            vehicle_leaving_cy_map = {
                            'None': 0,
                            'Hit Rd Sign/Ats': 1,
                            'Hit Lamp Post': 2,
                            'Hit Telegraph': 3,
                            'Hit Tree': 4,
                            'Hit Bus Stop': 5,
                            'Hit Central Bar': 6,
                            'Hit Nr/Off Bar': 7,
                            'Entered Ditch': 8,
                            'Hit Oth Object': 9
                        }
            age_range_map = {
                            "0-5": 1,
                            "5-10": 2,
                            "11-15": 3,
                            "16-20": 4,
                            "21-25": 5,
                            "26-35": 6,
                            "36-45": 7,
                            "46-55": 8,
                            "56-65": 9,
                            "66-75": 10,
                            "76+": 11,
                            "unknown": 12
                        }


            with st.form("input_form"):
                col1, col2, col3,col4 = st.columns(4)
                with col1:
                    number_of_vehicles = st.number_input('Number of vehicles', value=1, min_value=0)
                    speed_limit = st.number_input('Speed limit', value=20, min_value=0)
                    number_of_casualties = st.number_input('Number of casualties', value=1, min_value=0)

                with col2:
                    age_of_vehicle = st.number_input('Age of vehicle', value=1, min_value=0)
                    first_road_class = st.selectbox('First road class', options=first_road_class_map.keys())
                    light_conditions = st.selectbox('Light conditions', options=light_conditions_map.keys())

                with col3:
                    weather_conditions = st.selectbox('Weather conditions', options=weather_conditions_map.keys())
                    road_surface_conditions = st.selectbox('Road surface conditions', options=road_surface_conditions_map.keys())
                    urban_or_rural_area = st.selectbox('Urban or rural area', options=urban_or_rural_area_map.keys())

                with col4:
                    age_band_of_driver = st.selectbox('Age band of driver', options=age_range_map.keys())
                    vehicle_leaving_carriageway = st.selectbox('Vehicle leaving carriageway', options=vehicle_leaving_cy_map.keys())
                submitted = st.form_submit_button("Submit")


            if submitted:
                input_df['number_of_vehicles'] = number_of_vehicles
                input_df['number_of_casualties'] = number_of_casualties
                input_df['speed_limit'] = speed_limit
                input_df['age_of_vehicle'] = age_of_vehicle
                input_df[f'first_road_class_{first_road_class_map[first_road_class]}'] = 1
                input_df[f'light_conditions_{light_conditions_map[light_conditions]}'] = 1
                input_df[f'weather_conditions_{weather_conditions_map[weather_conditions]}'] = 1
                input_df[f'road_surface_conditions_{road_surface_conditions_map[road_surface_conditions]}'] = 1
                input_df[f'urban_or_rural_area_{urban_or_rural_area_map[urban_or_rural_area]}'] = 1
                input_df[f'age_band_of_driver_{age_range_map[age_band_of_driver]}'] = 1
                input_df[f'vehicle_leaving_carriageway_{vehicle_leaving_cy_map[vehicle_leaving_carriageway]}'] = 1
                
                ensemble_prediction = ensemble_model.predict(input_df)
                ans_maps = {0:'Fatal',1:'Serious',2:'Slight'}
                ans = 'The incident severity might be '+str(ans_maps[ensemble_prediction[0]])
                st.subheader(f':orange[{ans}]',divider='rainbow')

if __name__ == "__main__":
    main()
