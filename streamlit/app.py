import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import glob
from datetime import datetime
import os
import wget
import pathlib
pathlib.PosixPath=pathlib.WindowsPath
import sys
import pandas as pd
import warnings
import socket
import numpy as np
import csv
import matplotlib.pyplot as plt
#from food_recommender import FoodRecommendationSystem
from streamlit_echarts import st_echarts
from newfood_recommend import FoodRecommendationSystem,SVMModel,IoTCycleAnalysis



st.set_page_config(page_title="Food Detection", page_icon="🍝")




from ast import literal_eval
from collections import Counter
from tqdm.auto import tqdm

import warnings

pd.set_option('display.max_colwidth', 1000)
warnings.filterwarnings("ignore")
tqdm.pandas()

# Configurations
CFG_MODEL_PATH = "models/best(1).pt"
CFG_ENABLE_URL_DOWNLOAD = False
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"
# End of Configurations


data = {
    'Name': [
        'Aloo Gobi', 'Aloo Matar', 'Aloo Methi', 'Aloo Tikki', 'Apple', 'Bhindi Masala',
        'Biryani', 'Boiled Egg', 'Bread', 'Burger', 'Butter Chicken', 'Chai', 'Chicken Curry',
        'Chicken Tikka', 'Chicken Wings', 'Chole', 'Daal', 'French Fries', 'French Toast', 'Fried Egg',
        'Kadhi Pakora', 'Kheer', 'Lobia Curry', 'Omelette', 'Onion Pakora', 'Onion Rings', 'Palak Paneer',
        'Pancakes', 'Paratha', 'Rice', 'Roti', 'Samosa', 'Sandwich', 'Spring Rolls', 'Waffles', 'White Rice'
    ],

    'Calories': [
        150, 170, 160, 200, 95, 120, 320, 70, 80, 250, 350, 45, 220, 180, 280, 200, 160, 220, 200,
        90, 150, 220, 180, 150, 190, 210, 230, 220, 180, 130, 100, 180, 280, 160, 250, 150
    ]
}

food_calories = [(name, calorie) for name, calorie in zip(data['Name'], data['Calories'])]


num_items_detected_list = [] 


def bluetooth_client(weightperson):

    client=socket.socket(socket.AF_BLUETOOTH,socket.SOCK_STREAM,socket.BTPROTO_RFCOMM)
    client.connect(('B8:27:EB:93:70:AE',4))

    person_w=weightperson 
    client.send(str(person_w).encode('utf-8'))

    count = 0
    speedsum = 0
    try:
        while True:
    #        message="sent"
    #        client.send(message.encode('utf-8'))
            data=client.recv(1024)
            if not data:
                break
            ctime, totdist, speed, calsburned, num = data.decode('utf-8').split()
            count += 1
            speedsum += float(speed)
            speed = speedsum/count
            print("Received")
            if float(num) == 0:
                speed = speedsum/count
                print(f"{ctime} {totdist} {speed} {calsburned}")
                with open('cycle_data.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ctime, totdist, speed, calsburned])

    except OSError as e:
        pass

    client.close() 


def imageInput(model, src):
    output_text = ""  
    num_items_detected_list = []  
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader(
            "Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image',
                         use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join(
                'data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                
                pred = model(imgpath)
                pred.render()
                # save output to file
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)

            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)',
                         use_column_width='always')

            
            output_text += "\n"
            num_items_detected = 0
            for det in pred.pred[0]:
                label = model.names[int(det[-1])]
                prob = det[4]
                output_text += f"{label}\n"
                num_items_detected += 1
            output_text += f"Number of items detected: {num_items_detected}\n"
            num_items_detected_list.append(num_items_detected)
                
        
        with open("output.txt", "a") as text_file:
            text_file.write(output_text)
            
            
        
        with open("number_items_counter.txt", "a") as counter_file:
            for num_items in num_items_detected_list:
                counter_file.write(f"{num_items}\n")
    
    elif src == 'From example data.':
        
        imgpaths = glob.glob('data/example_images/*')
        imgpaths = [img_path for img_path in imgpaths if img_path.lower().endswith(('.png', '.jpg', '.jpeg'))]  
        if len(imgpaths) == 0:
            st.error('No images found, Please upload example images in data/example_images')
            return
        for image_file in imgpaths:
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(image_file)
                st.image(img, caption='Selected Image', use_column_width='always')
            with col2:
                with st.spinner(text="Predicting..."):
                    
                    pred = model(image_file)
                    pred.render()
                    
                    outputpath = os.path.join('data/outputs', os.path.basename(image_file))
                    for im in pred.ims:
                        im_base64 = Image.fromarray(im)
                        im_base64.save(outputpath)
                    
                    img_ = Image.open(outputpath)
                    st.image(img_, caption='Model Prediction(s)', use_column_width='always')
                    
                    
                    output_text += f"\n"
                    num_items_detected = 0
                    for det in pred.pred[0]:
                        label = model.names[int(det[-1])]
                        prob = det[4]
                        output_text += f"{label}\n"
                        num_items_detected += 1
                    output_text += f"Number of items detected: {num_items_detected}\n"
                    num_items_detected_list.append(num_items_detected)
                        
        
        with open("output.txt", "w") as text_file:
            text_file.write(output_text)
    
        
        with open("number_items_counter.txt", "w") as counter_file:
            for num_items in num_items_detected_list:
                counter_file.write(f"{num_items}\n")
                
        num_items_detected_list = []
        
    
    
    
    print(output_text)


        
def detected_food_items():
    df = pd.read_csv('indianrecipes.csv')

    columns_to_drop = ['RecipeIngredientParts', 'RecipeInstructions', 'RecipeId', 'Unnamed: 0']
    df.drop(columns=columns_to_drop, inplace=True)

    df.drop_duplicates(subset='Name', keep='first', inplace=True)

    df.to_csv('indianmodrecipes.csv', index=False)

    data = pd.read_csv('indianmodrecipes.csv')

    with open('output.txt', 'r') as file:
        items = [line.strip() for line in file]

    filtered_rows = []

    for item in items:
        filtered_rows.append(data[data['Name'] == item])

    newdata = pd.concat(filtered_rows, ignore_index=True)

    newdata_mod = newdata.assign(foodnumber=[i for i in range(len(newdata))])

    newdata_mod.to_csv('detectedfooditems.csv', index=False)
    
    df=pd.read_csv('detectedfooditems.csv')

    with open('number_items_counter.txt', 'r') as file:
        num_items_list = [int(line.strip()) for line in file]

    grouped_rows = []
    idx = 0

    for num_items in num_items_list:
        group_rows = df.iloc[idx:idx+num_items]
        combined_names = ', '.join(group_rows['Name'])
        combined_data = {
            'Name': combined_names,
            'CookTime': group_rows['CookTime'].sum(),
            'PrepTime': group_rows['PrepTime'].sum(),
            'TotalTime': group_rows['TotalTime'].sum(),
            'Calories': group_rows['Calories'].sum(),
            'FatContent': group_rows['FatContent'].sum(),
            'SaturatedFatContent': group_rows['SaturatedFatContent'].sum(),
            'CholesterolContent': group_rows['CholesterolContent'].sum(),
            'SodiumContent': group_rows['SodiumContent'].sum(),
            'CarbohydrateContent': group_rows['CarbohydrateContent'].sum(),
            'FiberContent': group_rows['FiberContent'].sum(),
            'SugarContent': group_rows['SugarContent'].sum(),
            'ProteinContent': group_rows['ProteinContent'].sum()
        }
        grouped_rows.append(combined_data)
        idx += num_items

    newdata_mod = pd.DataFrame(grouped_rows)
    newdata_mod['foodnumber']=[i for i in range(len(newdata_mod))]
    
    columns_to_drop = ['CookTime','PrepTime','TotalTime']
    newdata_mod.drop(columns=columns_to_drop, inplace=True)
    newdata_mod.to_csv('finaldetectedfooditems.csv', index=False)
    


nutrition_values=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']
all_columns=['Name','Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']


def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    
    st.title("Food Detection 🍝")
    
    df_users = pd.read_csv('combined_columns.csv')  
    
    def select_user():
        
        df_users = pd.read_csv('combined_columns.csv')  
        
        
        selected_user_id = st.selectbox('Select User ID:', ['Select an option'] + df_users['User'].tolist())
        
        
        if selected_user_id == 'Select an option':
            return None
        else:
            return selected_user_id
        
    selected_user_id=select_user()
    weightperson = st.number_input("Enter your weight", value=None, placeholder="Type a number...")
   
    #bluetooth_client(weightperson)
    
    if selected_user_id is None or weightperson==None:
        st.warning('Please ensure that user or weight is selected.')
    else:
        datasrc = st.radio("Select input source:", ['From example data.', 'Upload your own data.'])
        
        
        imageInput(loadmodel('cpu'), datasrc)
        detected_food_items()
        
        nutrition_values=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']

        
        st.subheader("Detected food items information: ")
        detected_food_items_df=pd.read_csv("finaldetectedfooditems.csv")
        st.dataframe(detected_food_items_df)
        
        st.text('Check whether uploaded foods are recommended by doctors')
        
        st.title("SVM Model Tester")
        
        

        data_df = pd.read_csv('finaldetectedfooditems.csv')
        X_test = data_df.drop(columns=['foodnumber','Name'])

        svm_model = SVMModel("PCASSS_model.pkl")
        predictions = svm_model.predict(X_test,detected_food_items_df)
        st.subheader("Predictions")
        st.write(predictions)
        
        
        food_recommendation_system = FoodRecommendationSystem("recommended.csv", "ratings.csv")


        
        st.title("Food Recommendation System")
        st.write("Welcome to the Food Recommendation System! Choose a tag to get personalized recommendations.")

        
        tag_options = ['Vegan', 'Less Time to Make', 'Low Calorie', 'Inexpensive', 'Low Cholesterol']
        selected_tag = st.sidebar.selectbox('Select a tag:', tag_options)
        
        letter=""
        sum_value=0
        
        def create_pie_chart_with_class(recommender_instance, nutrition_values, letter):
            
            if(letter=="A"):
                inst=recommender_instance.recommend_vegan()
                sum_value=inst[nutrition_values].sum()
            elif(letter=="B"):
                inst=recommender_instance.recommend_less_time()
                sum_value=inst[nutrition_values].sum()
            elif(letter=="C"):
                inst=recommender_instance.recommend_low_calorie()
                sum_value=inst[nutrition_values].sum()
            elif(letter=="D"):
                inst=recommender_instance.recommend_inexpensive()
                sum_value=inst[nutrition_values].sum()
            elif(letter=="E"):
                inst=recommender_instance.recommend_low_cholesterol()
                sum_value=inst[nutrition_values].sum()
                
                
            
                
            options = {
                "tooltip": {
                    "trigger": 'item',
                },
                "legend": {
                    "orient": 'vertical',
                    "left": 'left',
                },
                "series": [
                    {
                        "name": 'Access From',
                        "type": 'pie',
                        "radius": '90%',
                        "data": [{"value": nv, "name": nt} for nv, nt in zip(sum_value, nutrition_values)],
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": 'rgba(0, 0, 0, 0.5)'
                            }
                        },
                    }
                ],
            }
            st_echarts(options=options)
            

        def display_recommendations(tag):
            if tag == 'Vegan':
                st.subheader("Veg Meals Recommendations")
                letter="A"
                st.write(food_recommendation_system.recommend_vegan()[all_columns])
            elif tag == 'Less Time to Make':
                st.subheader("Meals with Less Time to Make Recommendations")
                letter="B"
                st.write(food_recommendation_system.recommend_less_time()[all_columns])
            elif tag == 'Low Calorie':
                st.subheader("Low Calorie Meals Recommendations")
                letter="C"
                st.write(food_recommendation_system.recommend_low_calorie()[all_columns])
            elif tag == 'Inexpensive':
                st.subheader("Inexpensive Meals Recommendations")
                letter="D"
                st.write(food_recommendation_system.recommend_inexpensive()[all_columns])
            elif tag == 'Low Cholesterol':
                st.subheader("Low Cholesterol Meals Recommendations")
                letter="E"
                st.write(food_recommendation_system.recommend_low_cholesterol()[all_columns])
                
            st.write('')
            create_pie_chart_with_class(food_recommendation_system,nutrition_values,letter)
            
        
        display_recommendations(selected_tag)
        
        
        
        
        filtered_data = df_users[df_users['User'] == selected_user_id].values
        st.title(f"IOT Data by User {int(filtered_data[0][1])}")
        st.write(f"Distance Travelled: {filtered_data[0][4]} km")
        st.write(f"Time Taken: {filtered_data[0][3]} s")
        st.write(f"Average Speed: {filtered_data[0][5]} km/h")
        st.write(f"Total Calories Burned: {filtered_data[0][6]} cal")
        
        totalcaloriesburned=float(round(filtered_data[0][6],1))
        totalcaloriesconsumed=float(round(detected_food_items_df['Calories'].sum(),1))
        
        
        

        st.title("Calories Comparison")
        
        selected_recipe = st.selectbox("Select a recipe:", detected_food_items_df['Name'].unique())
        
        selected_recipe_cal = (detected_food_items_df[detected_food_items_df['Name'] == selected_recipe].iloc[0])['Calories']
        

        if(selected_recipe):
            total_calories_graph_options = {
                "xAxis": {
                    "type": "category",
                    "data": ['Total Calories you chose', 'Total Calories you burn'],
                },
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": [
                            {"value":selected_recipe_cal, "itemStyle": {"color":["#33FF8D","#FF3333"][selected_recipe_cal>totalcaloriesburned]}},
                            {"value": totalcaloriesburned, "itemStyle": {"color": "#3339FF"}},
                            {"value": 600, "itemStyle": {"color": "gray"}}
                        ],
                        "type": "bar",
                    }
                ],
            }
            
            st_echarts(options=total_calories_graph_options, height=500)
            st.write(f"Total Calories Burned: {totalcaloriesburned}cal <br>Calories Consumed for selected dish: {selected_recipe_cal}cal",unsafe_allow_html=True)
            st.write(f"Difference: {abs(selected_recipe_cal-totalcaloriesburned)}cal")
        else:
            st.warning('Please ensure that recipe is selected for comparison.')
            
            
                
        analysis = IoTCycleAnalysis('combined_columns.csv')
        analysis.load_data()
        st.plotly_chart(analysis.display_histogram())
        st.subheader("Leaderboard")
        st.write(analysis.display_results())




@st.cache_data
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")

@st.cache_data
def loadmodel(device):
    model_path = "models/best(1).pt"  
    if not os.path.exists(model_path):
        st.error('Model not found at the specified path. Please make sure the model is available locally.')
        return None
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=model_path, force_reload=True, device=device)
    return model

if __name__ == '__main__':
    main()