import pandas as pd
import numpy as np
from ast import literal_eval
from collections import Counter
from tqdm.auto import tqdm
import warnings
import plotly.graph_objects as go
import csv


from surprise import Reader, Dataset, NMF, accuracy
from surprise.model_selection import train_test_split

from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

class FoodRecommendationSystem:
    def __init__(self, recommended_csv, ratings_csv):
        self.food_df = pd.read_csv(recommended_csv)
        self.ratings_df = pd.read_csv(ratings_csv)
        self.model = None
        self.trainset = None
        self.testset = None
        self.svm_model = None
        self.train_svm_model()  # Call train_svm_model during initialization
        self._setup()

    def _setup(self):
        # Set pandas display options
        pd.set_option('display.max_colwidth', 1000)
        warnings.filterwarnings("ignore")
        tqdm.pandas()

        # Initialize surprise Reader
        reader = Reader(rating_scale=(0, 5))
        data = Dataset.load_from_df(self.ratings_df[['foodnumber', 'User', 'Rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.25)

        # Non-negative Matrix Factorization
        self.model = NMF()
        self.model.fit(self.trainset)

        # Evaluate the model
        predictions = self.model.test(self.testset)
        mse = accuracy.mse(predictions)
        rmse = accuracy.rmse(predictions)

        # Count tags
        tags_count = Counter()
        self.food_df["tags"].progress_apply(lambda tags: tags_count.update(literal_eval(tags)))

        TIME_TAGS = [
            '15-minutes-or-less',
            '30-minutes-or-less',
            '60-minutes-or-less',
            '4-hours-or-less',
        ]
        VEGAN_TAGS = ['vegan']
        MEAT_TAGS =[
            'beef',
            'chicken',
        ]

        CAL_TAGS = ['low-calorie']
        COL_TAGS = ['low-cholesterol']
        SOD_TAGS = ['low-sodium']
        PRICE_TAGS = ['inexpensive']

        FEATURE_COLS = TIME_TAGS+VEGAN_TAGS+MEAT_TAGS+CAL_TAGS+COL_TAGS+SOD_TAGS+PRICE_TAGS
        

        # Function to extract feature tags
        def fe_tags(food_tags):
            values = []
            
            for group_tag in [TIME_TAGS, VEGAN_TAGS, CAL_TAGS, COL_TAGS, SOD_TAGS, PRICE_TAGS]:
                for tag in group_tag:
                    values.append(True) if tag in food_tags else values.append(False)

            for tag in MEAT_TAGS:
                values.append(True) if tag in food_tags else values.append(False)

            return values    

        # Apply feature tags to dataframe
        self.food_df['tmp'] = self.food_df["tags"].progress_apply(lambda food_tags: fe_tags(food_tags))
        self.food_df[FEATURE_COLS] = pd.DataFrame(self.food_df['tmp'].tolist(), index=self.food_df.index)
        self.food_df.drop(columns='tmp', inplace=True)

        # Assign time values based on tags
        conds = [
            (self.food_df['4-hours-or-less']),
            (self.food_df['60-minutes-or-less']),
            (self.food_df['30-minutes-or-less']),
            (self.food_df['15-minutes-or-less']),
        ]
        choices = [4,3,2,1]
        self.food_df['time'] = np.select(conds, choices, default=5)
        self.food_df["time"].value_counts()
        

    def recommend_meal(self, model, uid, filtered_ids, topk):
        preds = []
        for iid in filtered_ids:
            pred_rating = model.predict(uid=uid, iid=iid).est
            preds.append([iid, pred_rating])
        preds.sort(key=lambda x:x[1], reverse=True)
        
        return preds[:topk]

    def recommend_vegan(self):
        filtered_ids = self.food_df[(self.food_df['vegan'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No vegan meals recommended."
        else:
            return recommended_df

    def recommend_chicken(self):
        filtered_ids = self.food_df[(self.food_df['chicken'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent'] 
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No chicken meals recommended."
        else:
            return recommended_df

    def recommend_less_time(self):
        filtered_ids = self.food_df[(self.food_df['time']<=1)]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No meals with less time to make recommended."
        else:
            return recommended_df
        
    def recommend_low_calorie(self):
        filtered_ids = self.food_df[(self.food_df['low-calorie'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No low calorie meals recommended."
        else:
            return recommended_df
        
    def recommend_inexpensive(self):
        filtered_ids = self.food_df[(self.food_df['inexpensive'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No meals that are inexpensive to make recommended."
        else:
            return recommended_df
        
    def recommend_low_cholesterol(self):
        filtered_ids = self.food_df[(self.food_df['low-cholesterol'])]['foodnumber'].to_list()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=10)
        selected_columns=['Name','Calories','SaturatedFatContent','CholesterolContent','FiberContent']
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        if recommended_df.empty:
            return "No meals with less cholesterol to make recommended."
        else:
            return recommended_df

    def train_svm_model(self):
        X = self.food_df.drop(['recommended', 'foodnumber', 'Name', 'tags'], axis=1)
        y = self.food_df['recommended']
        X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=101)
        parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        svm_clf = GridSearchCV(svm.SVC(), parameters, refit=True, verbose=2)
        svm_clf.fit(X_train, y_train)
        self.svm_model = svm_clf

    def get_svm_predictions(self, input_data):
        predictions = self.svm_model.predict(input_data)
        return predictions

    def recommend_based_on_svm(self, input_data, topk=10):
        predictions = self.get_svm_predictions(input_data)
        recommended_ids = [idx for idx, pred in enumerate(predictions) if pred == 0]  # Filter out recommended ones
        filtered_ids = self.food_df.loc[self.food_df.index.isin(recommended_ids), 'foodnumber'].tolist()
        random_user = self.ratings_df['User'].sample(1).values[0]
        filtered_df = self.food_df[self.food_df['recommended'] == 0]
        preds = self.recommend_meal(self.model, uid=random_user, filtered_ids=filtered_ids, topk=topk)
        recommended_df = filtered_df[filtered_df['foodnumber'].isin([x[0] for x in preds])]
        selected_columns=['Name','Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent','CarbohydrateContent','FiberContent','SugarContent','ProteinContent']
        if recommended_df.empty:
            return "No meals recommended based on SVM."
        else:
            return recommended_df[selected_columns]
        
    
class SVMModel:
    def __init__(self, model_path):
        # Load the SVM model from the pickle file
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, input_data, detected_food_items_df):
        # Make predictions using the loaded SVM model
        predictions = self.model.predict(input_data)
        detected_food_items_df['recommended'] = np.array(predictions)
        def map_recommendation(value):
            if value == 1:
                return "Yes"
            else:
                return "No"
        detected_food_items_df['recommended'] = detected_food_items_df['recommended'].map(map_recommendation)
        return detected_food_items_df 
    
class IoTCycleAnalysis:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None

    def load_data(self):
        with open('ratings.csv', 'r', newline='') as f1, open('cycle_data.csv', 'r', newline='') as f2, open(self.data_file, 'w', newline='') as output:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)

            writer = csv.writer(output)

            for row1, row2 in zip(reader1, reader2):
                combined_row = row1 + row2
                writer.writerow(combined_row)

    
        self.df = pd.read_csv(self.data_file)

    def display_bargraph(self):
        fig = go.Figure()
        for user_id, calories_burned in self.df.groupby('User')['Total Calories Burned (cal)']:
            fig.add_trace(go.Bar(x=[f'User {user_id}'], y=[calories_burned.sum()], name=f'User {user_id}', width=0.5))

        # Add threshold line for recommended calorie value
        fig.update_layout(
            title='Bar Graph of Total Calories Burned by User',
            xaxis_title='User',
            yaxis_title='Total Calories Burned (cal)',
            bargap=1,
            shapes=[
                dict(
                    type='line',
                    x0=-0.5,
                    y0=600,
                    x1=len(self.df) - 0.5,
                    y1=600,
                    line=dict(color='red', width=2),
                )
            ]
        )
        return fig
    
    
    def display_histogram(self):
        calories_burned = self.df['Total Calories Burned (cal)']
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=calories_burned, marker_color='skyblue', opacity=0.75))
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='y',
                    x0=600,
                    y0=0,
                    x1=600,
                    y1=30,
                    line=dict(color='red', width=2)
                )
            ]
        )
        fig.add_annotation(
            x=600,
            y=30,
            text='Recommended: 600 Calories to burn in one cycling activity',
            showarrow=True,
            arrowhead=1,
            arrowcolor='white',
            arrowwidth=2,
            ax=20,
            ay=-30,
            opacity=0.8
        )
        fig.update_layout(
            title='Histogram of Total Calories Burned',
            xaxis_title='Total Calories Burned (cal)',
            yaxis_title='Frequency',
            bargap=0.05
        )
        
        return fig


    def display_results(self):
        df_sorted = self.df.sort_values(by='Total Calories Burned (cal)', ascending=False)
        df_sorted['Rank'] = range(1, len(df_sorted) + 1)
        top_10_users = df_sorted.head(10)
        top_10_users = top_10_users.drop(columns=['foodnumber', 'Rating'])
        return top_10_users