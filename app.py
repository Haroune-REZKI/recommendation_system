import json
from flask import Flask, jsonify, render_template, request
from flask import jsonify
import pickle

import numpy as np 

app = Flask(__name__)
from recommendation_functions.udemy_recommender import recommend_udemy_courses
from recommendation_functions.coursera_recommender import recommend_coursera_courses
from recommendation_functions.youtube_recommender import recommend_youtube_courses


recommended_courses = recommend_udemy_courses('web dev', 10)


file_path = 'C:\\Users\\HR\\Downloads\\Emily.json'

try:
    with open(file_path, 'r') as file:
        json_content = file.read()
        print("json content", json_content)
        data = json.loads(json_content)
        print("daaaaaaaaaata", data)
        course_list = data["courses"]
        print(course_list)

        # Load JSON content from the text

        # Assuming 'course_list' contains course titles
        # Now you can pass 'course_list' to your recommendation functions
        recommendations_udemy = recommend_udemy_courses(course_list, 10)
        recommendations_coursera = recommend_coursera_courses(course_list, 10)
        recommendations_youtube = recommend_youtube_courses(course_list, 10)

        # Do something with the recommendations
        print("Udemy Recommendations:", recommendations_udemy)
        print("Coursera Recommendations:", recommendations_coursera)
        print("Youtube Recommendations:", recommendations_youtube)

except FileNotFoundError:
    print(f'The file {file_path} was not found.')
except json.JSONDecodeError as e:
    print(f'Error decoding JSON in file {file_path}: {e}')
except Exception as e:
    print(f'An error occurred: {e}')


@app.route('/')
def home():
    return render_template('profiles.html')

@app.route('/index')
def home1():
    name = request.args.get('name', 'name')
    return render_template('index.html',name=name)

@app.route('/myCourses', methods=['POST'])
def myCourses():
    if request.is_json:
        data = request.get_json()
        print('daaaaaaaaaaaaaaaaaata', data)
        # Retrieve variables from JSON data
        nameio = data.get('nameio')
        list = data.get('list')
        print(nameio)
        print('', list)
    else:

        return jsonify({'error': 'Invalid request format'}), 400
    
@app.route('/myCourses', methods=['GET'])
def myCourses2():
    return render_template('my_courses.html')


    

@app.route('/input')
def input_page():
    return render_template('input.html')
@app.route('/profiles')
def profiles_page():
    return render_template('profiles.html')
    
@app.route('/recommend', methods=['POST'])  

def get_recommendations():
    name = request.args.get('name', 'name')
    if request.is_json:
        data = request.get_json()
        title = data['title']  # Access title from JSON
    else:
        title = request.form['title']  # Access title from form data



    udemy_recommendations = recommend_udemy_courses(title, 10)
    udemy_recommendations = udemy_recommendations[udemy_recommendations['similarity_score'] > 0.5]
    udemy_recommendations = udemy_recommendations.to_dict('records')
    
    

    coursera_recommendations = recommend_coursera_courses(title, 10)
    coursera_recommendations = coursera_recommendations[coursera_recommendations['similarity_score'] > 0.4]
    coursera_recommendations = coursera_recommendations.to_dict('records')
    

    youtube_recommendations = recommend_youtube_courses(title, 10)
    youtube_recommendations = youtube_recommendations.to_dict('records')
    print(youtube_recommendations)

   


    


    name = request.args.get('name','name')
    print("==================================================================================",name)
    return render_template('input.html', udemy_recommendations=udemy_recommendations, coursera_recommendations= coursera_recommendations, youtube_recommendations=youtube_recommendations,nameio=name)


#---------model import-----------------------------------------------

my_model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        prediction = my_model.predict(data)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)  