#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import sklearn as sk
from fuzzywuzzy import process
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

# create X and y variables
filtered_df = pd.read_csv('career_skills_dataset_with_pattern.csv')
skillsDf = pd.read_csv('skill_descriptions.csv')
X = filtered_df.drop(columns=['Career','EncodeCareer'])
y = filtered_df['EncodeCareer']

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

resources = [
    ("Coursera", "https://www.coursera.org"),
    ("edX", "https://www.edx.org"),
    ("Khan Academy", "https://www.khanacademy.org"),
    ("MIT OpenCourseWare", "https://ocw.mit.edu/index.htm"),
    ("Codecademy", "https://www.codecademy.com"),
    ("freeCodeCamp", "https://www.freecodecamp.org"),
    ("CS50 by Harvard", "https://www.youtube.com/playlist?list=PLhQjrBD2T381-6J-fwP3yY8e3z4owdZkN"),
    ("Traversy Media", "https://www.youtube.com/user/TechGuyWeb"),
    ("The Odin Project", "https://www.theodinproject.com"),
    ("GeeksforGeeks", "https://www.geeksforgeeks.org")
]


# define skill search by career function 
st.title("Career Granny")
def skillsByCareer():
    global filtered_df
    global resources
    chosen_career = st.text_input("What career would you like to find skills for? ").strip().lower()
    best_match, score = process.extractOne(chosen_career, filtered_df['Career'].str.strip().values)
    if chosen_career:
        if score < 70:
            st.write("Career not found. Please try a different career.")
        else:
            chosen_career = best_match
            st.write(f"Skills and proficiencies required for {chosen_career}:")
            filtered_df = filtered_df.query('Career == @chosen_career')
            filtered_df = filtered_df.drop(columns=['Career','EncodeCareer'])
            top_skills = filtered_df.mean().nlargest(5)
            top_skills_df = top_skills.reset_index()
            top_skills_df.columns = ['Skill', 'Average Proficiency']
            top_skills_df.index = top_skills_df.index + 1
            st.dataframe(top_skills_df)
            st.title("Skills Breakdown")
            for skills in top_skills_df['Skill']:
                st.subheader(f"{skills}")
                st.write(skillsDf[skillsDf['Skill'] == skills]['Description'].values[0])
            st.title("Free Resources:")
            st.write("Here are some free resources to help you learn these skills through interactive courses, practice projects, and more:")
            
            for i, (title, link) in enumerate(resources, 1):
                st.write(f"{i}. [{title}]({link})")




# define career search by skills function
def careerBySkills():
    skill_set = []
    proficiency_set = []
    global filtered_df
    global resources
    skills = filtered_df.columns[:-2]
    st.write("Welcome to Career Granny.") 
    st.write("Choose your most proficient computer science skills.") 
    numSkills = st.number_input("How many skills would you like to enter? (The rest of the skills will be assumed as average or novice proficiency.)")
    if numSkills % 1 != 0 or numSkills < 1:
        st.write("Please enter an integer number of skills.")
    else:
        numSkills = int(numSkills)
        for i in range(numSkills):
            skill = st.text_input(f"Skill {i+1}: ").strip()
            if skill: 
                stripped_skills = [s.strip() for s in skills]
                best, score = process.extractOne(skill, stripped_skills)
                if score < 70:
                    st.write("Skill not found. Please try a different skill.")
                else:
                    print(best)
                    skill_set.append(best)
                    proficiency = st.slider(f"Proficiency {i+1}:", 1, 10, 1)
                    proficiency_set.append(proficiency)
        originalSkills = dict(zip(skill_set, proficiency_set))
        originalSkills_df = pd.DataFrame([originalSkills], columns=skill_set)
        if numSkills == 0 or len(skill_set)!=numSkills or len(proficiency_set)!=numSkills:
            st.write("Please enter the chosen number skills and proficiencies.")     
        else: 
            existing_skills_set = set([s.strip() for s in skills])
            for skill in existing_skills_set:
                if skill not in skill_set:
                    skill_set.append(skill)
                    proficiency_set.append(random.randint(1, 5))
            skills_with_prof = dict(zip(skill_set, proficiency_set))
            X_new_df = pd.DataFrame([skills_with_prof], columns=skills)
            prediction = dtree.predict(X_new_df)
        
            
            for value in filtered_df['EncodeCareer'].unique():
                if prediction[0] == value:
                    predicted_career = filtered_df[filtered_df['EncodeCareer'] == value]['Career'].values[0]
                    st.title(f"Predicted Career: {predicted_career}")
                
            filtered_df = filtered_df.query('Career == @predicted_career')
            filtered_df = filtered_df.drop(columns=['Career','EncodeCareer'])
            top_skills = filtered_df.mean().nlargest(5)
            top_skills_df = top_skills.reset_index()
            top_skills_df.columns = ['Skill', 'Average Proficiency']
            top_skills_df.index = top_skills_df.index + 1
            recoSkills = top_skills_df['Skill'].values
            recoProfs = top_skills_df['Average Proficiency'].values
            
            matching_skills = []
            for skill, prof in originalSkills.items():
                if skill in recoSkills:
                    skill_index = list(recoSkills).index(skill)
                    if prof >= recoProfs[skill_index] or prof >= recoProfs[skill_index] - 0.5:
                        matching_skills.append((skill, prof, recoProfs[skill_index]))

            if matching_skills:
                st.subheader("Your skills that match or exceed the recommended proficiency:")
                for skill, user_prof, reco_prof in matching_skills:
                    st.write(f"{skill}")
            st.subheader("Consider improving on the remaining skills and attaining atleast the proficiency shown for each.")
            st.dataframe(top_skills_df)
            st.title("Skills Breakdown")
            for skills in top_skills_df['Skill']:
                st.subheader(f"{skills}")
                st.write(skillsDf[skillsDf['Skill'] == skills]['Description'].values[0])
            st.subheader("Here are some free resources to help you learn these skills through interactive courses, practice projects, and more:")


            for i, (title, link) in enumerate(resources, 1):
                st.write(f"{i}. [{title}]({link})")
          
            

# define main function
def main():
    st.subheader(" to Career Granny!")
    st.write("Get started by selecting a feature below!")
    st.write("You can either search for skills required for a specific career or predict a career based on your skills.")
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

    st.title("Choose a Feature to Try!")

    # Handle button clicks and update the session state
    if st.button("Search skills by career"):
        st.session_state.selected_option = "Search skills by career"

    if st.button("Search career by skills"):
        st.session_state.selected_option = "Search career by skills"

    # Check the session state and display the corresponding content
    if st.session_state.selected_option == "Search skills by career":
        skillsByCareer()  # Call the function for this functionality

    elif st.session_state.selected_option == "Search career by skills":
        careerBySkills()  # Call the function for this functionality


if __name__ == "__main__":
    main()
