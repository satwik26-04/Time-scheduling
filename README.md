# **Student Task Scheduling and Wellness Assistant**

## **Project Overview**
This project aims to address the issue of intense student workloads by providing a comprehensive solution for task scheduling and wellness recommendations. Using machine learning models and association rule mining, the system predicts task completion times, prioritizes tasks, generates optimal study schedules, and suggests wellness activities to help students manage stress and maintain productivity.

## **Features**
- **Task Time Estimation**: Predicts the time required for tasks using a Random Forest Regression model based on task difficulty, type, and deadline.
- **Task Prioritization**: Classifies tasks into high, medium, or low priority using Random Forest Classification, considering task difficulty and subject priority.
- **Personalized Scheduling**: Generates a personalized schedule that accounts for routine blocks, user preferences, and break strategies (e.g., Pomodoro technique).
- **Wellness Monitoring**: Analyzes student data for indicators of high screen time, stress levels, and low grades to suggest mindfulness activities and learning techniques.
- **Association Rule Mining**: Uses the Apriori algorithm to discover patterns in student task data and generate insights using metrics like support, confidence, and lift.
- **Interactive GUI**: A user-friendly Tkinter interface allows students to input task details, view generated schedules, and receive wellness recommendations.

## **Technologies Used**
- **Python**: Programming language for implementing the core logic.
- **Scikit-Learn**: For building and training machine learning models (Random Forest).
- **Pandas**: Data manipulation and analysis.
- **mlxtend**: For association rule mining (Apriori algorithm).
- **Tkinter**: GUI library for creating the interactive user interface.

## **Usage**
1. Launch the application.
2. Enter task details including difficulty, type, deadline, and subject priority.
3. Click "Add Task" to include the task in the schedule.
4. Click "Generate Schedule" to view the personalized schedule and wellness recommendations.
5. Review the generated schedule and follow wellness suggestions to improve productivity.

## **Dataset**
The project uses a sample student task dataset with the following features:
- **Task Difficulty** (1-10 scale)
- **Task Type** (Reading, Writing, Problem-Solving)
- **Days Until Deadline**
- **Subject Priority** (1-10 scale)
- **Routine Blocks** (Morning and Night routines)
- **Break Preference** (Pomodoro, Long Breaks)
- **Task Completion Rate** (0-1 scale)
- **Grades** (0-100 scale)
- **Screen Time** (hours)
- **Stress Level** (1-10 scale)

## **Machine Learning Models**
- **Random Forest Regression**: Used for task time estimation.
- **Random Forest Classification**: Used for task priority prediction.
- **Association Rule Mining**: Uses the Apriori algorithm to discover meaningful patterns in student data.

## **Wellness and Learning Suggestions**
- **Screen Time Management**: Recommends reducing screen time if it exceeds 2 hours during study periods.
- **Stress Relief**: Suggests mindfulness activities like deep breathing or breaks if stress levels are high.
- **Learning Techniques**: Recommends techniques like active recall for subjects where grades are low.

## **Sample Output**
- **Predicted Task Time**: "85.14 minutes"
- **Predicted Task Priority**: "High"
- **Generated Schedule**: Displays a time-blocked schedule including study sessions and breaks.
- **Wellness Recommendations**: "Reduce screen time", "Take a stress-relief break", "Consider using active recall"

## **Contributing**
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Contributions for bug fixes, new features, and improvements are welcome.




