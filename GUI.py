import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv(r'C:/Users/navee/Documents/Python Project/Expanded_Sample_Student_Task_Dataset.csv')

numeric_features = ['Task_Difficulty', 'Days_Until_Deadline', 'Subject_Priority']
categorical_features = ['Task_Type']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

priority_encoder = LabelEncoder()

if 'Predicted_Priority' in data.columns:
    data['Predicted_Priority'] = priority_encoder.fit_transform(data['Predicted_Priority'])

X = data[['Task_Difficulty', 'Task_Type', 'Days_Until_Deadline', 'Subject_Priority']]
y_time = data['Predicted_Time'] if 'Predicted_Time' in data.columns else np.random.randint(30, 180, size=len(data))
y_priority = data['Predicted_Priority'] if 'Predicted_Priority' in data.columns else priority_encoder.fit_transform(np.random.choice(['High', 'Medium', 'Low'], size=len(data)))

time_model = RandomForestRegressor(n_estimators=100, random_state=42)
priority_model = RandomForestClassifier(n_estimators=100, random_state=42)

time_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', time_model)])
priority_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', priority_model)])

time_pipeline.fit(X, y_time)
priority_pipeline.fit(X, y_priority)

tasks = []

def add_task():
    try:
        task_difficulty = int(task_difficulty_var.get())
        task_type = task_type_var.get()
        days_until_deadline = int(deadline_var.get())
        subject_priority = int(subject_priority_var.get())
        task_completion_rate = float(completion_rate_var.get())
        grades = float(grades_var.get())
        screen_time = float(screen_time_var.get())
        stress_level = int(stress_level_var.get())
        break_preference = break_preference_var.get()

        user_input = pd.DataFrame({
            'Task_Difficulty': [task_difficulty],
            'Task_Type': [task_type],
            'Days_Until_Deadline': [days_until_deadline],
            'Subject_Priority': [subject_priority]
        })

        predicted_time = time_pipeline.predict(user_input)[0]
        predicted_priority = priority_encoder.inverse_transform(priority_pipeline.predict(user_input))[0]

        wellness_recommendations = []
        if screen_time > 2:
            wellness_recommendations.append("Reduce screen time.")
        if stress_level > 7:
            wellness_recommendations.append("Take a stress-relief break.")
        if grades < 60:
            wellness_recommendations.append("Consider using active recall.")

        task = {
            'Task_ID': f'User Task {len(tasks) + 1}',
            'Predicted_Time': predicted_time,
            'Predicted_Priority': predicted_priority,
            'Grades': grades,
            'Completion_Rate': task_completion_rate,
            'Break_Preference': break_preference,
            'Screen_Time': screen_time,
            'Stress_Level': stress_level,
            'Type': 'Task',
            'Wellness_Recommendations': ', '.join(wellness_recommendations) or 'None'
        }

        tasks.append(task)
        showinfo("Task Added", f"Task {len(tasks)} added successfully.")
    except ValueError as e:
        showinfo("Input Error", f"Invalid input: {e}")

def generate_schedule():
    try:
        task_df = pd.DataFrame(tasks)
        task_df['Priority_Score'] = priority_encoder.transform(task_df['Predicted_Priority'])
        task_df = task_df.sort_values(by=['Priority_Score', 'Predicted_Time']).reset_index(drop=True)

        schedule = []
        current_time = pd.to_datetime("08:00")

        for _, task in task_df.iterrows():
            adjusted_duration = 1.5 * task['Predicted_Time'] if task['Grades'] < 60 and task['Completion_Rate'] < 0.8 else task['Predicted_Time']
            task_start_time = current_time
            task_end_time = task_start_time + pd.Timedelta(minutes=adjusted_duration)

            schedule.append({
                'Task_ID': task['Task_ID'],
                'Start_Time': task_start_time.strftime("%H:%M"),
                'End_Time': task_end_time.strftime("%H:%M"),
                'Priority': task['Predicted_Priority'],
                'Type': task['Type']
            })

            if task['Break_Preference'] == 'Pomodoro' and adjusted_duration > 25:
                break_duration = 5
            elif task['Break_Preference'] == 'Long Breaks' and adjusted_duration > 60:
                break_duration = 15
            else:
                break_duration = 0

            if break_duration > 0:
                break_start_time = task_end_time
                break_end_time = break_start_time + pd.Timedelta(minutes=break_duration)
                schedule.append({
                    'Task_ID': 'Break',
                    'Start_Time': break_start_time.strftime("%H:%M"),
                    'End_Time': break_end_time.strftime("%H:%M"),
                    'Priority': 'None',
                    'Type': 'Break'
                })
                current_time = break_end_time
            else:
                current_time = task_end_time

        schedule_df = pd.DataFrame(schedule)
        wellness_recommendations = task_df['Wellness_Recommendations'].tolist()
        recommendations_text = '\n'.join(wellness_recommendations)

        results = (
            f"Generated Schedule:\n{schedule_df.to_string(index=False)}\n\n"
            f"Wellness Recommendations:\n{recommendations_text}"
        )
        showinfo("Model Results", results)
    except Exception as e:
        showinfo("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Task Scheduler & Wellness Assistant")
root.geometry("400x700")
root.configure(bg="#f7f7f7")

style = ttk.Style()
style.configure("TLabel", background="#f7f7f7", font=("Helvetica", 10))
style.configure("TButton", background="#4CAF50", foreground="white", font=("Helvetica", 10, "bold"))

task_difficulty_var = tk.StringVar()
task_type_var = tk.StringVar()
deadline_var = tk.StringVar()
subject_priority_var = tk.StringVar()
completion_rate_var = tk.StringVar()
grades_var = tk.StringVar()
screen_time_var = tk.StringVar()
stress_level_var = tk.StringVar()
break_preference_var = tk.StringVar()

fields = [
    ("Task Difficulty (1-10):", task_difficulty_var),
    ("Task Type (Reading/Writing/Problem-Solving):", task_type_var),
    ("Days Until Deadline:", deadline_var),
    ("Subject Priority (1-10):", subject_priority_var),
    ("Task Completion Rate (0-1):", completion_rate_var),
    ("Grades (out of 100):", grades_var),
    ("Screen Time (hours):", screen_time_var),
    ("Stress Level (1-10):", stress_level_var),
    ("Break Preference (Pomodoro/Long Breaks):", break_preference_var),
]

for label_text, var in fields:
    frame = ttk.Frame(root)
    frame.pack(fill="x", padx=20, pady=5)
    label = ttk.Label(frame, text=label_text)
    label.pack(side="left")
    entry = ttk.Entry(frame, textvariable=var)
    entry.pack(side="right", fill="x", expand=True)
style.configure("Custom.TButton", foreground="red")


add_button = ttk.Button(root, text="Add Task", style="Custom.TButton", command=add_task)
add_button.pack(pady=10)

generate_button = ttk.Button(root, text="Generate Schedule", style="Custom.TButton", command=generate_schedule)
generate_button.pack(pady=20)

root.mainloop()
