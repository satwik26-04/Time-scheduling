import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score

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

X_train, X_test, y_time_train, y_time_test, y_priority_train, y_priority_test = train_test_split(
    X, y_time, y_priority, test_size=0.2, random_state=42
)

time_model = RandomForestRegressor(n_estimators=100, random_state=42)
priority_model = RandomForestClassifier(n_estimators=100, random_state=42)

time_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', time_model)])
priority_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', priority_model)])

time_pipeline.fit(X_train, y_time_train)
priority_pipeline.fit(X_train, y_priority_train)

predicted_times = time_pipeline.predict(X)
predicted_priorities = priority_encoder.inverse_transform(priority_pipeline.predict(X))

data['Predicted_Priority'] = predicted_priorities

print("Task Time Estimation RMSE:", mean_squared_error(y_time, predicted_times, squared=False))
print("Task Priority Classification Accuracy:", accuracy_score(y_priority, priority_encoder.transform(predicted_priorities)))

priority_mapping = {'High': 1, 'Medium': 2, 'Low': 3}
data['Priority_Score'] = priority_encoder.fit_transform(data['Predicted_Priority']) if 'Predicted_Priority' in data.columns else np.random.choice([0, 1, 2], size=len(data))

data = data.sort_values(by=['Priority_Score', 'Days_Until_Deadline']).reset_index(drop=True)

task_completion_rate = data['Task_Completion_Rate'].mean()
data['Adjusted_Time'] = data.apply(lambda row: 1.5 if row['Grades'] < 60 and task_completion_rate < 80 else 1, axis=1)

def schedule_tasks(data, routine_blocks, break_preference):
    schedule = []
    current_time = pd.to_datetime("08:00")
    routine_blocks = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in routine_blocks]

    for _, task in data.iterrows():
        adjusted_duration = task['Adjusted_Time'] * np.random.randint(30, 180)
        task_start_time = current_time

        for block_start, block_end in routine_blocks:
            if task_start_time >= block_start and task_start_time < block_end:
                task_start_time = block_end

        task_end_time = task_start_time + pd.Timedelta(minutes=adjusted_duration)
        schedule.append({'Task_ID': task['Task_ID'], 'Start_Time': task_start_time.time(), 'End_Time': task_end_time.time(), 'Priority': task['Predicted_Priority']})

        if break_preference == 'Pomodoro' and adjusted_duration > 25:
            break_duration = 5
        elif break_preference == 'Long Breaks' and adjusted_duration > 60:
            break_duration = 15
        else:
            break_duration = 0

        if break_duration > 0:
            break_start_time = task_end_time
            break_end_time = break_start_time + pd.Timedelta(minutes=break_duration)
            schedule.append({'Task_ID': f'{task["Task_ID"]} Break', 'Start_Time': break_start_time.time(), 'End_Time': break_end_time.time(), 'Type': 'Break'})
            current_time = break_end_time
        else:
            current_time = task_end_time

    return pd.DataFrame(schedule)

routine_blocks = [('07:00', '08:00'), ('21:00', '22:00')]
break_preference = 'Pomodoro'

schedule = schedule_tasks(data, routine_blocks, break_preference)

def wellness_and_learning_suggestions(row):
    suggestions = []
    if row['Screen_Time'] > 2:
        suggestions.append("Reduce screen time.")
    if row['Stress_Level'] > 7:
        suggestions.append("Take a stress-relief break.")
    if row['Grades'] < 60:
        suggestions.append("Consider using active recall.")
    return suggestions

data['Wellness_Recommendations'] = data.apply(wellness_and_learning_suggestions, axis=1)

print("\nFinal Organized Task List:")
print(data[['Task_ID', 'Predicted_Priority', 'Days_Until_Deadline', 'Grades', 'Adjusted_Time']].head())

print("\nGenerated Schedule:")
print(schedule.head())

print("\nWellness Recommendations:")
print(data[['Task_ID', 'Wellness_Recommendations']].head())
