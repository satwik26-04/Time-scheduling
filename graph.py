import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (use raw string or forward slashes for the file path)
data = pd.read_csv(r'C:/Users/navee/Documents/Python Project/Expanded_Sample_Student_Task_Dataset.csv')

# Set up the plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Task Difficulty Distribution
def plot_task_difficulty_distribution(data):
    plt.figure()
    sns.histplot(data['Task_Difficulty'], kde=True, color='skyblue')
    plt.title('Task Difficulty Distribution')
    plt.xlabel('Task Difficulty (1-10)')
    plt.ylabel('Frequency')
    plt.show()

# 2. Task Type Count
def plot_task_type_count(data):
    plt.figure()
    sns.countplot(data=data, x='Task_Type', hue='Task_Type', palette='viridis', legend=False)
    plt.title('Task Type Count')
    plt.xlabel('Task Type')
    plt.ylabel('Count')
    plt.show()

# 3. Task Completion Rate Distribution
def plot_task_completion_rate(data):
    plt.figure()
    sns.boxplot(data=data, y='Task_Completion_Rate', color='lightgreen')
    plt.title('Task Completion Rate Distribution')
    plt.ylabel('Completion Rate (0-1)')
    plt.show()

# 4. Correlation Heatmap
def plot_correlation_heatmap(data):
    plt.figure()
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()

# 5. Grades vs. Predicted Priority
def plot_grades_vs_priority(data):
    plt.figure()
    if 'Predicted_Priority' in data.columns:
        sns.violinplot(data=data, x='Predicted_Priority', y='Grades', palette='muted')
        plt.title('Grades vs. Predicted Priority')
        plt.xlabel('Predicted Priority')
        plt.ylabel('Grades (0-100)')
        plt.show()
    else:
        print("Warning: 'Predicted_Priority' column not found in the dataset. Skipping this plot.")

# 6. Screen Time vs. Stress Level (Improved)
def plot_screen_time_vs_stress(data):
    plt.figure()

    # Check for missing values and drop them
    if data[['Screen_Time', 'Stress_Level']].isnull().any().any():
        print("Missing values found in 'Screen_Time' or 'Stress_Level'. Dropping missing values.")
        data = data.dropna(subset=['Screen_Time', 'Stress_Level'])

    # Ensure data types are numeric
    data['Screen_Time'] = pd.to_numeric(data['Screen_Time'], errors='coerce')
    data['Stress_Level'] = pd.to_numeric(data['Stress_Level'], errors='coerce')

    # Drop rows with invalid (NaN) values after type conversion
    data = data.dropna(subset=['Screen_Time', 'Stress_Level'])

    # Check if 'Predicted_Priority' exists and has valid categories
    if 'Predicted_Priority' in data.columns and not data['Predicted_Priority'].isnull().all():
        sns.scatterplot(data=data, x='Screen_Time', y='Stress_Level', hue='Predicted_Priority', palette='deep')
        plt.title('Screen Time vs. Stress Level (With Priority)')
    else:
        sns.scatterplot(data=data, x='Screen_Time', y='Stress_Level', color='blue')
        plt.title('Screen Time vs. Stress Level (No Priority Data)')

    plt.xlabel('Screen Time (hours)')
    plt.ylabel('Stress Level (1-10)')
    plt.show()


# Main function to generate all plots
def generate_all_plots(data):
    print("Generating data visualizations...")
    plot_task_difficulty_distribution(data)
    plot_task_type_count(data)
    plot_task_completion_rate(data)
    plot_correlation_heatmap(data)
    plot_grades_vs_priority(data)
    plot_screen_time_vs_stress(data)

# Call the main function
generate_all_plots(data)
