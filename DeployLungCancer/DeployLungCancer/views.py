from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load the dataset
    lung_data = pd.read_csv(r"C:\Users\Maryum Urooj Ahmed\PycharmProjects\LungCancer\Dataset\survey lung cancer.csv")
    lung_data.GENDER = lung_data.GENDER.map({"M": 1, "F": 2})
    lung_data.LUNG_CANCER = lung_data.LUNG_CANCER.map({"YES": 1, "NO": 2})

    # Split data into features (X) and target (y)
    X = lung_data.iloc[:, 0:-1]
    y = lung_data.iloc[:, -1:]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    # Initialize and train the RandomForest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    # Get values from the form
    try:
        val1 = request.POST.get("n1")  # Gender (e.g., "M" or "F")
        val2 = int(request.POST.get("n2"))  # Age
        val3 = int(request.POST.get("n3"))  # Smoking
        val4 = int(request.POST.get("n4"))  # Yellow Fingers
        val5 = int(request.POST.get("n5"))  # Anxiety Level
        val6 = int(request.POST.get("n6"))  # Peer Pressure
        val7 = int(request.POST.get("n7"))  # Chronic Disease
        val8 = int(request.POST.get("n8"))  # Fatigue
        val9 = int(request.POST.get("n9"))  # Allergy
        val10 = int(request.POST.get("n10"))  # Wheezing
        val11 = int(request.POST.get("n11"))  # Alcohol Consumption
        val12 = int(request.POST.get("n12"))  # Coughing
        val13 = int(request.POST.get("n13"))  # Shortness of Breath
        val14 = int(request.POST.get("n14"))  # Swallowing Difficulty
        val15 = int(request.POST.get("n15"))  # Chest Pain Severity

        # Combine the inputs into a single row for prediction
        input_data = [[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15]]

        # Make predictions using the trained model
        prediction = rf_classifier.predict(input_data)[0]  # Get the first prediction

        # Map prediction to a readable result
        result1 = "You are at a Higher Risk of Developing Lung Cancer" if prediction == 2 else "You are at Lower Risk of Developing Lung Cancer"
    except Exception as e:
        result1 = f"Error: {e}"

    # Render the result page with the prediction
    return render(request, 'predict.html', {"result2": result1})
