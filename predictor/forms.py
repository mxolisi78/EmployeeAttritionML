"""
Form mirroring the model's input features.
Categorical fields use choices that match the training data exactly.
"""
from django import forms
from .ml.loader import get_feature_names


class AttritionForm(forms.Form):
    # ---- Numeric ----
    Age                       = forms.IntegerField(min_value=18, max_value=65, initial=30)
    DailyRate                 = forms.IntegerField(min_value=100, max_value=1500, initial=800)
    DistanceFromHome          = forms.IntegerField(min_value=1, max_value=30, initial=5)
    Education                 = forms.ChoiceField(choices=[(1,"1 - Below College"),(2,"2 - College"),(3,"3 - Bachelor"),(4,"4 - Master"),(5,"5 - Doctor")])
    EnvironmentSatisfaction   = forms.ChoiceField(choices=[(1,"1 - Low"),(2,"2 - Medium"),(3,"3 - High"),(4,"4 - Very High")])
    HourlyRate                = forms.IntegerField(min_value=30, max_value=100, initial=65)
    JobInvolvement            = forms.ChoiceField(choices=[(1,"1 - Low"),(2,"2 - Medium"),(3,"3 - High"),(4,"4 - Very High")])
    JobLevel                  = forms.ChoiceField(choices=[(1,"1"),(2,"2"),(3,"3"),(4,"4"),(5,"5")])
    JobSatisfaction           = forms.ChoiceField(choices=[(1,"1 - Low"),(2,"2 - Medium"),(3,"3 - High"),(4,"4 - Very High")])
    MonthlyIncome             = forms.IntegerField(min_value=1000, max_value=20000, initial=5000)
    MonthlyRate               = forms.IntegerField(min_value=2000, max_value=27000, initial=14000)
    NumCompaniesWorked        = forms.IntegerField(min_value=0, max_value=10, initial=2)
    PercentSalaryHike         = forms.IntegerField(min_value=11, max_value=25, initial=15)
    PerformanceRating         = forms.ChoiceField(choices=[(3,"3 - Excellent"),(4,"4 - Outstanding")])
    RelationshipSatisfaction  = forms.ChoiceField(choices=[(1,"1 - Low"),(2,"2 - Medium"),(3,"3 - High"),(4,"4 - Very High")])
    StockOptionLevel          = forms.ChoiceField(choices=[(0,"0"),(1,"1"),(2,"2"),(3,"3")])
    TotalWorkingYears         = forms.IntegerField(min_value=0, max_value=40, initial=10)
    TrainingTimesLastYear     = forms.IntegerField(min_value=0, max_value=6, initial=3)
    WorkLifeBalance           = forms.ChoiceField(choices=[(1,"1 - Bad"),(2,"2 - Good"),(3,"3 - Better"),(4,"4 - Best")])
    YearsAtCompany            = forms.IntegerField(min_value=0, max_value=40, initial=5)
    YearsInCurrentRole        = forms.IntegerField(min_value=0, max_value=18, initial=3)
    YearsSinceLastPromotion   = forms.IntegerField(min_value=0, max_value=15, initial=1)
    YearsWithCurrManager      = forms.IntegerField(min_value=0, max_value=17, initial=3)

    # ---- Categorical (will become one-hot) ----
    BusinessTravel = forms.ChoiceField(choices=[
        ("Travel_Rarely", "Travel Rarely"),
        ("Travel_Frequently", "Travel Frequently"),
        ("Non-Travel", "Non-Travel"),
    ])
    Department = forms.ChoiceField(choices=[
        ("Sales", "Sales"),
        ("Research & Development", "Research & Development"),
        ("Human Resources", "Human Resources"),
    ])
    EducationField = forms.ChoiceField(choices=[
        ("Life Sciences","Life Sciences"),
        ("Other","Other"),
        ("Medical","Medical"),
        ("Marketing","Marketing"),
        ("Technical Degree","Technical Degree"),
        ("Human Resources","Human Resources"),
    ])
    Gender = forms.ChoiceField(choices=[("Male","Male"),("Female","Female")])
    JobRole = forms.ChoiceField(choices=[
        ("Sales Executive","Sales Executive"),
        ("Research Scientist","Research Scientist"),
        ("Laboratory Technician","Laboratory Technician"),
        ("Manufacturing Director","Manufacturing Director"),
        ("Healthcare Representative","Healthcare Representative"),
        ("Manager","Manager"),
        ("Sales Representative","Sales Representative"),
        ("Research Director","Research Director"),
        ("Human Resources","Human Resources"),
    ])
    MaritalStatus = forms.ChoiceField(choices=[
        ("Single","Single"),("Married","Married"),("Divorced","Divorced"),
    ])
    OverTime = forms.ChoiceField(choices=[("Yes","Yes"),("No","No")])

    def cleaned_dict(self):
        """Return form data as a plain dict for the prediction pipeline."""
        return {k: v for k, v in self.cleaned_data.items()}