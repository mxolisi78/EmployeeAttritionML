from django.shortcuts import render
from .forms import AttritionForm
from .ml.predict import predict_one


def home(request):
    if request.method == "POST":
        return predict(request)
    return render(request, "predictor/home.html", {"form": AttritionForm()})


def predict(request):
    """Handle submitted form → run model → show result."""
    if request.method != "POST":
        return home(request)

    form = AttritionForm(request.POST)
    if not form.is_valid():
        return render(request, "predictor/home.html", {"form": form})

    result = predict_one(form.cleaned_dict())

    return render(request, "predictor/result.html", {
        "result": result,
        "form_data": form.cleaned_data,
    })