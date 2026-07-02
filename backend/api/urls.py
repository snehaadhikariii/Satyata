
from django.urls import path
from . import views

urlpatterns = [
    path('analyze', views.AnalyzeView.as_view()),
    path('history', views.HistoryView.as_view()),
    path('health',  views.HealthView.as_view()),
]