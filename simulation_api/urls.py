from django.urls import path
from . import views

urlpatterns = [
    path('planning/task/submit/', views.SubmitPlanningTaskView.as_view(), name='planning-task-submit'),
    path('simulation/control/', views.SimulationControlView.as_view(), name='simulation_control'),
]