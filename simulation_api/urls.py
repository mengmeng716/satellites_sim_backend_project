from django.urls import path
from . import views

urlpatterns = [
    path('simulation/control/', views.SimulationControlView.as_view(), name='simulation_control'),
    path('simulation/benchmark/latest', views.BenchmarkLatestView.as_view(), name='simulation_benchmark_latest'),
    path('simulation/correlation/latest', views.CorrelationLatestView.as_view(), name='simulation_correlation_latest'),
    path('simulation/capacity/latest', views.CapacityLatestView.as_view(), name='simulation_capacity_latest'),
    path('data-generation/datasets', views.DataGenerationDatasetsView.as_view(), name='data_generation_datasets'),
    path('data-generation/jobs/<str:job_id>', views.DataGenerationJobStatusView.as_view(), name='data_generation_job_status'),
    path('data-generation/datasets/<str:dataset_id>/summary', views.DataGenerationDatasetSummaryView.as_view(), name='data_generation_dataset_summary'),
    path('data-generation/datasets/<str:dataset_id>/series', views.DataGenerationSeriesView.as_view(), name='data_generation_dataset_series'),
    path('data-generation/datasets/<str:dataset_id>/association', views.DataGenerationAssociationView.as_view(), name='data_generation_dataset_association'),
    path('data-generation/datasets/<str:dataset_id>/validation', views.DataGenerationValidationView.as_view(), name='data_generation_dataset_validation'),
]