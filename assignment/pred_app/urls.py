from django.urls import path
from . import views

app_name = 'pred_app'

urlpatterns = [
    path('', views.TopView.as_view(), name='top'),
    path('home/', views.HomeView.as_view(), name='home'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('result/', views.ResultView.as_view(), name='result'),

]