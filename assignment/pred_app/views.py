from re import I
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import TemplateView

import os
from glob import glob
from . import forms
from . import demand_pred as pred

from tensorflow.keras.models import load_model

class TopView(TemplateView):
    """ トップページ """
    template_name = 'pred_app/top.html'


class HomeView(LoginRequiredMixin, TemplateView):
    """ ホームページ """
    template_name = 'pred_app/home.html'


class ResultView(LoginRequiredMixin, TemplateView):
    """ 推論結果ページ """
    template_name = 'pred_app/result.html'

    def get_context_data(self, **kwargs):
        inference_value = pred.predict_power()
        context = super().get_context_data(**kwargs)
        context["result"] = inference_value

        return context


class LoginView(LoginView):
    """ ログインページ """
    form_class = forms.LoginForm 
    template_name = 'pred_app/login.html'


class LogoutView(LoginRequiredMixin, LogoutView):
    """ ログアウトページ """
    template_name = 'pred_app/login.html'